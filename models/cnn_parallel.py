import torch
from torch import nn

class SideAdapter(nn.Module):
    """ 每一层新增的平行路径 (类似 LoRA 的下投影+上投影) """
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        mid_channels = max(in_channels // 4, 16) 
        self.adapter = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False), #TODO: 算一算
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        # nn.init.zeros_(self.adapter[0].weight)
        nn.init.zeros_(self.adapter[3].weight)

    def forward(self, x):
        return self.adapter(x)

class ParallelBaseline(nn.Module):
    def __init__(self, base_model, **kwargs):
        super().__init__()
        self.base_model = base_model
        self.adapters = nn.ModuleDict()
        
        for name, module in self.base_model.named_modules():
            if isinstance(module, nn.Conv2d) and 'conv' in name:
                # 每层创建一个对应的Adapter
                adapter = SideAdapter(module.in_channels, module.out_channels, module.stride)
                self.adapters[name.replace('.', '_')] = adapter

        # hook实现平行计算
        for name, module in self.base_model.named_modules():
            if isinstance(module, nn.Conv2d) and 'conv' in name:
                key = name.replace('.', '_')
                module.register_forward_hook(self._get_hook(key))

    def _get_hook(self, key):
        def hook(module, input, output):
            side_output = self.adapters[key](input[0])
            return output + side_output
        return hook

    def forward(self, x):
        return self.base_model(x)

def replace_module(root_model, target_name, new_module):
    names = target_name.split('.')
    for name in names[:-1]:
        root_model = getattr(root_model, name)
    setattr(root_model, names[-1], new_module)

def expand_resnet_with_zero_init(model, checkpoint_model, n=1, grad_allow=5):
    """
    针对 ResNet 进行手术：
    1. 增加卷积层和 BN 层的通道数 (in_channels 或 out_channels)
    2. 从 checkpoint 加载旧权重到新层的起始位置，新增部分填 0
    """
    def get_mask_hook(mask):
        def hook(grad):
            return grad * mask.to(grad.device)
        return hook
    for name, module in model.named_modules():
        # 卷积层
        if isinstance(module, nn.Conv2d):
            # 除了第一层输入(in=3)外，所有输入输出均扩容
            in_ext = n if module.in_channels > 3 else 0
            out_ext = n
            
            new_conv = nn.Conv2d(
                in_channels=module.in_channels + in_ext,
                out_channels=module.out_channels + out_ext,
                kernel_size=module.kernel_size,
                stride=module.stride,
                padding=module.padding,
                dilation=module.dilation,
                groups=module.groups,
                bias=(module.bias is not None)
            ).to(module.weight.device)
            
            # 创建Mask: 旧区域为0，新通道区域为1
            # weight shape = $(out_ch, in_ch, k, k)
            mask = torch.zeros_like(new_conv.weight)
            mask[module.out_channels-grad_allow:, :, :, :] = 1.0 
            mask[:, module.in_channels-grad_allow:, :, :] = 1.0

            # 权重缝合
            if name + '.weight' in checkpoint_model:
                old_w = checkpoint_model[name + '.weight']
                with torch.no_grad():
                    new_conv.weight.zero_() # 全初始为 0
                    # 拷贝旧权重到左上角 [old_out, old_in, k, k]
                    new_conv.weight[:old_w.shape[0], :old_w.shape[1], :, :].copy_(old_w)
                    if module.bias is not None:
                        new_conv.bias[:module.bias.shape[0]].copy_(checkpoint_model[name + '.bias'])
            
            # 注册hook
            new_conv.weight.register_hook(get_mask_hook(mask))

            replace_module(model, name, new_conv)

        # 处理 BN 层
        elif isinstance(module, nn.BatchNorm2d):
            new_bn = nn.BatchNorm2d(
                num_features=module.num_features + n,
                eps=module.eps,
                momentum=module.momentum
            ).to(module.weight.device)
            
            mask_w = torch.zeros_like(new_bn.weight)
            mask_w[module.num_features-grad_allow:] = 1.0 # 只准更新最后一个通道的 scale

            if name + '.weight' in checkpoint_model:
                with torch.no_grad():
                    new_bn.weight.zero_() # 缩放因子初始为0
                    new_bn.weight[:module.num_features].copy_(checkpoint_model[name + '.weight'])
                    
                    new_bn.bias[:module.num_features].copy_(checkpoint_model[name + '.bias'])
                    
                    new_bn.running_mean.zero_()
                    new_bn.running_mean[:module.num_features].copy_(checkpoint_model[name + '.running_mean'])
                    
                    new_bn.running_var.fill_(1.0) # 方差默认为 1
                    new_bn.running_var[:module.num_features].copy_(checkpoint_model[name + '.running_var'])
            
            new_bn.weight.register_hook(get_mask_hook(mask_w))
            new_bn.bias.register_hook(get_mask_hook(mask_w)) # bias 同理
            
            replace_module(model, name, new_bn)
            
        # 处理全连接层 (分类头)
        elif isinstance(module, nn.Linear):
            # 判断是否是最后的分类层：它的输入应该是原来的 2048 (或对应的特征维度)
            # 我们需要把它的 in_features 也增加 n
            new_linear = nn.Linear(
                in_features=module.in_features + n, 
                out_features=module.out_features,
                bias=(module.bias is not None)
            ).to(module.weight.device)
            
            with torch.no_grad():
                new_linear.weight.zero_()
                # 填入新权重的[num_classes, :2048]位置
                new_linear.weight[:, :module.in_features].copy_(module.weight)
                if module.bias is not None:
                    new_linear.bias.copy_(module.bias)
            
            replace_module(model, name, new_linear)

    print(f"Surgery Complete: Expanded model channels by n={n} with zero-init.")
    return model