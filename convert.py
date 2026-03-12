from collections import OrderedDict
import torch
from safetensors import safe_open

def convert_videomae_pretrain(path):
    old_ckpts = torch.load(path, map_location='cpu')
    new_ckpts = OrderedDict()

    for k, v in old_ckpts['model'].items():
        if not k.startswith('encoder.'):
            continue
        if k.startswith('encoder.blocks.'):
            spk = k.split('.')
            if '.'.join(spk[3:]) == 'attn.qkv.weight':
                assert v.shape[0] % 3 == 0, v.shape
                qi, ki, vi = torch.split(v, v.shape[0] // 3, dim=0)
                new_ckpts['.'.join(spk[:3] + ['attn', 'q_proj', 'weight'])] = qi
                new_ckpts['.'.join(spk[:3] + ['attn', 'k_proj', 'weight'])] = ki
                new_ckpts['.'.join(spk[:3] + ['attn', 'v_proj', 'weight'])] = vi
            elif '.'.join(spk[3:]) == 'mlp.fc1.bias':  # 'blocks.1.norm1.weight' --> 'norm1.weight'
                new_ckpts['.'.join(spk[:3] + ['fc1', 'bias'])] = v
            elif '.'.join(spk[3:]) == 'mlp.fc1.weight':
                new_ckpts['.'.join(spk[:3] + ['fc1', 'weight'])] = v
            elif '.'.join(spk[3:]) == 'mlp.fc2.bias':
                new_ckpts['.'.join(spk[:3] + ['fc2', 'bias'])] = v
            elif '.'.join(spk[3:]) == 'mlp.fc2.weight':
                new_ckpts['.'.join(spk[:3] + ['fc2', 'weight'])] = v
            else:
                new_ckpts[k] = v
        else:
            new_ckpts[k] = v

    assert path.endswith('.pth'), path
    new_path = path[:-4] + '_new.pth'
    torch.save(OrderedDict(model=new_ckpts), new_path)
    print('Finished :', path)

def convert_timm_vit_pretrain(path, save_path=None):
    """
    专门用于将 timm 官方的 ViT 权重转换为适配本项目的格式。
    支持 .safetensors 和 .pth/.bin 格式。
    """
    # 1. 加载原始权重
    if path.endswith('.safetensors'):
        from safetensors.torch import load_file as load_safetensors
        old_ckpts = load_safetensors(path)
    else:
        old_ckpts = torch.load(path, map_location='cpu')
        # 如果是 timm 格式，通常权重在 'model' 或直接是字典
        if 'model' in old_ckpts:
            old_ckpts = old_ckpts['model']
        elif 'state_dict' in old_ckpts:
            old_ckpts = old_ckpts['state_dict']
    
    if "resnet" in path:
        torch.save({'model': old_ckpts}, save_path)
        return

    new_ckpts = OrderedDict()

    for k, v in old_ckpts.items():
        # 移除可能存在的前缀 (比如 encoder. 或 model.)
        if k.startswith('encoder.'):
            k = k.replace('encoder.', '')
        if k.startswith('model.'):
            k = k.replace('model.', '')

        # --- 核心转换逻辑 ---
        if k.startswith('blocks.'):
            # 拆分 Key：['blocks', '0', 'attn', 'qkv', 'weight']
            spk = k.split('.')
            
            # 1. 处理 QKV 拆分 (适配 custom_modules.Attention)
            if 'attn.qkv.' in k:
                # 假设 v 的形状是 [3*dim, dim] 或 [3*dim]
                suffix = spk[-1] # weight 或 bias
                qi, ki, vi = torch.split(v, v.shape[0] // 3, dim=0)
                
                # 重新拼装 Key
                # blocks.0.attn.q_proj.weight
                new_ckpts['.'.join(spk[:3] + ['q_proj', suffix])] = qi
                new_ckpts['.'.join(spk[:3] + ['k_proj', suffix])] = ki
                new_ckpts['.'.join(spk[:3] + ['v_proj', suffix])] = vi
                continue

            # 2. 处理 MLP 拍平 (从 blocks.0.mlp.fc1 变为 blocks.0.fc1)
            # 适配 vit_image.py 中对 Block 的定义
            if 'mlp.' in k:
                new_key = k.replace('mlp.', '')
                new_ckpts[new_key] = v
                continue

        # 3. 处理 Head 偏移 (适配 main_image.py 209行的 Sequential(BN, head) 逻辑)
        if k.startswith('head.'):
            # 原始 head.weight -> 变换后 head.1.weight
            new_ckpts[k.replace('head.', 'head.1.')] = v
            continue

        # 其他权重 (pos_embed, patch_embed, cls_token, norm) 保持原样
        new_ckpts[k] = v

    # 保存结果
    if save_path is None:
        save_path = path.replace('.safetensors', '.pth').replace('.bin', '.pth')
        if not save_path.endswith('.pth'):
            save_path += '_converted.pth'

    torch.save({'model': new_ckpts}, save_path)
    print(f"转换成功！\n原始文件: {path}\n保存位置: {save_path}")
    return save_path

if __name__ == '__main__':
    # path = '/path/to/videomae/pretrained/checkpoint.pth'
    # convert_videomae_pretrain(path)
    import os
    pretrained_name = os.environ['MODEL_PRETRAINED']
    if 'resnet' in pretrained_name:
        raw_path = f"raw_weights/{pretrained_name}/model.safetensors"
        save_path = f"pretrained_checkpoint/{pretrained_name}_converted.pth"
        convert_timm_vit_pretrain(raw_path, save_path)
    else:
        raw_path = f"raw_weights/{pretrained_name}/model.safetensors"
        save_path = f"pretrained_checkpoint/{pretrained_name}_converted.pth"
        convert_timm_vit_pretrain(raw_path, save_path)
