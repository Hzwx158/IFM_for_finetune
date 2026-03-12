import argparse
import datetime
import json
import numpy as np
import os
import time
from pathlib import Path
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

import timm
from timm.data import create_transform
from torchvision import datasets, transforms

import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler
from engine_finetune import train_one_epoch, evaluate
from datasets.image_datasets import build_image_dataset
from models.cnn_parallel import (
    ParallelBaseline,
    expand_resnet_with_zero_init,
)

def print_parameter_stats(model:torch.nn.Module, result_dir:Path):
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    if misc.is_main_process():
        with open(result_dir / "trainable_args.txt", "w") as f:
            f.write("-" * 30 + '\n')
            f.write(f"Total Parameters: {total_params:,}\n")
            f.write(f"Trainable Parameters: {trainable_params:,}\n")
            f.write(f"Non-trainable Parameters: {total_params - trainable_params:,}\n")
            f.write(f"Trainable Ratio: {100 * trainable_params / total_params:.4f}%\n")
            f.write("-" * 30)
        print("Write!")#, input()
    exit(0)

def get_args_parser():
    parser = argparse.ArgumentParser('CNN Fine-tuning for Image Classification', add_help=False)

    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--model', default='resnet50', type=str, help='Name of model to train')
    parser.add_argument('--nb_classes', default=100, type=int, help='number of the classification types')
    
    parser.add_argument('--dataset', default='imagenet', choices=['imagenet', 'cifar100', 'flowers102', 'svhn', 'food101'])
    parser.add_argument('--data_path', default='./data/cifar100', type=str, help='dataset path')
    parser.add_argument('--output_dir', default='./output_resnet', help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda', help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--log_dir', default=None,
                        help='path where to tensorboard log')
    # DDP
    parser.add_argument('--not_distributed', action='store_false')
    parser.add_argument('--world_size', default=1, type=int)
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_url', default='env://')
    parser.add_argument('--dist_on_itp', action='store_true')
    
    parser.add_argument('--finetune', default='', help='finetune from checkpoint')
    
    parser.add_argument('--parallel_baseline', action="store_true", help="Open ParallelBaseline")
    parser.add_argument('--parallel_only_new', action="store_true")
    parser.add_argument('--expand', type=int, default=None)
    
    parser.add_argument('--merge_before_finetune', action='store_true', help='Apply gradient masks')
    parser.add_argument('--mask_dict', default='', type=str, help='path to mask_dict .pth file')
    
    parser.add_argument('--weight_decay', type=float, default=0.,
                        help='weight decay (default: 0 for linear probe following MoCo v1)')
    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=0.1, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')
    parser.add_argument('--warmup_epochs', type=int, default=20, metavar='N',
                        help='epochs to warmup LR')
    parser.add_argument('--accum_iter', default=1, type=int,
                    help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    return parser

def get_mask_hook(mask):
    def hook(grad):
        return grad * mask.to(grad.device)
    return hook

def main(args):
    args.distributed = not args.not_distributed
    misc.init_distributed_mode(args)
    device = torch.device(args.device)
    cudnn.benchmark = True

    # 1. 构建数据集
    dataset_train, dataset_val, args.nb_classes = build_image_dataset(args)

    if args.distributed:
        sampler_train = DistributedSampler(dataset_train, num_replicas=misc.get_world_size(), rank=misc.get_rank(), shuffle=True)
        sampler_val = DistributedSampler(dataset_val, num_replicas=misc.get_world_size(), rank=misc.get_rank(), shuffle=True)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    data_loader_train = DataLoader(dataset_train, sampler=sampler_train, batch_size=args.batch_size, num_workers=4, pin_memory=True, drop_last=True)
    data_loader_val = DataLoader(dataset_val, sampler=sampler_val, batch_size=args.batch_size, num_workers=4, pin_memory=True, drop_last=False)
    
    log_writer = None
    if misc.get_rank() == 0 and args.log_dir:
        log_writer = SummaryWriter(log_dir=args.log_dir)
    
    # 2. 创建模型
    print(f"Creating model: {args.model}")
    model = timm.create_model(args.model, pretrained=False, num_classes=args.nb_classes)
    model = model.to(device)

    # 3. 加载合并后的权重
    if args.finetune:
        checkpoint = torch.load(args.finetune, map_location='cpu')
        checkpoint_model = checkpoint['model'] if 'model' in checkpoint else checkpoint
        state_dict = model.state_dict()
        for k in ['fc.weight', 'fc.bias', 'head.weight', 'head.bias']:
            if k in checkpoint_model and k in state_dict:
                if checkpoint_model[k].shape != state_dict[k].shape:
                    print(f"Removing key {k} from checkpoint due to shape mismatch")
                    del checkpoint_model[k]
        
        if args.expand is not None:
            model = expand_resnet_with_zero_init(model, checkpoint_model, n=args.expand)
        else:
            model.load_state_dict(checkpoint_model, strict=False)
    
    if args.parallel_baseline:
        model = ParallelBaseline(model)
        model = model.to(device)
        
    if args.parallel_only_new:
        for name, param in model.named_parameters():
            if "adapters" in name or "head" in name or "fc" in name: 
                param.requires_grad = True
            else: # 冻结 base_model
                param.requires_grad = False
    
    
                
                
    _trainable = [0, 0]
    for _, param in model.named_parameters():
        _trainable[0] += int(param.requires_grad)
        _trainable[1] += 1
    print(f"{_trainable[0]} / {_trainable[1]}") #,input()
    # 多卡
    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    # 4. 注册梯度屏蔽 Hook
    if args.merge_before_finetune and args.mask_dict:
        print(f"Loading mask_dict from {args.mask_dict}")
        mask_dict = torch.load(args.mask_dict, map_location='cpu')
        count = 0
        for name, p in model.named_parameters():
            # 兼容处理前缀
            clean_name = name.replace('model.', '').replace('module.', '')
            if clean_name in mask_dict:
                p.register_hook(get_mask_hook(mask_dict[clean_name]))
                count += 1
        print(f"Successfully registered {count} gradient hooks.")
        # input()

    # 5. 优化器与调度器
    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()
    if args.lr is None:
        args.lr = args.blr * eff_batch_size / 256
        
        if misc.is_main_process():
            print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
            
    if misc.is_main_process():
        print("actual lr: %.2e" % args.lr)
        print("accumulate grad iterations: %d" % args.accum_iter)
        print("effective batch size: %d" % eff_batch_size)

    param_groups = [
        {
            'params': [p for name, p in model_without_ddp.named_parameters() 
                       if (p.requires_grad and "scale" in name)], 
            'lr_scale': 0.005
        },
        {
            'params': [p for name, p in model_without_ddp.named_parameters() 
                       if (p.requires_grad and "scale" not in name)], 
            'lr_scale': 1.0
        }
    ]
    
    if misc.is_main_process():
        num_scale_params = len(param_groups[0]['params'])
        print(f"Number of scale parameters: {num_scale_params}")

    optimizer = torch.optim.SGD(
        param_groups, 
        lr=args.lr, 
        weight_decay=args.weight_decay, 
        momentum=0.9
    )
    
    print_parameter_stats(
        model=model_without_ddp,
        result_dir=Path(args.output_dir),
    )
    
    print(optimizer)
    loss_scaler = NativeScaler()
    criterion = torch.nn.CrossEntropyLoss()

    # 6. 训练循环
    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    max_accuracy = 0.0

    for epoch in range(args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        train_stats = train_one_epoch(
            model, criterion, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            max_norm=None, log_writer=log_writer, 
            args=args
        )
        
        test_stats = evaluate(data_loader_val, model, device)
        print(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
        
        max_accuracy = max(max_accuracy, test_stats['acc1'])
        print(f'Max accuracy: {max_accuracy:.2f}%')

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     **{f'test_{k}': v for k, v in test_stats.items()},
                     'epoch': epoch}

        print(args.output_dir)
        
        if args.output_dir and misc.is_main_process():
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

        if args.output_dir and max_accuracy == test_stats["acc1"]:
            if torch.distributed.get_rank()==0:
                for p in Path(args.output_dir).glob("checkpoint-*.pth"):
                    p.unlink()
            misc.save_model(
                args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                loss_scaler=loss_scaler, epoch=epoch)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)