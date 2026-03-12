import argparse
import datetime
import json
import numpy as np
import os
import math
import time
from pathlib import Path
from easydict import EasyDict
import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
from matplotlib import pyplot as plt

import timm
# assert timm.__version__ == "0.3.2" # version check
from timm.models.layers import trunc_normal_

import util.misc as misc
from util.pos_embed import interpolate_pos_embed_ori as interpolate_pos_embed
from util.misc import NativeScalerWithGradNormCount as NativeScaler

from datasets.image_datasets import build_image_dataset
from engine_finetune import train_one_epoch, evaluate
import models.vit_image as vit_image
# from warnings import deprecated


def merge_based_on_cosine(matrix1, matrix2, threshold=0.9, bias1=None, bias2=None):
    multiply = torch.einsum("hij, hik -> hijk", matrix1, matrix2)

    multiply = multiply.view(multiply.shape[0], multiply.shape[1], -1)
    multiply_norm = torch.norm(multiply, dim=-1, keepdim=True)
    multiply = multiply / multiply_norm

    multiply_cosine = torch.einsum("hij, hkj -> hik", multiply, multiply)

    n, h, w = multiply_cosine.shape
    assert h == w
    for i in range(h):
        multiply_cosine[:, i, i] *= 0

    mask1 = torch.ones_like(matrix1)
    mask2 = torch.ones_like(matrix2)

    total_num = matrix1.shape[0] * matrix1.shape[1]
    cutoff_num = 0

    for head in range(multiply.shape[0]):
        while multiply_cosine[head].abs().max() > threshold:
            index = torch.argmax(multiply_cosine[head].abs(), dim=None)
            i = index // multiply_cosine.shape[-1]
            j = index % multiply_cosine.shape[-1]
            assert i != j
            if multiply_norm[head, i] >= multiply_norm[head, j]:
                multiply_cosine[head, :, j] *= 0
            else:
                multiply_cosine[head, i, :] *= 0
                tmp = j
                j = i
                i = tmp
            # remove j th channel
            matrix1[head, j, :] *= 0
            matrix2[head, j, :] *= 0
            if bias1 is not None:
                bias1[head, j] = 0
            if bias2 is not None:
                bias2[head, j] = 0
            mask1[head, j, :] *= 0
            mask2[head, j, :] *= 0
            # rescale i th channel
            ratio = torch.sqrt((multiply_norm[head, i] + multiply_norm[head, j]) / multiply_norm[head, i]).item()
            matrix1[head, i, :] *= ratio
            matrix2[head, i, :] *= ratio
            cutoff_num += 1

    return matrix1, matrix2, bias1, bias2, mask1, mask2, total_num, cutoff_num

def load_and_merge(
    ckpt_path, threshold=0.9, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, 
    merge_qk=True, merge_vo=True, merge_ffn=True,
    merge_cnn=True,
):
    checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    #print(checkpoint.keys())
    #print(checkpoint["args"])
    checkpoint_model = checkpoint["model"] if 'model' in checkpoint else checkpoint

    total_channel_num = 0
    cutoff_channel_num = 0
    for key in checkpoint_model.keys():
        if "attn.q_proj.weight" in key and merge_qk:
            q_name = key
            k_name = key.replace("attn.q_proj.weight", "attn.k_proj.weight")
            q_weight = checkpoint_model[q_name]
            k_weight = checkpoint_model[k_name]
            q_bias = checkpoint_model[q_name.replace("weight", "bias")]
            k_bias = checkpoint_model[k_name.replace("weight", "bias")]

            q_h, q_w = q_weight.shape
            k_h, k_w = k_weight.shape

            q_weight = q_weight.view(num_heads, q_h//num_heads, q_w)
            k_weight = k_weight.view(num_heads, k_h//num_heads, k_w)
            q_bias = q_bias.view(num_heads, q_h//num_heads)
            k_bias = k_bias.view(num_heads, k_h//num_heads)

            q_weight, k_weight, q_bias, k_bias, _, _, total_num, cutoff_num = merge_based_on_cosine(q_weight, k_weight, threshold, q_bias, k_bias)
            q_weight = q_weight.view(q_h, q_w)
            k_weight = k_weight.view(k_h, k_w)
            q_bias = q_bias.view(q_h)
            k_bias = k_bias.view(k_h)
            total_channel_num += total_num
            cutoff_channel_num += cutoff_num

            checkpoint_model[q_name] = q_weight
            checkpoint_model[k_name] = k_weight
            checkpoint_model[q_name.replace("weight", "bias")] = q_bias
            checkpoint_model[k_name.replace("weight", "bias")] = k_bias

        if "attn.v_proj.weight" in key and merge_vo:
            v_name = key
            o_name = key.replace("attn.v_proj.weight", "attn.proj.weight")
            v_weight = checkpoint_model[v_name]
            o_weight = checkpoint_model[o_name]
            v_bias = checkpoint_model[v_name.replace("weight", "bias")]
            # transpose O weight
            o_weight = o_weight.T

            v_h, v_w = v_weight.shape
            o_h, o_w = o_weight.shape

            v_weight = v_weight.view(num_heads, v_h//num_heads, v_w)
            o_weight = o_weight.view(num_heads, o_h//num_heads, o_w)
            v_bias = v_bias.view(num_heads, v_h//num_heads)

            v_weight, o_weight, v_bias, _, _, _, total_num, cutoff_num = merge_based_on_cosine(v_weight, o_weight, threshold, v_bias, None)
            v_weight = v_weight.view(v_h, v_w)
            o_weight = o_weight.view(o_h, o_w)
            v_bias = v_bias.view(v_h)
            total_channel_num += total_num
            cutoff_channel_num += cutoff_num

            o_weight = o_weight.T

            checkpoint_model[v_name] = v_weight
            checkpoint_model[o_name] = o_weight
            checkpoint_model[v_name.replace("weight", "bias")] = v_bias

        if "fc1.weight" in key and merge_ffn:
            fc1_name = key
            fc2_name = key.replace("fc1.weight", "fc2.weight")
            fc1_weight = checkpoint_model[fc1_name]
            fc2_weight = checkpoint_model[fc2_name]
            fc1_bias = checkpoint_model[fc1_name.replace("weight", "bias")]

            fc2_weight = fc2_weight.T

            fc1_h, fc1_w = fc1_weight.shape
            fc2_h, fc2_w = fc2_weight.shape

            fc1_weight = fc1_weight.view(1, fc1_h, fc1_w)
            fc2_weight = fc2_weight.view(1, fc2_h, fc2_w)
            fc1_bias = fc1_bias.view(1, fc1_h)

            fc1_weight, fc2_weight, fc1_bias, _, _, _, total_num, cutoff_num = merge_based_on_cosine(fc1_weight, fc2_weight, threshold, fc1_bias, None)
            fc1_weight = fc1_weight.squeeze()
            fc2_weight = fc2_weight.squeeze()
            fc1_bias = fc1_bias.squeeze()
            total_channel_num += total_num
            cutoff_channel_num += cutoff_num

            fc2_weight = fc2_weight.T

            checkpoint_model[fc1_name] = fc1_weight
            checkpoint_model[fc2_name] = fc2_weight
            checkpoint_model[fc1_name.replace("weight", "bias")] = fc1_bias
    return checkpoint_model, cutoff_channel_num, total_channel_num

def get_args_parser():
    parser = argparse.ArgumentParser('AdaptFormer fine-tuning for action recognition for image classification', add_help=False)
    parser.add_argument('--batch_size', default=512, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    # Model parameters
    parser.add_argument('--model', default='vit_base_patch16', type=str, metavar='MODEL',
                        help='Name of model to train')

    # Optimizer parameters
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

    # * Finetuning params
    parser.add_argument('--finetune', default='/home/ma-user/modelarts/work/ytchen/LowRankOptim/AdaptFormer-main/pretrained_checkpoint/mae_pretrain_vit_b.pth',
                        help='finetune from checkpoint')
    parser.add_argument('--global_pool', action='store_true')
    parser.set_defaults(global_pool=False)
    parser.add_argument('--cls_token', action='store_false', dest='global_pool',
                        help='Use class token instead of global pool for classification')

    # Dataset parameters
    parser.add_argument('--data_path', default='/home/ma-user/modelarts/work/ytchen/LowRankOptim/AdaptFormer-main/data', type=str,
                        help='dataset path')
    parser.add_argument('--nb_classes', default=1000, type=int,
                        help='number of the classification types')

    parser.add_argument('--output_dir', default='./output_dir',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default=None,
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true',
                        help='Perform evaluation only')
    parser.add_argument('--dist_eval', action='store_true', default=False,
                        help='Enabling distributed evaluation (recommended during training for faster monitor')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')

    # custom configs
    parser.add_argument('--dataset', default='imagenet', choices=['imagenet', 'cifar100', 'flowers102', 'svhn', 'food101'])
    parser.add_argument('--drop_path', type=float, default=0.0, metavar='PCT',
                        help='Drop path rate (default: 0.0)')

    parser.add_argument('--inception', default=False, action='store_true', help='whether use INCPETION mean and std'
                                                                                '(for Jx provided IN-21K pretrain')
    # AdaptFormer related parameters
    parser.add_argument('--ffn_adapt', default=False, action='store_true', help='whether activate AdaptFormer')
    parser.add_argument('--ffn_num', default=64, type=int, help='bottleneck middle dimension')
    parser.add_argument('--vpt', default=False, action='store_true', help='whether activate VPT')
    parser.add_argument('--vpt_num', default=1, type=int, help='number of VPT prompts')
    parser.add_argument('--fulltune', default=False, action='store_true', help='full finetune model')

    return parser

# @deprecated
def test_merge_model(ckpt_path, data_name, threshold=0.9, merge_qk=True, merge_vo=True, merge_ffn=True, eval_origin=True, merge_cnn=True):
    checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    # print(checkpoint.keys())
    # print(checkpoint["args"])
    checkpoint_model = checkpoint["model"] if 'model' in checkpoint else checkpoint

    tick = time.time()
    merged_checkpoint_model, cutoff_channel_num, total_channel_num = load_and_merge(
        ckpt_path, threshold, 
        merge_qk=merge_qk, merge_vo=merge_vo, merge_ffn=merge_ffn, 
        merge_cnn=merge_cnn,
    )
    tock = time.time()
    print("Merge model took {:.3f}s".format(tock - tick))
    print("Cutoff channel num: {:d}; {:.2f}%".format(cutoff_channel_num, 100 * cutoff_channel_num / total_channel_num))

    parser = get_args_parser()
    args = parser.parse_args()
    args.dataset = data_name
    args.distributed = args.world_size > 1
    args.cls_token = False
    dataset_train, dataset_val, args.nb_classes = build_image_dataset(args)

    print(dataset_train)
    print(dataset_val)

    if True:  # args.distributed:
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        print("Sampler_train = %s" % str(sampler_train))
        if args.dist_eval:
            if len(dataset_val) % num_tasks != 0:
                print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                      'This will slightly alter validation results as extra duplicate entries are added to achieve '
                      'equal num of samples per-process.')
            sampler_val = torch.utils.data.DistributedSampler(
                dataset_val, num_replicas=num_tasks, rank=global_rank,
                shuffle=True)  # shuffle=True to reduce monitor bias
        else:
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False
    )

    tuning_config = EasyDict(
        # AdaptFormer
        ffn_adapt=args.ffn_adapt,
        ffn_option="parallel",
        ffn_adapter_layernorm_option="none",
        ffn_adapter_init_option="lora",
        ffn_adapter_scalar="0.1",
        ffn_num=args.ffn_num,
        d_model=768,
        # VPT related
        vpt_on=args.vpt,
        vpt_num=args.vpt_num,
    )

    device = torch.device(args.device)
    if eval_origin:
        if args.model.startswith('vit'):
            model = vit_image.__dict__[args.model](
                num_classes=args.nb_classes,
                global_pool=args.global_pool,
                drop_path_rate=args.drop_path,
                tuning_config=tuning_config,
            )
        else:
            raise NotImplementedError(args.model)
        model.head = torch.nn.Sequential(torch.nn.BatchNorm1d(model.head.in_features, affine=False, eps=1e-6),
                                         model.head)
        model.load_state_dict(checkpoint_model)
        model.to(device)
        if args.distributed:
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
            model_without_ddp = model.module
        test_stats = evaluate(data_loader_val, model, device)
        print(f"Accuracy of the original network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")

    if args.model.startswith('vit'):
        model = vit_image.__dict__[args.model](
        num_classes=args.nb_classes,
        global_pool=args.global_pool,
        drop_path_rate=args.drop_path,
        tuning_config=tuning_config,
        )
    else:
        raise NotImplementedError(args.model)
    model.head = torch.nn.Sequential(torch.nn.BatchNorm1d(model.head.in_features, affine=False, eps=1e-6), model.head)
    model.load_state_dict(merged_checkpoint_model)
    model.to(device)
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module
    test_stats = evaluate(data_loader_val, model, device)
    print(f"Accuracy of the merged model on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")

def merge_save(
    ckpt_path, save_path, threshold=0.9, 
    embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, 
    merge_qk=True, merge_vo=True, merge_ffn=True, merge_cnn=True,
):
    checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    # print(checkpoint.keys())
    # print(checkpoint["args"])
    checkpoint_model = checkpoint["model"] if 'model' in checkpoint else checkpoint

    mask_dict = {}
    from models.merge_weight import merge_based_on_cosine
    total_cutoff_num = 0
    tick = time.time()
    for key in checkpoint_model.keys():
        print(key)
        if "attn.q_proj.weight" in key and merge_qk:
            q_name = key
            k_name = key.replace("attn.q_proj.weight", "attn.k_proj.weight")
            q_weight = checkpoint_model[q_name]
            k_weight = checkpoint_model[k_name]
            q_bias = checkpoint_model[q_name.replace("weight", "bias")]
            k_bias = checkpoint_model[k_name.replace("weight", "bias")]

            q_h, q_w = q_weight.shape
            k_h, k_w = k_weight.shape

            q_weight = q_weight.view(num_heads, q_h // num_heads, q_w)
            k_weight = k_weight.view(num_heads, k_h // num_heads, k_w)
            q_bias = q_bias.view(num_heads, q_h // num_heads)
            k_bias = k_bias.view(num_heads, k_h // num_heads)

            q_weight, k_weight, q_bias, k_bias, q_mask, k_mask, q_bias_mask, k_bias_mask, total_num, cutoff_num = merge_based_on_cosine(q_weight, k_weight,
                                                                                             threshold, q_bias, k_bias, is_qk=True)

            q_weight = q_weight.view(q_h, q_w)
            k_weight = k_weight.view(k_h, k_w)
            q_bias = q_bias.view(q_h)
            k_bias = k_bias.view(k_h)
            q_mask = q_mask.view(q_h, q_w)
            k_mask = k_mask.view(k_h, k_w)
            q_bias_mask = q_bias_mask.view(q_h)
            k_bias_mask = k_bias_mask.view(k_h)

            checkpoint_model[q_name] = q_weight
            checkpoint_model[k_name] = k_weight
            checkpoint_model[q_name.replace("weight", "bias")] = q_bias
            checkpoint_model[k_name.replace("weight", "bias")] = k_bias
            mask_dict[q_name] = q_mask
            mask_dict[k_name] = k_mask
            mask_dict[q_name.replace("weight", "bias")] = q_bias_mask
            mask_dict[k_name.replace("weight", "bias")] = k_bias_mask
            total_cutoff_num += cutoff_num
        if "attn.v_proj.weight" in key and merge_vo:
            v_name = key
            o_name = key.replace("attn.v_proj.weight", "attn.proj.weight")
            v_weight = checkpoint_model[v_name]
            o_weight = checkpoint_model[o_name]
            v_bias = checkpoint_model[v_name.replace("weight", "bias")]
            o_bias = checkpoint_model[o_name.replace("weight", "bias")]
            # transpose O weight
            o_weight = o_weight.T

            v_h, v_w = v_weight.shape
            o_h, o_w = o_weight.shape

            v_weight = v_weight.view(num_heads, v_h // num_heads, v_w)
            o_weight = o_weight.view(num_heads, o_h // num_heads, o_w)
            v_bias = v_bias.view(num_heads, v_h // num_heads)
            o_bias = o_bias.view(num_heads, o_w // num_heads)

            v_weight, o_weight, v_bias, o_bias, v_mask, o_mask, v_bias_mask, o_bias_mask, total_num, cutoff_num = merge_based_on_cosine(v_weight, o_weight,
                                                                                        threshold, v_bias, o_bias, is_qk=False)
            v_weight = v_weight.view(v_h, v_w)
            o_weight = o_weight.view(o_h, o_w)
            v_bias = v_bias.view(v_h)
            o_bias = o_bias.view(o_w)
            v_mask = v_mask.view(v_h, v_w)
            o_mask = o_mask.view(o_h, o_w)
            v_bias_mask = v_bias_mask.view(v_h)
            o_bias_mask = o_bias_mask.view(o_w)

            o_weight = o_weight.T
            o_mask = o_mask.T

            checkpoint_model[v_name] = v_weight
            checkpoint_model[o_name] = o_weight
            checkpoint_model[v_name.replace("weight", "bias")] = v_bias
            checkpoint_model[o_name.replace("weight", "bias")] = o_bias
            mask_dict[v_name] = v_mask
            mask_dict[o_name] = o_mask
            mask_dict[v_name.replace("weight", "bias")] = v_bias_mask
            mask_dict[o_name.replace("weight", "bias")] = o_bias_mask
            total_cutoff_num += cutoff_num
        if "fc1.weight" in key and merge_ffn:
            fc1_name = key
            fc2_name = key.replace("fc1.weight", "fc2.weight")
            fc1_weight = checkpoint_model[fc1_name]
            fc2_weight = checkpoint_model[fc2_name]
            fc1_bias = checkpoint_model[fc1_name.replace("weight", "bias")]
            fc2_bias = checkpoint_model[fc2_name.replace("weight", "bias")]

            fc2_weight = fc2_weight.T

            fc1_h, fc1_w = fc1_weight.shape
            fc2_h, fc2_w = fc2_weight.shape

            fc1_weight = fc1_weight.view(1, fc1_h, fc1_w)
            fc2_weight = fc2_weight.view(1, fc2_h, fc2_w)
            fc1_bias = fc1_bias.view(1, fc1_h)
            fc2_bias = fc2_bias.view(1, fc2_w)

            fc1_weight, fc2_weight, fc1_bias, _, fc1_mask, fc2_mask, fc1_bias_mask, fc2_bias_mask, total_num, cutoff_num = merge_based_on_cosine(fc1_weight, fc2_weight,
                                                                                                  threshold, fc1_bias, fc2_bias, is_qk=False)
            fc1_weight = fc1_weight.squeeze()
            fc2_weight = fc2_weight.squeeze()
            fc1_bias = fc1_bias.squeeze()
            fc2_bias = fc2_bias.squeeze()

            fc1_mask = fc1_mask.squeeze()
            fc2_mask = fc2_mask.squeeze()
            fc1_bias_mask = fc1_bias_mask.squeeze()
            fc2_bias_mask = fc2_bias_mask.squeeze()

            fc2_weight = fc2_weight.T
            fc2_mask = fc2_mask.T

            checkpoint_model[fc1_name] = fc1_weight
            checkpoint_model[fc2_name] = fc2_weight
            checkpoint_model[fc1_name.replace("weight", "bias")] = fc1_bias
            checkpoint_model[fc2_name.replace("weight", "bias")] = fc2_bias
            mask_dict[fc1_name] = fc1_mask
            mask_dict[fc2_name] = fc2_mask
            mask_dict[fc1_name.replace("weight", "bias")] = fc1_bias_mask
            mask_dict[fc2_name.replace("weight", "bias")] = fc2_bias_mask
            total_cutoff_num += cutoff_num
        
        if "conv" in key and ".weight" in key and merge_cnn:
            conv_w_name = key
            conv_b_name = key.replace("weight", "bias")
            conv_weight = checkpoint_model[conv_w_name]
            conv_bias = checkpoint_model.get(conv_b_name, None)

            # 记录原始形状 [OC, IC, KH, KW]
            oc, ic, kh, kw = conv_weight.shape
            
            # 风格对齐：按照作者处理 FFN 的方式，view 成 (1, OC, -1)
            # 这里的 -1 相当于 IC * KH * KW
            conv_weight_v = conv_weight.view(1, oc, -1)
            if conv_bias is not None:
                conv_bias_v = conv_bias.view(1, oc)
            else:
                conv_bias_v = None

            # 调用函数：CNN 是自合并，所以 matrix1 和 matrix2 都传权重本身
            # is_qk 设为 False，避免处理不存在的 bias2
            (
                conv_w1, conv_w2, conv_b1, _, 
                c_mask1, c_mask2, cb_mask1, _, 
                total_num, cutoff_num
            ) = merge_based_on_cosine(
                conv_weight_v, conv_weight_v, 
                threshold, conv_bias_v, None, is_qk=False
            )

            # 作者在 FFN 里取的是 matrix2 (合并后的权重) 和 mask2
            conv_weight = conv_w2.view(oc, ic, kh, kw)
            conv_mask = c_mask2.view(oc, ic, kh, kw)
            
            checkpoint_model[conv_w_name] = conv_weight
            mask_dict[conv_w_name] = conv_mask
            
            if conv_bias is not None:
                conv_bias = conv_b1.view(oc)
                conv_bias_mask = cb_mask1.view(oc)
                checkpoint_model[conv_b_name] = conv_bias
                mask_dict[conv_b_name] = conv_bias_mask
            
            total_cutoff_num += cutoff_num
            # print(f"CNN Layer {key}: Merged {cutoff_num}/{oc} channels")
        
        
    tock = time.time()
    print(f"Merge finished. Take {tock - tick}s. Merged {total_cutoff_num} channels")
    if merge_qk and merge_vo and merge_ffn:
        suffix = ""
    elif not merge_qk and merge_vo and merge_ffn:
        suffix = "_no_qk"
    elif not merge_qk and not merge_vo and merge_ffn:
        suffix = "_only_ffn"
    else:
        raise ValueError("Unknown setting")
    n = Path(ckpt_path).name
    n = n[:n.rfind(".")]
    torch.save(mask_dict, os.path.join(save_path, f"{n}_mask_dict_{threshold}{suffix}.pth"))
    # saved_file = os.path.join(save_path, f"mae_pretrain_vit_b_merge_{threshold}{suffix}.pth")
    saved_file = str(Path(save_path) / f"{n}_merge_{threshold}{suffix}.pth")
    torch.save(checkpoint_model, saved_file)
    print(f"Saved to {saved_file}")
    

if __name__ == '__main__':
    #load_and_merge("/home/ma-user/modelarts/work/ytchen/LowRankOptim/AdaptFormer-main/result/cifar100_fulltune/checkpoint-99.pth")
    #test_merge_model("/home/ma-user/modelarts/work/ytchen/LowRankOptim/AdaptFormer-main/result/result_food101_fulltune/checkpoint-99.pth", "food101", threshold=0.95, merge_qk=True, merge_vo=True, eval_origin=True)
    
    PRETRAINED_DIR = Path(__file__).parent/"pretrained_checkpoint/"
    pretrained_name = os.environ["MODEL_PRETRAINED"]
    if('resnet' in pretrained_name):
        merge_save(
            str(PRETRAINED_DIR/f"{pretrained_name}_converted.pth"),
            str(PRETRAINED_DIR),
            threshold=0.99, merge_qk=False,
            merge_cnn=True,
        )
    else:
        model_args = dict(
        #     embed_dim=1024,
        #     depth=24, 
        #     num_heads=16, 
        #     mlp_ratio=4, 
        ) #从vit_image里抄一下
        merge_save(
            # str(PRETRAINED_DIR/"mae_pretrain_vit_b.pth"), 
            str(PRETRAINED_DIR/f"{pretrained_name}_converted.pth"),
            str(PRETRAINED_DIR), 
            threshold=0.99, merge_qk=False, 
            **model_args
        )