import math
import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
from matplotlib import pyplot as plt



def merge_based_on_cosine(matrix1, matrix2, threshold=0.9, bias1=None, bias2=None, is_qk=False):
    #multiply = torch.einsum("hij, hik -> hijk", matrix1, matrix2)

    #multiply = multiply.view(multiply.shape[0], multiply.shape[1], -1)
    #multiply_norm = torch.norm(multiply, dim=-1, keepdim=True)
    #multiply = multiply / multiply_norm

    #multiply_cosine = torch.einsum("hij, hkj -> hik", multiply, multiply)

    matrix1_norm = torch.norm(matrix1, dim=-1, keepdim=True)
    matrix1_normalized = matrix1 / (matrix1_norm + 1e-5)
    
    matrix1_cosine = torch.einsum("hij, hkj -> hik", matrix1_normalized, matrix1_normalized)
    
    n, h, w = matrix1_cosine.shape
    assert h == w
    for i in range(h):
        matrix1_cosine[:, i, i] *= 0

    mask1 = torch.zeros_like(matrix1)
    mask2 = torch.zeros_like(matrix2)
    if bias1 is not None:
        bias1_mask = torch.zeros_like(bias1)
    else:
        bias1_mask = None
    if bias2 is not None:
        bias2_mask = torch.zeros_like(bias2)
    else:
        bias2_mask = None

    total_num = matrix1.shape[0] * matrix1.shape[1]
    cutoff_num = 0

    for head in range(matrix1.shape[0]):
        while matrix1_cosine[head].abs().max() > threshold:
            index = torch.argmax(matrix1_cosine[head].abs(), dim=None)
            i = index // matrix1_cosine.shape[-1]
            j = index % matrix1_cosine.shape[-1]
            assert i != j
            flag = matrix1_cosine[head, i, j].sign()
            if matrix1_norm[head, i] >= matrix1_norm[head, j]:
                matrix1_cosine[head, :, j] *= 0
                matrix1_cosine[head, j, :] *= 0
            else:
                matrix1_cosine[head, i, :] *= 0
                matrix1_cosine[head, :, i] *= 0
                tmp = j
                j = i
                i = tmp
            
            # rescale i th channel
            ratio = (flag * matrix1_norm[head, j] / matrix1_norm[head, i]).item()
            matrix2[head, i, :] = matrix2[head, i, :] + matrix2[head, j, :] * ratio
            if bias2 is not None and is_qk:
                bias2[head, i] = bias2[head, i] + bias2[head, j]
            # remove j th channel
            bound = math.sqrt(6) / math.sqrt(matrix1.shape[-1])
            matrix1[head, j, :].uniform_(-bound, bound)
            matrix2[head, j, :] *= 0
            if bias1 is not None:
                bias1[head, j] = 0
                bias1_mask[head, j] = 1
            if bias2 is not None and is_qk:
                bias2[head, j] = 0
                bias2_mask[head, j] = 1
            mask1[head, j, :] += 1
            mask2[head, j, :] += 1
            
            cutoff_num += 1

    return matrix1, matrix2, bias1, bias2, mask1, mask2, bias1_mask, bias2_mask, total_num, cutoff_num