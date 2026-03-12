import argparse
import datetime
import json
import numpy as np
import os
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


def plot_cosine_similarity_histogram(matrix_tensor, dim=1, bins=30, density=False,
                                     title="Cosine Similarity Distribution",
                                     figsize=(12, 8), color='steelblue',
                                     show_stats=True, show_kde=True,
                                     include_diagonal=False, save_path=None):
    """
    计算矩阵行向量间的余弦相似度并绘制直方图

    参数:
    ----------
    matrix_tensor : torch.Tensor
        输入的PyTorch张量，形状为 (n_samples, n_features)
    bins : int 或 sequence, 可选
        直方图的箱子数量或边界
    density : bool, 可选
        是否将直方图归一化为概率密度
    title : str, 可选
        图表标题
    figsize : tuple, 可选
        图形大小 (宽度, 高度)
    color : str, 可选
        直方图颜色
    show_stats : bool, 可选
        是否在图中显示统计信息
    show_kde : bool, 可选
        是否显示核密度估计(KDE)曲线
    include_diagonal : bool, 可选
        是否包含对角线元素（自相似度，总是为1）
    save_path : str, 可选
        保存图像的路径，如果为None则不保存

    返回:
    -------
    similarity_values : numpy.ndarray
        用于绘制直方图的相似度值
    fig : matplotlib.figure.Figure
        图形对象
    ax : matplotlib.axes.Axes
        坐标轴对象
    stats_dict : dict
        统计信息字典
    """

    # 确保输入是PyTorch张量
    if not isinstance(matrix_tensor, torch.Tensor):
        matrix_tensor = torch.tensor(matrix_tensor, dtype=torch.float32)

    # 确保是2D张量
    if matrix_tensor.dim() != 2:
        raise ValueError(f"输入张量必须是2D矩阵，但得到的是{matrix_tensor.dim()}D")

    # 计算余弦相似度矩阵
    normalized = matrix_tensor / torch.norm(matrix_tensor, dim=dim, keepdim=True)
    if dim == 1:
        similarity_matrix = torch.mm(normalized, normalized.T)
    elif dim == 0:
        similarity_matrix = torch.mm(normalized.T, normalized)

    # 转换为numpy数组
    similarity_np = similarity_matrix.detach().cpu().numpy()

    # 提取用于直方图的值
    n = similarity_np.shape[1-dim]

    if include_diagonal:
        # 包含对角线元素
        similarity_values = similarity_np.flatten()
    else:
        # 排除对角线元素（自相似度总是1）
        mask = ~np.eye(n, dtype=bool)
        similarity_values = similarity_np[mask]

    # 计算统计信息
    stats_dict = {
        'mean': np.mean(similarity_values),
        'median': np.median(similarity_values),
        'std': np.std(similarity_values),
        'min': np.min(similarity_values),
        'max': np.max(similarity_values),
        'num_pairs': len(similarity_values),
        'num_samples': n
    }

    # 创建图形
    fig, ax = plt.subplots(figsize=figsize)

    # 绘制直方图
    n, bins, patches = ax.hist(similarity_values, bins=bins, density=density,
                               alpha=0.7, color=color, edgecolor='black', linewidth=0.5)

    '''# 添加KDE曲线
    if show_kde:
        from scipy.stats import gaussian_kde
        kde = gaussian_kde(similarity_values)
        x_range = np.linspace(similarity_values.min(), similarity_values.max(), 1000)
        ax.plot(x_range, kde(x_range), color='darkred', linewidth=2, label='KDE')'''

    # 设置图表属性
    ax.set_xlabel('Cosine Similarity', fontsize=14)
    ylabel = 'Probability Density' if density else 'Frequency'
    ax.set_ylabel(ylabel, fontsize=14)

    # 设置标题
    ax.set_title(title, fontsize=16, fontweight='bold')

    # 设置x轴范围
    ax.set_xlim(-1.05, 1.05)

    # 添加网格
    ax.grid(True, alpha=0.3, linestyle='--')

    # 添加垂直线表示均值和零值
    ax.axvline(x=stats_dict['mean'], color='red', linestyle='--', linewidth=2, label=f"Mean: {stats_dict['mean']:.3f}")
    ax.axvline(x=0, color='black', linestyle='-', linewidth=1, alpha=0.5, label='Zero')

    # 添加图例
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, fontsize=12, loc='best')

    # 显示统计信息
    if show_stats:
        stats_text = f"""
        Statistics:
        - Mean: {stats_dict['mean']:.3f}
        - Median: {stats_dict['median']:.3f}
        - Std: {stats_dict['std']:.3f}
        - Min: {stats_dict['min']:.3f}
        - Max: {stats_dict['max']:.3f}
        - Samples: {stats_dict['num_samples']}
        - Vector Pairs: {stats_dict['num_pairs']}
        """

        # 添加文本框
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=props, fontfamily='monospace')

    # 调整布局
    plt.tight_layout()

    # 保存图像
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"直方图已保存至: {save_path}")

    plt.show()

    return similarity_values, fig, ax, stats_dict

def plot_cosine_similarity_heatmap(matrix_tensor, dim=1, title="Cosine Similarity Heatmap",
                                   figsize=(10, 8), cmap='RdYlBu', vmin=-1, vmax=1,
                                   annotate=False, show_colorbar=True, save_path=None):
    """
    计算矩阵行向量间的余弦相似度并绘制热图

    参数:
    ----------
    matrix_tensor : torch.Tensor
        输入的PyTorch张量，形状为 (n_samples, n_features)
    title : str, 可选
        热图标题
    figsize : tuple, 可选
        图形大小 (宽度, 高度)
    cmap : str, 可选
        颜色映射 (参考matplotlib colormaps)
    vmin, vmax : float, 可选
        颜色映射范围
    annotate : bool, 可选
        是否在热图上显示数值
    show_colorbar : bool, 可选
        是否显示颜色条
    save_path : str, 可选
        保存图像的路径，如果为None则不保存

    返回:
    -------
    similarity_matrix : numpy.ndarray
        余弦相似度矩阵
    fig : matplotlib.figure.Figure
        图形对象
    ax : matplotlib.axes.Axes
        坐标轴对象
    """

    # 确保输入是PyTorch张量
    if not isinstance(matrix_tensor, torch.Tensor):
        matrix_tensor = torch.tensor(matrix_tensor, dtype=torch.float32)

    # 确保是2D张量
    if matrix_tensor.dim() != 2:
        raise ValueError(f"输入张量必须是2D矩阵，但得到的是{matrix_tensor.dim()}D")

    # 计算余弦相似度
    # 方法1: 使用归一化点积
    normalized = matrix_tensor / torch.norm(matrix_tensor, dim=dim, keepdim=True)
    if dim == 1:
        similarity_matrix = torch.mm(normalized, normalized.T)
    elif dim == 0:
        similarity_matrix = torch.mm(normalized.T, normalized)
    # 方法2: 使用torch.nn.functional.cosine_similarity (逐对计算)
    # from torch.nn.functional import cosine_similarity
    # n = matrix_tensor.size(0)
    # similarity_matrix = torch.zeros((n, n))
    # for i in range(n):
    #     for j in range(n):
    #         similarity_matrix[i, j] = cosine_similarity(
    #             matrix_tensor[i].unsqueeze(0),
    #             matrix_tensor[j].unsqueeze(0)
    #         )

    # 转换为numpy数组用于绘图
    similarity_np = similarity_matrix.detach().cpu().numpy()

    # 创建热图
    fig, ax = plt.subplots(figsize=figsize)

    # 绘制热图
    im = ax.imshow(similarity_np, cmap=cmap, vmin=vmin, vmax=vmax)

    # 添加标题
    ax.set_title(title, fontsize=16, fontweight='bold')

    # 设置坐标轴标签
    ax.set_xlabel('Row Index', fontsize=12)
    ax.set_ylabel('Row Index', fontsize=12)

    '''# 添加刻度
    #n_rows = similarity_np.shape[1-dim]
    #ax.set_xticks(np.arange(n_rows, step=50))
    #ax.set_yticks(np.arange(n_rows, step=50))
    #ax.set_xticklabels(np.arange(n_rows))
    #ax.set_yticklabels(np.arange(n_rows))

    # 可选：添加数值标注
    if annotate:
        for i in range(n_rows):
            for j in range(n_rows):
                # 根据背景颜色调整文本颜色
                val = similarity_np[i, j]
                text_color = 'white' if abs(val) > 0.5 else 'black'
                ax.text(j, i, f'{val:.2f}',
                        ha='center', va='center',
                        color=text_color, fontsize=8)'''

    # 添加颜色条
    if show_colorbar:
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Cosine Similarity', rotation=270, labelpad=15)

    # 调整布局
    plt.tight_layout()

    # 保存图像
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"热图已保存至: {save_path}")

    plt.show()

    return similarity_np, fig, ax


def plot_weight_difference_scatter(weight_matrix1, weight_matrix2,
                                   title="Weight Change vs Similarity Scatter Plot",
                                   dim=1,
                                   figsize=(14, 10),
                                   color_by_density=False,
                                   show_trend_line=False,
                                   show_contour=False,
                                   annotate_outliers=False,
                                   top_k_outliers=5,
                                   save_path=None):
    """
    绘制权重矩阵差值散点图

    参数:
    ----------
    weight_matrix1 : torch.Tensor
        第一个权重矩阵，形状为 (n_neurons, n_features)
    weight_matrix2 : torch.Tensor
        第二个权重矩阵，形状与weight_matrix1相同
    title : str, 可选
        图表标题
    figsize : tuple, 可选
        图形大小
    color_by_density : bool, 可选
        是否根据点密度着色
    show_trend_line : bool, 可选
        是否显示趋势线
    show_contour : bool, 可选
        是否显示密度等高线
    annotate_outliers : bool, 可选
        是否标注异常点
    top_k_outliers : int, 可选
        标注的异常点数量
    save_path : str, 可选
        保存路径

    返回:
    -------
    scatter_data : dict
        散点图数据
    fig : matplotlib.figure.Figure
        图形对象
    ax : matplotlib.axes.Axes
        坐标轴对象
    """

    # 确保输入是PyTorch张量
    if not isinstance(weight_matrix1, torch.Tensor):
        weight_matrix1 = torch.tensor(weight_matrix1, dtype=torch.float32)
    if not isinstance(weight_matrix2, torch.Tensor):
        weight_matrix2 = torch.tensor(weight_matrix2, dtype=torch.float32)

    # 检查形状是否相同
    if weight_matrix1.shape != weight_matrix2.shape:
        raise ValueError(f"权重矩阵形状必须相同，但得到 {weight_matrix1.shape} 和 {weight_matrix2.shape}")

    # 确保是2D张量
    if weight_matrix1.dim() != 2:
        raise ValueError(f"权重矩阵必须是2D矩阵，但得到的是{weight_matrix1.dim()}D")

    n_neurons = weight_matrix1.shape[0]

    # 1. 计算第一个权重矩阵的行向量间的余弦相似度
    normalized1 = weight_matrix1 / torch.norm(weight_matrix1, dim=dim, keepdim=True)
    if dim == 1:
        similarity_matrix = torch.mm(normalized1, normalized1.T)
    elif dim == 0:
        similarity_matrix = torch.mm(normalized1.T, normalized1)
    similarity_matrix_np = similarity_matrix.detach().cpu().numpy()

    # 2. 对于每个行向量，计算与其他向量余弦相似度绝对值的最大值
    # 排除自身（将对角线设为最小值）
    similarity_matrix_abs = np.abs(similarity_matrix_np)
    np.fill_diagonal(similarity_matrix_abs, -np.inf)  # 排除自身
    max_similarities = np.max(similarity_matrix_abs, axis=1)

    # 3. 计算两个权重矩阵对应行向量的差值大小
    # 使用L2范数（欧氏距离）衡量差值
    weight_differences = torch.norm(weight_matrix2 - weight_matrix1, dim=dim)
    weight_differences_np = weight_differences.detach().cpu().numpy()

    # 4. 计算其他统计量
    # 相对变化率（差值相对于原始权重大小的比例）
    weight_norms = torch.norm(weight_matrix1, dim=dim)
    relative_changes = weight_differences / (weight_norms + 1e-8)  # 防止除零
    relative_changes_np = relative_changes.detach().cpu().numpy()

    # 创建散点图
    fig, ax = plt.subplots(figsize=figsize)

    # 准备数据
    x_data = max_similarities
    y_data = weight_differences_np

    # 点的颜色（根据密度或相对变化率）

    scatter = ax.scatter(x_data, y_data, c=relative_changes_np, s=50, cmap='RdYlBu_r',
                             alpha=0.7, edgecolors='black', linewidth=0.5)
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label('Relative Change', rotation=270, labelpad=15)

    # 添加等高线（如果开启）
    if show_contour:
        from scipy.stats import gaussian_kde
        kde = gaussian_kde(np.vstack([x_data, y_data]))
        x_range = np.linspace(x_data.min(), x_data.max(), 100)
        y_range = np.linspace(y_data.min(), y_data.max(), 100)
        X, Y = np.meshgrid(x_range, y_range)
        Z = kde(np.vstack([X.ravel(), Y.ravel()])).reshape(X.shape)
        ax.contour(X, Y, Z, levels=8, colors='black', alpha=0.5, linewidths=0.5)

    # 添加趋势线（如果开启）
    if show_trend_line:
        # 使用线性回归
        from scipy.stats import linregress
        slope, intercept, r_value, p_value, std_err = linregress(x_data, y_data)

        # 生成趋势线
        x_trend = np.linspace(x_data.min(), x_data.max(), 100)
        y_trend = slope * x_trend + intercept

        ax.plot(x_trend, y_trend, 'r--', linewidth=2,
                label=f'Trend line: y = {slope:.3f}x + {intercept:.3f}\n(R² = {r_value ** 2:.3f})')

        # 添加相关系数文本
        ax.text(0.02, 0.98, f'Correlation: {r_value:.3f}\np-value: {p_value:.3e}',
                transform=ax.transAxes, fontsize=12,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    # 标注异常点（如果开启）
    if annotate_outliers and top_k_outliers > 0:
        # 根据y值（权重差值）找出最大的top_k个点
        top_indices = np.argsort(y_data)[-top_k_outliers:]

        for idx in top_indices:
            ax.annotate(f'N{idx}',
                        xy=(x_data[idx], y_data[idx]),
                        xytext=(5, 5),
                        textcoords='offset points',
                        fontsize=9,
                        color='red',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.5),
                        arrowprops=dict(arrowstyle='->', color='red', lw=0.5))

    # 设置图表属性
    ax.set_xlabel('Max Absolute Cosine Similarity (to other neurons)', fontsize=14)
    ax.set_ylabel('Weight Difference (L2 norm)', fontsize=14)
    ax.set_title(title, fontsize=16, fontweight='bold')

    # 添加网格
    ax.grid(True, alpha=0.3, linestyle='--')

    # 添加图例（如果有趋势线）
    if show_trend_line:
        ax.legend(fontsize=12, loc='best')

    # 计算并显示统计信息
    stats_text = f"""
    Statistics:
    - Neurons: {n_neurons}
    - Mean max similarity: {x_data.mean():.3f}
    - Mean weight difference: {y_data.mean():.3f}
    - Median weight difference: {np.median(y_data):.3f}
    - Max weight difference: {y_data.max():.3f}
    - Min weight difference: {y_data.min():.3f}
    """

    # 添加文本框
    props = dict(boxstyle='round', facecolor='lightblue', alpha=0.8)
    ax.text(0.02, 0.02, stats_text, transform=ax.transAxes, fontsize=11,
            verticalalignment='bottom', bbox=props, fontfamily='monospace')

    # 调整布局
    plt.tight_layout()

    # 保存图像
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"散点图已保存至: {save_path}")

    plt.show()

    # 返回数据
    scatter_data = {
        'max_similarities': x_data,
        'weight_differences': y_data,
        'relative_changes': relative_changes_np,
        'neuron_indices': np.arange(n_neurons),
        'similarity_matrix': similarity_matrix_np
    }

    return scatter_data, fig, ax

def plot_weight_difference_compare_scatter(weight_matrix1, weight_matrix2, weight_matrix3,
                                        name2, name3,
                                   title="Weight Change Comparison Scatter Plot",
                                   dim=1,
                                   figsize=(14, 10),
                                   color_by_density=False,
                                   show_trend_line=False,
                                   show_contour=False,
                                   annotate_outliers=False,
                                   top_k_outliers=5,
                                   save_path=None):
    """
    绘制权重矩阵差值散点图

    参数:
    ----------
    weight_matrix1 : torch.Tensor
        第一个权重矩阵，形状为 (n_neurons, n_features)
    weight_matrix2 : torch.Tensor
        第二个权重矩阵，形状与weight_matrix1相同
    title : str, 可选
        图表标题
    figsize : tuple, 可选
        图形大小
    color_by_density : bool, 可选
        是否根据点密度着色
    show_trend_line : bool, 可选
        是否显示趋势线
    show_contour : bool, 可选
        是否显示密度等高线
    annotate_outliers : bool, 可选
        是否标注异常点
    top_k_outliers : int, 可选
        标注的异常点数量
    save_path : str, 可选
        保存路径

    返回:
    -------
    scatter_data : dict
        散点图数据
    fig : matplotlib.figure.Figure
        图形对象
    ax : matplotlib.axes.Axes
        坐标轴对象
    """

    # 确保输入是PyTorch张量
    if not isinstance(weight_matrix1, torch.Tensor):
        weight_matrix1 = torch.tensor(weight_matrix1, dtype=torch.float32)
    if not isinstance(weight_matrix2, torch.Tensor):
        weight_matrix2 = torch.tensor(weight_matrix2, dtype=torch.float32)

    # 检查形状是否相同
    if weight_matrix1.shape != weight_matrix2.shape:
        raise ValueError(f"权重矩阵形状必须相同，但得到 {weight_matrix1.shape} 和 {weight_matrix2.shape}")

    # 确保是2D张量
    if weight_matrix1.dim() != 2:
        raise ValueError(f"权重矩阵必须是2D矩阵，但得到的是{weight_matrix1.dim()}D")

    n_neurons = weight_matrix1.shape[0]

    # 1. 计算第一个权重矩阵的行向量间的余弦相似度
    normalized1 = weight_matrix1 / torch.norm(weight_matrix1, dim=dim, keepdim=True)
    if dim == 1:
        similarity_matrix = torch.mm(normalized1, normalized1.T)
    elif dim == 0:
        similarity_matrix = torch.mm(normalized1.T, normalized1)
    similarity_matrix_np = similarity_matrix.detach().cpu().numpy()

    # 2. 对于每个行向量，计算与其他向量余弦相似度绝对值的最大值
    # 排除自身（将对角线设为最小值）
    similarity_matrix_abs = np.abs(similarity_matrix_np)
    np.fill_diagonal(similarity_matrix_abs, -np.inf)  # 排除自身
    max_similarities = np.max(similarity_matrix_abs, axis=1)

    # 3. 计算两个权重矩阵对应行向量的差值大小
    # 使用L2范数（欧氏距离）衡量差值
    weight_differences = torch.norm(weight_matrix2 - weight_matrix1, dim=dim)
    weight_differences_13 = torch.norm(weight_matrix3 - weight_matrix1, dim=dim)
    weight_differences_np = weight_differences.detach().cpu().numpy()
    weight_differences_13_np = weight_differences_13.detach().cpu().numpy()

    # 4. 计算其他统计量
    # 相对变化率（差值相对于原始权重大小的比例）
    '''weight_norms = torch.norm(weight_matrix1, dim=dim)
    relative_changes = weight_differences / (weight_norms + 1e-8)  # 防止除零
    relative_changes_np = relative_changes.detach().cpu().numpy()'''

    # 创建散点图
    fig, ax = plt.subplots(figsize=figsize)

    # 准备数据
    x_data = weight_differences_13_np
    y_data = weight_differences_np

    # 点的颜色（根据密度或相对变化率）

    scatter = ax.scatter(x_data, y_data,c=max_similarities, s=50, cmap='RdYlBu_r',
                             alpha=0.7, edgecolors='black', linewidth=0.5)
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label('Max Cosine Similarity', rotation=270, labelpad=15)

    # 添加等高线（如果开启）
    if show_contour:
        from scipy.stats import gaussian_kde
        kde = gaussian_kde(np.vstack([x_data, y_data]))
        x_range = np.linspace(x_data.min(), x_data.max(), 100)
        y_range = np.linspace(y_data.min(), y_data.max(), 100)
        X, Y = np.meshgrid(x_range, y_range)
        Z = kde(np.vstack([X.ravel(), Y.ravel()])).reshape(X.shape)
        ax.contour(X, Y, Z, levels=8, colors='black', alpha=0.5, linewidths=0.5)

    # 添加趋势线（如果开启）
    if show_trend_line:
        # 使用线性回归
        from scipy.stats import linregress
        slope, intercept, r_value, p_value, std_err = linregress(x_data, y_data)

        # 生成趋势线
        x_trend = np.linspace(x_data.min(), x_data.max(), 100)
        y_trend = slope * x_trend + intercept

        ax.plot(x_trend, y_trend, 'r--', linewidth=2,
                label=f'Trend line: y = {slope:.3f}x + {intercept:.3f}\n(R² = {r_value ** 2:.3f})')

        # 添加相关系数文本
        ax.text(0.02, 0.98, f'Correlation: {r_value:.3f}\np-value: {p_value:.3e}',
                transform=ax.transAxes, fontsize=12,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    # 标注异常点（如果开启）
    if annotate_outliers and top_k_outliers > 0:
        # 根据y值（权重差值）找出最大的top_k个点
        top_indices = np.argsort(y_data)[-top_k_outliers:]

        for idx in top_indices:
            ax.annotate(f'N{idx}',
                        xy=(x_data[idx], y_data[idx]),
                        xytext=(5, 5),
                        textcoords='offset points',
                        fontsize=9,
                        color='red',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.5),
                        arrowprops=dict(arrowstyle='->', color='red', lw=0.5))

    # 设置图表属性
    ax.set_xlabel(f'Weight Difference (L2 norm) of {name3}', fontsize=14)
    ax.set_ylabel(f'Weight Difference (L2 norm) of {name2}', fontsize=14)
    ax.set_title(title, fontsize=16, fontweight='bold')

    # 添加网格
    ax.grid(True, alpha=0.3, linestyle='--')

    # 添加图例（如果有趋势线）
    if show_trend_line:
        ax.legend(fontsize=12, loc='best')

    # 计算并显示统计信息
    stats_text = f"""
    Statistics:
    - Mean max similarity: {x_data.mean():.3f}
    - Mean weight difference: {y_data.mean():.3f}
    - Median weight difference: {np.median(y_data):.3f}
    - Max weight difference: {y_data.max():.3f}
    - Min weight difference: {y_data.min():.3f}
    """

    # 添加文本框
    #props = dict(boxstyle='round', facecolor='lightblue', alpha=0.8)
    #ax.text(0.02, 0.02, stats_text, transform=ax.transAxes, fontsize=11,
            #verticalalignment='bottom', bbox=props, fontfamily='monospace')

    # 调整布局
    plt.tight_layout()

    # 保存图像
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"散点图已保存至: {save_path}")

    plt.show()

    # 返回数据
    scatter_data = {
        'max_similarities': x_data,
        'weight_differences': y_data,
        #'relative_changes': relative_changes_np,
        #'neuron_indices': np.arange(n_neurons),
        #'similarity_matrix': similarity_matrix_np
    }

    return scatter_data, fig, ax

def main(checkpoint_path, save_path):
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    print("Load pre-trained checkpoint from: %s" % checkpoint_path)
    checkpoint_model = checkpoint['model'] if 'model' in checkpoint else checkpoint
    for k in checkpoint_model.keys():
        print(k)
        if "attn.q_proj.weight" in k:
            q_weight = checkpoint_model[k]
            plot_cosine_similarity_heatmap(q_weight, dim=1, save_path=os.path.join(save_path, f"{k}_cos.png"))
            plot_cosine_similarity_histogram(q_weight, dim=1, save_path=os.path.join(save_path, f"{k}_hist.png"))
        elif "attn.k_proj.weight" in k:
            k_weight = checkpoint_model[k]
            plot_cosine_similarity_heatmap(k_weight, dim=1, save_path=os.path.join(save_path, f"{k}_cos.png"))
            plot_cosine_similarity_histogram(k_weight, dim=1, save_path=os.path.join(save_path, f"{k}_hist.png"))
        elif "attn.v_proj.weight" in k:
            v_weight = checkpoint_model[k]
            plot_cosine_similarity_heatmap(v_weight, dim=1, save_path=os.path.join(save_path, f"{k}_cos.png"))
            plot_cosine_similarity_histogram(v_weight, dim=1, save_path=os.path.join(save_path, f"{k}_hist.png"))
        elif "attn.proj.weight" in k:
            o_weight = checkpoint_model[k]
            plot_cosine_similarity_heatmap(o_weight, dim=0, save_path=os.path.join(save_path, f"{k}_cos.png"))
            plot_cosine_similarity_histogram(o_weight, dim=0, save_path=os.path.join(save_path, f"{k}_hist.png"))
        elif "fc1.weight" in k:
            up_weight = checkpoint_model[k]
            plot_cosine_similarity_heatmap(up_weight, dim=1, save_path=os.path.join(save_path, f"{k}_cos.png"))
            plot_cosine_similarity_histogram(up_weight, dim=1, save_path=os.path.join(save_path, f"{k}_hist.png"))
        elif "fc2.weight" in k:
            down_weight = checkpoint_model[k]
            plot_cosine_similarity_heatmap(down_weight, dim=0, save_path=os.path.join(save_path, f"{k}_cos.png"))
            plot_cosine_similarity_histogram(down_weight, dim=0, save_path=os.path.join(save_path, f"{k}_hist.png"))

def weight_difference(pretrained_path, finetuned_path, save_path):
    pretrained_checkpoint = torch.load(pretrained_path, map_location='cpu', weights_only=False)
    finetuned_checkpoint = torch.load(finetuned_path, map_location='cpu', weights_only=False)

    print("Load pre-trained checkpoint from: %s" % pretrained_path)
    pretrained_model = pretrained_checkpoint['model'] if 'model' in pretrained_checkpoint else pretrained_checkpoint
    finetuned_model = finetuned_checkpoint['model'] if 'model' in finetuned_checkpoint else finetuned_checkpoint

    for k in pretrained_model.keys():
        print(k)
        if "attn.q_proj.weight" in k:
            pretrained_q_weight = pretrained_model[k]
            finetuned_q_weight = finetuned_model[k]
            plot_weight_difference_scatter(pretrained_q_weight, finetuned_q_weight, dim=1, save_path=os.path.join(save_path, f"{k}_diff_compare.png"))
        elif "attn.k_proj.weight" in k:
            pretrained_k_weight = pretrained_model[k]
            finetuned_k_weight = finetuned_model[k]
            plot_weight_difference_scatter(pretrained_k_weight, finetuned_k_weight, dim=1, save_path=os.path.join(save_path, f"{k}_diff_compare.png"))
        elif "attn.v_proj.weight" in k:
            pretrained_v_weight = pretrained_model[k]
            finetuned_v_weight = finetuned_model[k]
            plot_weight_difference_scatter(pretrained_v_weight, finetuned_v_weight, dim=1, save_path=os.path.join(save_path, f"{k}_diff_compare.png"))
        elif "attn.proj.weight" in k:
            pretrained_o_weight = pretrained_model[k]
            finetuned_o_weight = finetuned_model[k]
            plot_weight_difference_scatter(pretrained_o_weight, finetuned_o_weight, dim=0,save_path=os.path.join(save_path, f"{k}_diff_compare.png"))
        elif "fc1.weight" in k:
            pretrained_up_weight = pretrained_model[k]
            finetuned_up_weight = finetuned_model[k]
            plot_weight_difference_scatter(pretrained_up_weight, finetuned_up_weight, dim=1, save_path=os.path.join(save_path, f"{k}_diff_compare.png"))
        elif "fc2.weight" in k:
            pretrained_down_weight = pretrained_model[k]
            finetuned_down_weight = finetuned_model[k]
            plot_weight_difference_scatter(pretrained_down_weight, finetuned_down_weight, dim=0, save_path=os.path.join(save_path, f"{k}_diff_compare.png"))

def weight_difference_compare(pretrained_path, finetune1, finetune2, name1, name2, save_path):
    pretrained_checkpoint = torch.load(pretrained_path, map_location='cpu', weights_only=False)
    finetuned_checkpoint1 = torch.load(finetune1, map_location='cpu', weights_only=False)
    finetuned_checkpoint2 = torch.load(finetune2, map_location='cpu', weights_only=False)

    print("Load pre-trained checkpoint from: %s" % pretrained_path)
    pretrained_model = pretrained_checkpoint['model'] if 'model' in pretrained_checkpoint else pretrained_checkpoint
    finetuned_model1 = finetuned_checkpoint1['model'] if 'model' in finetuned_checkpoint1 else finetuned_checkpoint1
    finetuned_model2 = finetuned_checkpoint2['model'] if 'model' in finetuned_checkpoint2 else finetuned_checkpoint2

    for k in pretrained_model.keys():
        print(k)
        if "attn.q_proj.weight" in k:
            pretrained_q_weight = pretrained_model[k]
            finetuned_q_weight1 = finetuned_model1[k]
            finetuned_q_weight2 = finetuned_model2[k]
            plot_weight_difference_compare_scatter(pretrained_q_weight, finetuned_q_weight1, finetuned_q_weight2, name2=name1, name3=name2, dim=1, save_path=os.path.join(save_path, f"{k}_{name1}_{name2}_diff_compare.png"))
        elif "attn.k_proj.weight" in k:
            pretrained_k_weight = pretrained_model[k]
            finetuned_k_weight1 = finetuned_model1[k]
            finetuned_k_weight2 = finetuned_model2[k]
            plot_weight_difference_compare_scatter(pretrained_k_weight, finetuned_k_weight1, finetuned_k_weight2, name2=name1, name3=name2, dim=1, save_path=os.path.join(save_path, f"{k}_{name1}_{name2}_diff_compare.png"))
        elif "attn.v_proj.weight" in k:
            pretrained_v_weight = pretrained_model[k]
            finetuned_v_weight1 = finetuned_model1[k]
            finetuned_v_weight2 = finetuned_model2[k]
            plot_weight_difference_compare_scatter(pretrained_v_weight, finetuned_v_weight1, finetuned_v_weight2, name2=name1, name3=name2, dim=1, save_path=os.path.join(save_path, f"{k}_{name1}_{name2}_diff_compare.png"))
        elif "attn.proj.weight" in k:
            pretrained_o_weight = pretrained_model[k]
            finetuned_o_weight1 = finetuned_model1[k]
            finetuned_o_weight2 = finetuned_model2[k]
            plot_weight_difference_compare_scatter(pretrained_o_weight, finetuned_o_weight1, finetuned_o_weight2,name2=name1, name3=name2, dim=0, save_path=os.path.join(save_path, f"{k}_{name1}_{name2}_diff_compare.png"))
        elif "fc1.weight" in k:
            pretrained_up_weight = pretrained_model[k]
            finetuned_up_weight1 = finetuned_model1[k]
            finetuned_up_weight2 = finetuned_model2[k]
            plot_weight_difference_compare_scatter(pretrained_up_weight, finetuned_up_weight1, finetuned_up_weight2, name2=name1, name3=name2, dim=1, save_path=os.path.join(save_path, f"{k}_{name1}_{name2}_diff_compare.png"))
        elif "fc2.weight" in k:
            pretrained_down_weight = pretrained_model[k]
            finetuned_down_weight1 = finetuned_model1[k]
            finetuned_down_weight2 = finetuned_model2[k]
            plot_weight_difference_compare_scatter(pretrained_down_weight, finetuned_down_weight1, finetuned_down_weight2, name2=name1, name3=name2, dim=0, save_path=os.path.join(save_path, f"{k}_{name1}_{name2}_diff_compare.png"))


if __name__ == '__main__':
    #weight_difference("/home/ma-user/modelarts/work/ytchen/LowRankOptim/AdaptFormer-main/pretrained_checkpoint/mae_pretrain_vit_b.pth", "/home/ma-user/modelarts/work/ytchen/LowRankOptim/AdaptFormer-main/result/result_svhn_fulltune/checkpoint-99.pth", "/home/ma-user/modelarts/work/ytchen/LowRankOptim/AdaptFormer-main/result/result_svhn_fulltune/fig")
    weight_difference_compare("/home/ma-user/modelarts/work/ytchen/LowRankOptim/AdaptFormer-main/pretrained_checkpoint/mae_pretrain_vit_b.pth", "/home/ma-user/modelarts/work/ytchen/LowRankOptim/AdaptFormer-main/result/result_svhn_fulltune/checkpoint-99.pth", "/home/ma-user/modelarts/work/ytchen/LowRankOptim/AdaptFormer-main/result/cifar100_fulltune/checkpoint-99.pth", "svhn", "cifar100", "/home/ma-user/modelarts/work/ytchen/LowRankOptim/AdaptFormer-main/result/fig_diff_compare")
    weight_difference_compare("/home/ma-user/modelarts/work/ytchen/LowRankOptim/AdaptFormer-main/pretrained_checkpoint/mae_pretrain_vit_b.pth", "/home/ma-user/modelarts/work/ytchen/LowRankOptim/AdaptFormer-main/result/result_svhn_fulltune/checkpoint-99.pth", "/home/ma-user/modelarts/work/ytchen/LowRankOptim/AdaptFormer-main/result/result_food101_fulltune/checkpoint-99.pth", "svhn", "food101", "/home/ma-user/modelarts/work/ytchen/LowRankOptim/AdaptFormer-main/result/fig_diff_compare")
    weight_difference_compare("/home/ma-user/modelarts/work/ytchen/LowRankOptim/AdaptFormer-main/pretrained_checkpoint/mae_pretrain_vit_b.pth", "/home/ma-user/modelarts/work/ytchen/LowRankOptim/AdaptFormer-main/result/result_food101_fulltune/checkpoint-99.pth", "/home/ma-user/modelarts/work/ytchen/LowRankOptim/AdaptFormer-main/result/cifar100_fulltune/checkpoint-99.pth", "food101", "cifar100", "/home/ma-user/modelarts/work/ytchen/LowRankOptim/AdaptFormer-main/result/fig_diff_compare")
    #main("/home/ma-user/modelarts/work/ytchen/LowRankOptim/AdaptFormer-main/result/result_svhn_fulltune/checkpoint-99.pth", "/home/ma-user/modelarts/work/ytchen/LowRankOptim/AdaptFormer-main/result/result_svhn_fulltune/fig")