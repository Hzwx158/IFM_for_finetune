import os
import torchvision.datasets
import torchvision.transforms as transforms
from pathlib import Path

def prepare_dataset(data_path):
    dataset_train = torchvision.datasets.CIFAR100(os.path.join(data_path, 'cifar100'), transform=transforms.ToTensor(), train=True, download=True)
    dataset_val = torchvision.datasets.CIFAR100(os.path.join(data_path, 'cifar100'), transform=transforms.ToTensor(), train=False, download=True)
    dataset_train = torchvision.datasets.SVHN(os.path.join(data_path, 'svhn'), split='train', transform=transforms.ToTensor(), download=True)
    dataset_val = torchvision.datasets.SVHN(os.path.join(data_path, 'svhn'), split='test', transform=transforms.ToTensor(), download=True)
    from datasets.food101 import Food101
    dataset_train = Food101(os.path.join(data_path, 'food101'), split='train', transform=transforms.ToTensor(), download=True)
    dataset_val = Food101(os.path.join(data_path, 'food101'), split='test', transform=transforms.ToTensor(), download=True)
    
if __name__ == "__main__":
    root = Path(__file__).absolute().parent
    prepare_dataset(str(root/"data"))