import torchvision.transforms as transforms
import torchvision.datasets as datasets
from dataset.imbalance_cifar import IMBALANCECIFAR10, IMBALANCECIFAR100
import warnings
import torch
import numpy as np
from skimage.filters import gaussian as gblur
import torchvision.transforms as trn
from dataset.tinyimages_80mn_loader import TinyImages


RGB_statistics = {
    'cifar10': {
        'mean': (0.4914, 0.4822, 0.4465),
        'std': (0.2023, 0.1994, 0.2010)
    },
    'cifar100': {
        'mean': (0.4914, 0.4822, 0.4465),
        'std': (0.2023, 0.1994, 0.2010)
    }
}



def create_dataset(args):

    transform_train, transform_val = get_data_transform(args.dataset)

    if args.dataset == 'cifar10':
        train_dataset = IMBALANCECIFAR10(root='./data', imb_type=args.imb_type, imb_factor=args.imb_factor,
                                         rand_number=args.rand_number, train=True, download=True,
                                         transform=transform_train)
        val_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_val)
    elif args.dataset == 'cifar100':
        train_dataset = IMBALANCECIFAR100(root='./data', imb_type=args.imb_type, imb_factor=args.imb_factor,
                                          rand_number=args.rand_number, train=True, download=True,
                                          transform=transform_train)
        val_dataset = datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_val)
    else:
        warnings.warn('Dataset is not listed')
        return

    cls_num_list = train_dataset.get_cls_num_list()
    print('cls num list:')
    print(cls_num_list)
    return train_dataset, val_dataset, cls_num_list



def create_ood_dataset(args):
    mean, std = RGB_statistics[args.dataset]['mean'], RGB_statistics[args.dataset]['std']

    if args.aux_set == "TinyImages":
        ood_dataset = TinyImages(transform=trn.Compose(
            [trn.ToTensor(), trn.ToPILImage(), trn.RandomCrop(32, padding=4),
             trn.RandomHorizontalFlip(), trn.ToTensor(), trn.Normalize(mean, std)]), data_num=args.ood_num)
    elif args.aux_set == "cifar100":
        transform_train,_ = get_data_transform("cifar10")
        import torchvision
        ood_dataset = torchvision.datasets.CIFAR100(root="./data", train=True,
                 transform=transform_train, target_transform=None,
                 download=False)

    return ood_dataset


def create_ood_noise(noise_type, ood_num_examples, num_to_avg):
    if noise_type == "Gaussian":
        dummy_targets = torch.ones(ood_num_examples * num_to_avg)
        ood_data = torch.from_numpy(np.float32(np.clip(
            np.random.normal(size=(ood_num_examples * num_to_avg, 3, 32, 32), scale=0.5), -1, 1)))
        ood_data = torch.utils.data.TensorDataset(ood_data, dummy_targets)
    elif noise_type == "Rademacher":
        dummy_targets = torch.ones(ood_num_examples * num_to_avg)
        ood_data = torch.from_numpy(np.random.binomial(
            n=1, p=0.5, size=(ood_num_examples * num_to_avg, 3, 32, 32)).astype(np.float32)) * 2 - 1
        ood_data = torch.utils.data.TensorDataset(ood_data, dummy_targets)
    elif noise_type == "Blob":
        ood_data = np.float32(np.random.binomial(n=1, p=0.7, size=(ood_num_examples * num_to_avg, 32, 32, 3)))
        for i in range(ood_num_examples * num_to_avg):
            ood_data[i] = gblur(ood_data[i], sigma=1.5, multichannel=False)
            ood_data[i][ood_data[i] < 0.75] = 0.0

        dummy_targets = torch.ones(ood_num_examples * num_to_avg)
        ood_data = torch.from_numpy(ood_data.transpose((0, 3, 1, 2))) * 2 - 1
        ood_data = torch.utils.data.TensorDataset(ood_data, dummy_targets)
    return ood_data


def get_data_transform(dataset):

    rgb_mean, rbg_std = RGB_statistics[dataset]['mean'], RGB_statistics[dataset]['std']

    data_transforms = {
        'cifar10': {
            'train': transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(rgb_mean, rbg_std),
            ]),
            'val': transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(rgb_mean, rbg_std),
            ])
        },
        'cifar100': {
            'train': transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(rgb_mean, rbg_std),
            ]),
            'val': transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(rgb_mean, rbg_std),
            ])
        }
    }


    return data_transforms[dataset]['train'], data_transforms[dataset]['val']