import numpy as np
import torch
from bisect import bisect_left
import random
class TinyImages(torch.utils.data.Dataset):

    def __init__(self, transform=None, exclude_cifar=True, data_num=50000):

        data_file = open('../../data/tiny_images.bin', "rb")
        self.data_num = data_num
        def load_image(idx):
            data_file.seek(idx * 3072)
            data = data_file.read(3072)
            return np.fromstring(data, dtype='uint8').reshape(32, 32, 3, order="F")

        self.load_image = load_image
        self.offset = 0     # offset index

        self.transform = transform
        self.exclude_cifar = exclude_cifar
        id_list = list(range(79302017))

        if exclude_cifar:
            self.cifar_idxs = []
            with open('./dataset/80mn_cifar_idxs.txt', 'r') as idxs:
                for idx in idxs:
                    # indices in file take the 80mn database to start at 1, hence "- 1"
                    self.cifar_idxs.append(int(idx) - 1)

            # hash table option
            self.cifar_idxs = set(self.cifar_idxs)
            # self.in_cifar = lambda x: x in self.cifar_idxs
        self.id_no_cifar = [x for x in id_list if x not in self.cifar_idxs]

        if self.data_num == -1:
            self.id_sample = self.id_no_cifar
        else:
            self.id_sample = random.sample(self.id_no_cifar, data_num)

        self.labels = None

    def __getitem__(self, index):
        id = self.id_sample[index]
        img = self.load_image(id)
        if self.transform is not None:
            img = self.transform(img)

        if self.labels:
            return img, self.labels[index]
        return img, 0, index  # 0 is the class

    def resample(self):
        if self.data_num == -1:
            self.id_sample = self.id_no_cifar
        else:
            self.id_sample = random.sample(self.id_no_cifar, self.data_num)


    def __len__(self):

        return len(self.id_sample)


