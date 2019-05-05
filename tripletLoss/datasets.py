"""datasets
"""

import sys
import os
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image
import pandas as pd

from torch.utils.data import Dataset


class SpecificDataset(object):
    """load specific dataset
    1. dataset would be put on ../data/
    2. avaialable dataset: MNIST, cifar10, cifar100
    """

    def __init__(self, dataset_name='MNIST', data_augmentation=False, iter_no=0):
        self.iter_no = iter_no
        self.dataset_name = dataset_name
        self.data_augmentation = data_augmentation
        self.__load()


    def __load(self):
        if self.dataset_name == 'cifar10':
            self.n_classes = 10
            self.gap = False
            self.load_CIFAR10()
        elif self.dataset_name == 'cifar100':
            self.n_classes = 100
            self.gap = False
            self.load_CIFAR100()
        elif self.dataset_name == 'MNIST':
            self.n_classes = 10
            self.gap = False
            self.load_MNIST()
        elif self.dataset_name == 'SkinLesion':
            self.n_classes = 7
            self.gap = True
            self.load_skin_lesion()
        elif self.dataset_name == 'miniImageNet':
            self.n_classes = 100
            self.gap = True
            self.load_miniImageNet()
        elif self.dataset_name == 'CheXpert':
            self.n_classes = 14
            self.gap = True
            self.load_CheXpert()
        elif self.dataset_name == 'SD198':
            self.n_classes = 198
            self.gap = True
            self.load_sd_198()
        else:
            print('Must provide valid dataset')
            sys.exit(-1)

        self.train_dataset.dataset_name = self.dataset_name
        self.test_dataset.dataset_name = self.dataset_name

    def load_MNIST(self):
        self.mean, self.std = 0.1307, 0.3081

        train_transform = transforms.Compose([
            transforms.Grayscale(3),
            transforms.ToTensor(),
            transforms.Normalize((self.mean,), (self.std,))])

        test_transform = transforms.Compose([
            transforms.Grayscale(3),
            transforms.ToTensor(),
            transforms.Normalize((self.mean,), (self.std,))])

        self.train_dataset = torchvision.datasets.MNIST('../data', train=True, download=False,
                                                        transform=train_transform)
        self.train_dataset.data = self.train_dataset.data.numpy()
        self.test_dataset = torchvision.datasets.MNIST('../data', train=False, download=False, transform=test_transform)
        # self.test_dataset.data = self.test_dataset.data.numpy()
        self.width, self.height = 28, 28
        self.channels = 3

        self.classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')

    def load_CIFAR10(self):
        self.mean = (0.49, 0.48, 0.45)
        self.std = (0.25, 0.24, 0.26)
        train_transform = transforms.Compose([
            # transforms.RandomHorizontalFlip(),
            # transforms.RandomCrop(size=32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize(self.mean,
                                 self.std)])

        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(self.mean,
                                 self.std)])

        self.train_dataset = torchvision.datasets.CIFAR10(root='../data', train=True, download=False,
                                                          transform=train_transform)
        self.test_dataset = torchvision.datasets.CIFAR10(root='../data', train=False, download=False,
                                                         transform=test_transform)
        self.classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

        self.width, self.height = 32, 32
        self.channels = 3

    def load_CIFAR100(self):
        self.mean = (0.51, 0.49, 0.44)
        self.std = (0.27, 0.26, 0.27)

        train_transform = transforms.Compose([])
        test_transform_fc = transforms.Compose([])  # use five crop
        if self.data_augmentation:
            train_transform.transforms.append(transforms.RandomHorizontalFlip())
            train_transform.transforms.append(transforms.RandomCrop((32, 32), padding=4))
            test_transform_fc.transforms.append(transforms.Pad(4))
            test_transform_fc.transforms.append(transforms.FiveCrop(32))
            test_transform_fc.transforms.append(transforms.Lambda(lambda crops: torch.stack \
                ([transforms.Normalize(self.mean, self.std)(transforms.ToTensor()(crop)) for crop in crops])))

        train_transform.transforms.append(transforms.ToTensor())
        train_transform.transforms.append(transforms.Normalize(self.mean, self.std))
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std)
        ])

        self.train_dataset = torchvision.datasets.CIFAR100(root='../data', train=True, download=False,
                                                           transform=train_transform)
        self.test_dataset = torchvision.datasets.CIFAR100(root='../data', train=False, download=False,
                                                          transform=test_transform)
        self.test_dataset_fc = torchvision.datasets.CIFAR100(root='../data', train=False, download=False,
                                                             transform=test_transform_fc)

        self.classes = self.train_dataset.classes
        self.width, self.height = 32, 32
        self.channels = 3

    def load_skin_lesion(self):
        self.mean = (0.7626, 0.5453, 0.5714)
        self.std = (0.1404, 0.1519, 0.1685)

        test_transform_fc = transforms.Compose([transforms.Resize((224, 224)),
                                                transforms.Pad(28),
                                                transforms.FiveCrop(224),
                                                transforms.Lambda(lambda crops: torch.stack(
                                                    [transforms.Normalize(self.mean, self.std)(
                                                        transforms.ToTensor()(crop)) for crop in crops]))
                                                ])  # use five crop

        train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std)
        ])
        test_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std)
        ])

        self.train_dataset = SkinLesionDataset(train=True, transform=train_transform)

        self.test_dataset = SkinLesionDataset(train=False, transform=test_transform)

        self.test_dataset_fc = SkinLesionDataset(train=False, transform=test_transform_fc)

        self.width, self.height = 224, 224
        self.channels = 3
        self.classes = self.train_dataset.classes

    def load_miniImageNet(self):
        self.mean = (0.47, 0.45, 0.41)
        self.std = (0.28, 0.27, 0.29)

        train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std)])

        test_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std)])

        self.train_dataset = torchvision.datasets.ImageFolder('../data/miniImageNet/train', transform=train_transform)
        self.train_dataset.train = True
        self.train_dataset.data, _ = self.tuple2list(self.train_dataset)
        self.val_dataset = torchvision.datasets.ImageFolder('../data/miniImageNet/test', transform=test_transform)
        self.val_dataset.train = False
        self.val_dataset.data, _ = self.tuple2list(self.val_dataset)
        self.test_dataset = torchvision.datasets.ImageFolder('../data/miniImageNet/test', transform=test_transform)
        self.test_dataset.data, _ = self.tuple2list(self.test_dataset)
        self.test_dataset.train = False

        self.classes = self.train_dataset.classes
        self.width, self.height = 224, 224
        self.channels = 3


    def load_CheXpert(self):
        self.mean = (0.5013, 0.5013, 0.5013)
        self.std = (0.2911, 0.2911, 0.2911)

        test_transform_fc = transforms.Compose([transforms.Resize((224, 224)),
                                                transforms.Pad(12),
                                                transforms.FiveCrop(224),
                                                transforms.Lambda(lambda crops: torch.stack(
                                                    [transforms.Normalize(self.mean, self.std)(
                                                        transforms.ToTensor()(crop)) for crop in crops]))
                                                ])  # use five crop

        train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std)
        ])
        test_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std)
        ])

        self.train_dataset = CheXpertDataset(train=True, transform=train_transform)

        self.test_dataset = CheXpertDataset(train=False, transform=test_transform)

        self.test_dataset_fc = CheXpertDataset(train=False, transform=test_transform_fc)

        self.width, self.height = 224, 224
        self.channels = 3
        self.classes = self.train_dataset.classes

    def load_sd_198(self):
        """load sd-198"""

        self.mean = (0.592, 0.479, 0.451)
        self.std = (0.265, 0.245, 0.247)


        train_transform = transforms.Compose([])
        test_transform_fc = transforms.Compose([]) # use five crop

        if self.data_augmentation:
            train_transform.transforms.append(transforms.RandomHorizontalFlip())
            #train_transform.transforms.append(transforms.RandomResizedCrop((224, 224)))
            #train_transform.transforms.append(transforms.Resize((256, 256)))
            train_transform.transforms.append(transforms.Resize((300, 300)))
            train_transform.transforms.append(transforms.RandomCrop((224, 224)))
            test_transform_fc.transforms.append(transforms.Resize((300, 300)))
            #test_transform_fc.transforms.append(transforms.Resize((256, 256)))
            test_transform_fc.transforms.append(transforms.FiveCrop((224,224)))
            test_transform_fc.transforms.append(transforms.Lambda(lambda crops: torch.stack \
                ([transforms.Normalize(self.mean, self.std)(transforms.ToTensor()(crop)) for crop in crops])))

            #train_transform.transforms.append(transforms.RandomResizedCrop((224, 224)))
        train_transform.transforms.append(transforms.ToTensor())
        train_transform.transforms.append(transforms.Normalize(self.mean, self.std))

        test_transform = transforms.Compose([
            transforms.Resize((300,300)),
            #transforms.Resize((256,256)),
            transforms.CenterCrop((224,224)),
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std)
            ])

        self.train_dataset = SD198(train=True, transform=train_transform, iter_no=self.iter_no)
        self.test_dataset = SD198(train=False, transform=test_transform, iter_no=self.iter_no)
        self.test_dataset_fc = SD198(train=False, transform=test_transform_fc, iter_no=self.iter_no)
        self.classes = self.train_dataset.classes
        self.width, self.height = 32, 32
        self.channels = 3


    def tuple2list(self, pairs):
        data = []
        targets = []
        for img, label in pairs:
            data.append(img)
            targets.append(label)
        return data, targets


#class SampledDataset(Dataset):
#    """Sample data from original data"""
#
#    def __init__(self, dataset, channels, amount):
#        """
#        :param dataset:
#        :param channels:
#        :param amount: if amount = 0, do not sample
#        """
#        self.train = dataset.train
#        self.transform = dataset.transform
#        self.channels = channels
#
#        transform = transforms.Compose([
#            transform.RandomResizedCrop((224,224)),
#            #transforms.Resize((224, 224)),
#            transforms.ToTensor(),
#            transforms.Normalize(self.mean, self.std)
#            ])
#
#        self.train_dataset = SD198(train=True, transform=train_transform)
#
#        self.test_dataset = SD198(train=False, transform=test_transform)
#
#        self.width, self.height = 224, 224
#        self.channels = 3
#        #self.classes = ['MEL', 'NV', 'BCC', 'AKIEC', 'BKL', 'DF', 'VASC']




class SampledDataset(Dataset):
    """Sample data from original data"""

    def __init__(self, dataset, channels, amount):
        """
        :param dataset:
        :param channels:
        :param amount: if amount = 0, do not sample
        """
        self.train = dataset.train
        self.transform = dataset.transform
        self.channels = channels
        self.dataset_name = dataset.dataset_name

        if self.train:
            data = dataset.data
            labels = dataset.targets
            if amount != 0:
                labels = sample_labels(labels, np.unique(labels), amount)
                data, labels = select_data_by_labels(data, labels)
            self.targets = np.array(labels)
            self.data = data
        else:
            self.targets = dataset.targets
            self.data = dataset.data

        # dict store target:imgs
        self.target_img_dict = dict()
        self.targets_uniq = list(range(len(dataset.classes)))
        for target in self.targets_uniq:
            idx = np.nonzero(self.targets == target)[0]
            self.target_img_dict.update({target: idx})

    def __getitem__(self, index):
        if self.dataset_name in ['MNIST', 'cifar10', 'cifar100']:
            img = self.data[index]
            target = self.targets[index]
            img = Image.fromarray(img)
            img = self.transform(img)
        else:
            path = self.data[index]
            target = self.targets[index]
            img = pil_loader(path)
            img = self.transform(img)

        return img, target

    def __len__(self):
        return len(self.data)

    def shuffle(self):
        state = np.random.get_state()
        np.random.shuffle(self.data)
        np.random.set_state(state)
        np.random.shuffle(self.targets)


def select_data_by_labels(data, labels):
    """select_data_by_labels
    Args:
        data: [b, c, h, w]
        labels: [b]
    Returns:
        left_data: n_classes * len(classes)
        labels: n_classes * len(n_classes)
    """
    idxs = np.nonzero(labels)
    left_data = np.take(data, idxs, axis=0)[0]
    left_labels = np.take(labels, idxs, axis=0)[0]
    left_labels = [sum(x) for x in zip(len(left_labels) * [-1], left_labels)]
    return left_data, left_labels


def sample_labels(labels, classes, amount=100):
    """labels
    Args:
        labels: a list
        classes: [0,1,2 ...]
        amount: 100
    """
    count = [0] * len(classes)
    for i in classes:
        for idx, label in enumerate(labels):
            if label == i:
                if count[i] >= amount:
                    labels[idx] = -1
                else:
                    count[i] += 1
    labels = [sum(x) for x in zip(len(labels) * [1], labels)]
    return labels


class CheXpertDataset(Dataset):
    def __init__(self, train=True, transform=None, iterNo=1, data_dir='../data/CheXpert-v1.0-small'):
        self.transform = transform
        self.train = train
        self.data_dir = os.path.join(data_dir, 'train')
        self.data, self.targets = self.get_data(iterNo, data_dir)
        self.dataset_name = 'CheXpert'
        self.classes = ['No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity', 'Lung Lesion',
                        'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis', 'Pneumothorax', 'Pleural Effusion',
                        'Pleural Other', 'Fracture', 'Support Devices']

    def __getitem__(self, index):
        path = self.data[index]
        target = self.targets[index]
        img = pil_loader(path)
        if self.transform is not None:
            img = self.transform(img)

        return img, target

    def __len__(self):
        return len(self.data)

    def get_data(self, iterNo, data_dir):

        if self.train:
            csv = 'split_data/CheXpert_split_{}_train.csv'.format(iterNo)
        else:
            csv = 'split_data/CheXpert_split_{}_test.csv'.format(iterNo)

        fn = os.path.join(data_dir, csv)
        csvfile = pd.read_csv(fn, index_col=0)
        raw_data = csvfile.values

        data = []
        targets = []
        for (path, label) in raw_data:
            data.append(os.path.join('../data', path))
            targets.append(label)

        return data, targets


class SkinLesionDataset(Dataset):
    def __init__(self, train=True, transform=None, iterNo=1, data_dir='../data/SkinLesionDataset'):
        self.transform = transform
        self.train = train
        self.data_dir = os.path.join(data_dir, 'ISIC2018_Task3_Training_Input')
        self.data, self.targets = self.get_data(iterNo, data_dir)
        self.dataset_name = 'SkinLesion'
        self.classes = ['MEL', 'NV', 'BCC', 'AKIEC', 'BKL', 'DF', 'VASC']

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path = self.data[index]
        target = self.targets[index]
        img = pil_loader(path)
        if self.transform is not None:
            img = self.transform(img)

        return img, target

    def __len__(self):
        return len(self.data)

    def get_data(self, iterNo, data_dir):

        if self.train:
            csv = 'split_data/split_data_{}_fold_train.csv'.format(iterNo)
        else:
            csv = 'split_data/split_data_{}_fold_test.csv'.format(iterNo)

        fn = os.path.join(data_dir, csv)
        csvfile = pd.read_csv(fn, index_col=0)
        raw_data = csvfile.values

        data = []
        targets = []
        for (path, label) in raw_data:
            data.append(os.path.join(self.data_dir, path))
            targets.append(label)

        return data, targets

class SD198(Dataset):
    def __init__(self, train=True, transform=None, iter_no=0, data_dir='../data/SD-198'):
        self.transform = transform
        self.train = train
        self.data_dir = os.path.join(data_dir, 'images')
        self.data, self.targets = self.get_data(iter_no, data_dir)
        self.dataset_name = 'SD198'
        class_idx_path = os.path.join(data_dir, 'class_idx.npy')
        self.classes = self.get_classes_name(class_idx_path)
        self.classes = [class_name for class_name, _ in self.classes]

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path = self.data[index]
        target = self.targets[index]
        img = pil_loader(path)
        img = self.transform(img)

        return img, target

    def __len__(self):
        return len(self.data)

    def get_data(self, iter_no, data_dir):
        # iter_no=0
        if self.train:
            txt = '8_2_split/train_{}.txt'.format(iter_no)
        else:
            txt = '8_2_split/val_{}.txt'.format(iter_no)

        fn = os.path.join(data_dir, txt)
        txtfile = pd.read_csv(fn, sep=" ")
        raw_data = txtfile.values

        data = []
        targets = []
        for (path, label) in raw_data:
            data.append(os.path.join(self.data_dir, path))
            targets.append(label)

        return data, targets

    def get_classes_name(self, data_dir):
        classes_name = np.load(data_dir)
        return classes_name



def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


if __name__ == '__main__':
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()])
    cdataset = CheXpertDataset(train=True, transform=train_transform)
    loader = torch.utils.data.DataLoader(cdataset, batch_size=256, shuffle=True, num_workers=8)
    data, target = next(iter(loader))
    data = data.permute([1, 0, 2, 3])
    data = data.reshape(3, -1)
    print(data.mean(dim=1))
    print(data.std(dim=1))
