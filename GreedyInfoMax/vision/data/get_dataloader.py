import os
import string
import pickle

import torch
import torchvision.transforms as transforms
import torchvision

import lmdb
import numpy as np
import cv2
from torchvision.transforms import transforms


def get_dataloader(opt):
    if opt.dataset == "stl10":
        train_loader, train_dataset, supervised_loader, supervised_dataset, test_loader, test_dataset = get_stl10_dataloader(
            opt
        )
    else:
        raise Exception("Invalid option")

    return (
        train_loader,
        train_dataset,
        supervised_loader,
        supervised_dataset,
        test_loader,
        test_dataset,
    )

class Excrc:
    def __init__(self, path, split, transform=None):
        self.env = lmdb.open(path, map_size=1099511627776, readonly=True, lock=False)
        #with self.env.begin(write=False) as txn:
        #    self.image_names= [key.decode() for key in txn.cursor().iternext(keys=True, values=False)]

        cache_file = '_cache_' + ''.join(c for c in path if c in string.ascii_letters)
        cache_path = os.path.join(path, cache_file)
        if os.path.isfile(cache_path):
            self.image_names = pickle.load(open(cache_path, "rb"))
        else:
            with self.env.begin(write=False) as txn:
                self.image_names = [key for key in txn.cursor().iternext(keys=True, values=False)]
            pickle.dump(self.image_names, open(cache_path, "wb"))
        self.transform = transform

        indices = list(range(len(self.image_names)))
        import random
        random.Random(42).shuffle(indices)
        num_train = int(len(indices) * 0.8)
        train_indices = indices[:num_train]
        test_indices = indices[num_train:]


        if split == 'train':
            res = []
            for idx in train_indices:
                res.append(self.image_names[idx])
        elif split == 'test':
            res = []
            for idx in test_indices:
                res.append(self.image_names[idx])
        else:
            raise ValueError('wrong split')

        self.image_names = res

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):

        image_name = self.image_names[idx]
        with self.env.begin(write=False) as txn:
            image_data = txn.get(image_name)
            #image = np.frombuffer(image_data, np.uint8)
            #image = cv2.imdecode(image, -1)
            #image = Image.open(io.BytesIO(image_data))
            #image = pickle.loads(image_data)
            image = np.frombuffer(image_data, np.uint8)
            image = cv2.imdecode(image, -1)
            image = cv2.resize(image, (256, 256))
            #print(image.shape)

        if self.transform:
            image = self.transform(image)

        return image, 0

class Prostate:
    def __init__(self, path, split, transform=None):
        self.env = lmdb.open(path, map_size=1099511627776, readonly=True, lock=False)
        #with self.env.begin(write=False) as txn:
        #    self.image_names= [key.decode() for key in txn.cursor().iternext(keys=True, values=False)]

        cache_file = '_cache_' + ''.join(c for c in path if c in string.ascii_letters)
        cache_path = os.path.join(path, cache_file)
        if os.path.isfile(cache_path):
            self.image_names = pickle.load(open(cache_path, "rb"))
        else:
            with self.env.begin(write=False) as txn:
                self.image_names = [key for key in txn.cursor().iternext(keys=True, values=False)]
            pickle.dump(self.image_names, open(cache_path, "wb"))
        self.transform = transform

        indices = list(range(len(self.image_names)))
        import random
        random.Random(42).shuffle(indices)
        num_train = int(len(indices) * 0.8)
        train_indices = indices[:num_train]
        test_indices = indices[num_train:]


        if split == 'train':
            res = []
            for idx in train_indices:
                res.append(self.image_names[idx])
        elif split == 'test':
            res = []
            for idx in test_indices:
                res.append(self.image_names[idx])
        else:
            raise ValueError('wrong split')

        self.image_names = res




    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):

        image_name = self.image_names[idx]
        with self.env.begin(write=False) as txn:
            image_data = txn.get(image_name)
            #image = np.frombuffer(image_data, np.uint8)
            #image = cv2.imdecode(image, -1)
            #image = Image.open(io.BytesIO(image_data))
            #image = pickle.loads(image_data)
            image = np.frombuffer(image_data, np.uint8)
            image = cv2.imdecode(image, -1)

        if self.transform:
            image = self.transform(image)

        return image, 0

def get_stl10_dataloader(opt):
    base_folder = os.path.join(opt.data_input_dir, "stl10_binary")

    print(opt.grayscale)
    aug = {
        "stl10": {
            #"randcrop": 64,
            "randcrop": 256,
            "flip": True,
            "grayscale": opt.grayscale,
            "mean": [0.4313, 0.4156, 0.3663],  # values for train+unsupervised combined
            "std": [0.2683, 0.2610, 0.2687],
            "bw_mean": [0.4120],  # values for train+unsupervised combined
            "bw_std": [0.2570],
        }  # values for labeled train set: mean [0.4469, 0.4400, 0.4069], std [0.2603, 0.2566, 0.2713]
    }
    transform_train = transforms.Compose(
        [get_transforms(eval=False, aug=aug["stl10"])]
    )
    transform_valid = transforms.Compose(
        [get_transforms(eval=True, aug=aug["stl10"])]
    )

    #train_dataset = Prostate('/data/hdd1/by/Greedy_InfoMax/datasets/tcga_lmdb', 'train', transform=transform_train)
    #unsupervised_dataset = Prostate('/data/hdd1/by/Greedy_InfoMax/datasets/tcga_lmdb', 'train', transform=transform_train)
    #test_dataset = Prostate('/data/hdd1/by/Greedy_InfoMax/datasets/tcga_lmdb', 'test', transform=transform_valid)

    train_dataset = Excrc('/data/hdd1/by/Greedy_InfoMax/datasets/excrc_lmdb/', 'train', transform=transform_train)
    unsupervised_dataset = Excrc('/data/hdd1/by/Greedy_InfoMax/datasets/excrc_lmdb/', 'train', transform=transform_train)
    test_dataset = Excrc('/data/hdd1/by/Greedy_InfoMax/datasets/excrc_lmdb/', 'test', transform=transform_valid)
    #indices = list(range(train_dataset))

    #unsupervised_dataset = torchvision.datasets.STL10(
    #    base_folder,
    #    split="unlabeled",
    #    transform=transform_train,
    #    download=opt.download_dataset,
    #) #set download to True to get the dataset

    #train_dataset = torchvision.datasets.STL10(
    #    base_folder, split="train", transform=transform_train, download=opt.download_dataset
    #)

    #test_dataset = torchvision.datasets.STL10(
    #    base_folder, split="test", transform=transform_valid, download=opt.download_dataset
    #)

    # default dataset loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=opt.batch_size_multiGPU, shuffle=True, num_workers=4
    )

    unsupervised_loader = torch.utils.data.DataLoader(
        unsupervised_dataset,
        batch_size=opt.batch_size_multiGPU,
        shuffle=True,
        num_workers=4,
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=opt.batch_size_multiGPU, shuffle=False, num_workers=4
    )

    # create train/val split
    #if opt.validate:
        #print("Use train / val split")

        #if opt.training_dataset == "train":
        #    dataset_size = len(train_dataset)
        #    train_sampler, valid_sampler = create_validation_sampler(dataset_size)

        #    train_loader = torch.utils.data.DataLoader(
        #        train_dataset,
        #        batch_size=opt.batch_size_multiGPU,
        #        sampler=train_sampler,
        #        num_workers=4,
        #    )

        #elif opt.training_dataset == "unlabeled":
        #    print('here')
        #    dataset_size = len(unsupervised_dataset)
        #    train_sampler, valid_sampler = create_validation_sampler(dataset_size)

        #    unsupervised_loader = torch.utils.data.DataLoader(
        #        unsupervised_dataset,
        #        batch_size=opt.batch_size_multiGPU,
        #        sampler=train_sampler,
        #        num_workers=4,
        #    )

        #else:
        #    raise Exception("Invalid option")

        # overwrite test_dataset and _loader with validation set
        #test_dataset = torchvision.datasets.STL10(
        #    base_folder,
        #    split=opt.training_dataset,
        #    transform=transform_valid,
        #    download=opt.download_dataset,
        #)

        #test_loader = torch.utils.data.DataLoader(
        #    test_dataset,
        #    batch_size=opt.batch_size_multiGPU,
        #    sampler=valid_sampler,
        #    num_workers=4,
        #)

    #else:
        #print("Use (train+val) / test split")

    print(len(unsupervised_loader), len(train_loader), len(test_loader))
    return (
        unsupervised_loader,
        unsupervised_dataset,
        train_loader,
        train_dataset,
        test_loader,
        test_dataset,
    )


def create_validation_sampler(dataset_size):
    # Creating data indices for training and validation splits:
    validation_split = 0.2
    shuffle_dataset = True

    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))
    if shuffle_dataset:
        np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

    # Creating data samplers and loaders:
    train_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_indices)
    valid_sampler = torch.utils.data.sampler.SubsetRandomSampler(val_indices)

    return train_sampler, valid_sampler


def get_transforms(eval=False, aug=None):
    trans = []

    trans.append(transforms.ToPILImage())
    #if aug["randcrop"] and not eval:
    #    trans.append(transforms.RandomCrop(aug["randcrop"]))

    #if aug["randcrop"] and eval:
    #    trans.append(transforms.CenterCrop(aug["randcrop"]))

    if aug["flip"] and not eval:
        trans.append(transforms.RandomHorizontalFlip())

    if aug["grayscale"]:
        trans.append(transforms.Grayscale())
        trans.append(transforms.ToTensor())
        trans.append(transforms.Normalize(mean=aug["bw_mean"], std=aug["bw_std"]))
    elif aug["mean"]:
        trans.append(transforms.ToTensor())
        trans.append(transforms.Normalize(mean=aug["mean"], std=aug["std"]))
    else:
        trans.append(transforms.ToTensor())

    trans = transforms.Compose(trans)
    return trans