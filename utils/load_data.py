import os

import torch
import numpy as np
from torch.utils.data import Dataset
from pandas import read_table
from sklearn.preprocessing import OneHotEncoder


def list_dir_of_path(path):
    return [f for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))]


def get_dataset_file_path(datasetpath):
    """
    return a dict of dataset with the table of content\n
    the table of content sturcture is like:
    `dataset--class_i--subclass_i--I--file_name_i`

    params:
    -----------------------------------
    datasetpath: the path of dataset

    return:
    -----------------------------------
    dataset_path_dic: a dict of subclass file path

    example:
    -----------------------------------
    >>> dataset_path = "./data/OCVS/"
    >>> dataset = get_dataset_file_path(dataset_path)
    >>> dataset
    {
        'A': ['./data/OCVS/CEP\\A\\I\\OGLE-LMC-ACEP-001.dat',...],
        'T110': ['./data/OCVS/CEP\\T110\\I\\OGLE-LMC-T2CEP-001.dat',...],
        'T2': ['./data/OCVS/CEP\\T2\\I\\OGLE-LMC-T2CEP-001.dat',...],
        ...
    }

    """
    dataset_path_dic = {}
    class_dir = list_dir_of_path(datasetpath)
    for class_name in class_dir:
        class_dir_path = os.path.join(datasetpath, class_name)
        sub_class_name = list_dir_of_path(class_dir_path)
        for sub_class in sub_class_name:
            subdir_path = os.path.join(class_dir_path, sub_class)
            I_data_path = os.path.join(subdir_path, "I")
            I_file_list = os.listdir(I_data_path)
            I_file_list_path = [os.path.join(I_data_path, f) for f in I_file_list]
            dataset_path_dic[sub_class] = I_file_list_path
    return dataset_path_dic


def fattening_dataset(dataset_path_dic):
    """
    fattening dataset
    return a list of labels and a list of dataset
    """
    labels = []
    dataset = []

    for key, value in dataset_path_dic.items():
        for i in range(len(value)):
            labels.append(key)
            dataset.append(value[i])

    return labels, dataset


def min_length_of_dataset(dataset_path):
    """
    return the min length of all light curve data,
    and the index of light curve which length is less than 400
    """
    uncepted_length_index = []
    min_length = 100000000
    limit_length = 400
    for i, path in enumerate(dataset_path):
        light_curve = read_table(path, sep="\\s+", names=["time", "mag", "err"])
        light_curve = light_curve["mag"].values
        length = len(light_curve)
        if length < limit_length:
            uncepted_length_index.append(i)
            continue

        if length < min_length:
            min_length = length

    return min_length, uncepted_length_index


def get_mag_data_from_path(path, min_length):
    light_curve = read_table(path, sep="\\s+", names=["time", "mag", "err"])
    light_curve = light_curve["mag"].values
    light_curve = light_curve[:min_length]
    # transform to double
    light_curve = light_curve.astype(np.double)
    return light_curve


class LightcurveDataset(Dataset):
    def __init__(self, dataset_path, transform=True, target_transform=False):
        super().__init__()
        self.data_path_dic = get_dataset_file_path(dataset_path)
        self.labels, self.dataset = fattening_dataset(self.data_path_dic)
        self.min_length, uncepted_index = min_length_of_dataset(self.dataset)
        self.labels = np.delete(self.labels, uncepted_index)
        self.dataset = np.delete(self.dataset, uncepted_index)
        self.transform = transform
        self.target_transform = target_transform
        self.label_encoder = OneHotEncoder()
        self.label_encoder.fit(self.labels.reshape(-1, 1))

    def __len__(self):
        assert len(self.labels) == len(self.dataset)
        return len(self.labels)

    def __getitem__(self, index):
        light_curve = get_mag_data_from_path(self.dataset[index], self.min_length)
        label = self.labels[index]
        if self.transform:
            light_curve = torch.as_tensor(light_curve, dtype=torch.float32)
        if self.target_transform:
            label = self.label_encoder.transform(label.reshape(-1, 1)).toarray()
            label = torch.as_tensor(label, dtype=torch.float32).reshape(-1)
        return light_curve, label


class WrappedDataLoader:
    def __init__(self, dl, func):
        self.dl = dl
        self.func = func

    def __len__(self):
        return len(self.dl)

    def __iter__(self):
        for b in self.dl:
            yield (self.func(*b))

    def __next__(self):
        return self.__iter__()