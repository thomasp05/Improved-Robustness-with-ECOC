"""
This is where different datasets are downloaded
Supported datasets: MNIST, Fashion-MNIST, CIFAR10
"""
import torch
from torchvision import transforms
from torchvision.datasets import MNIST
from torchvision.datasets import CIFAR10
from torchvision.datasets import SVHN
from torchvision.datasets import FashionMNIST
from torch.utils.data import DataLoader
from torch.utils.data import Subset
from torch.utils.data import TensorDataset
import random

def getDataset(name): 
    '''
    Return dataset for a given dataset name

            Parameters:
                    name (string): name of the dataset ("MNIST", "Fashion-MNIST" or "CIFAR10") 

            Returns:
                    train_dataset, test_dataset: training and testing pytorch dataset objects 
    '''

    if name == "MNIST": 
        transform = transforms.Compose([
                                    transforms.ToTensor(),
                                ])
        train_data = MNIST("datasets/mnist", train=True, download=True, transform=transform)
        test_data = MNIST("datasets/mnist", train=False, download=True, transform=transforms.ToTensor())


    elif name == "CIFAR10":
        transform = transforms.Compose([
                                    transforms.RandomCrop(32, padding=2), 
                                    transforms.RandomHorizontalFlip(), 
                                    transforms.ToTensor(),
                                ])     
        train_data = CIFAR10("datasets/cifar10", train=True, download=True, transform=transform) 
        test_data = CIFAR10("datasets/cifar10", train=False, download=True, transform=transforms.ToTensor())

    elif name == "Fashion-MNIST": 
        transform = transforms.Compose([
                                    transforms.RandomHorizontalFlip(),  
                                    transforms.ToTensor(),
                                ])
        train_data = FashionMNIST("datasets/Fashion-MNIST", train=True, download=True, transform=transform)
        test_data = FashionMNIST("datasets/Fashion-MNIST", train=False, download=True, transform=transforms.ToTensor()) 

    else: 
        print("The dataset name provided is invalid")
        return 0

    return train_data, test_data
    


def get_dataLoader(dataset, batch_size, shuffle=True): 
    '''
    Return dataloader for a given dataset object 

            Parameters:
                    dataset (PyTorch dataset): dataset PyTorch object 
                    batch_size (int) : Batch size for the dataLoader 

            Returns:
                    dataloader: PyTorch dataloader corresponding to the dataset provided
    '''
    dataLoader = DataLoader(dataset, shuffle=shuffle, batch_size=batch_size)
    return dataLoader

def get_test_subset(dataset, batch_size, nb_images, seed=None): 
    '''
    Return dataloader containing a subset of the test data (useful for generating adversarial examples on a subset of the test data)

            Parameters:
                    dataset (PyTorch dataset): dataset PyTorch object (ca peu etre classifier.test_dataset)
                    batch_size (int) : Batch size for the dataLoader 

            Returns:
                    dataloader: PyTorch dataloader corresponding to a subset the dataset provided
    '''

    if seed != None: 
        random.seed(seed)

    complete_list = list(range(0, len(dataset), 1))
    random.shuffle(complete_list) 
    list_1 = complete_list[0:nb_images]
    test_data1 = Subset(dataset, list_1) 
    testloader1 = DataLoader(test_data1, batch_size=batch_size, shuffle=False)

    return testloader1

def get_test_class(dataset, batch_size, n,  seed=None): 
    '''
    Return dataloader containing all images of the class n from the test dataset

            Parameters:
                    dataset (PyTorch dataset): dataset PyTorch object (ca peu etre classifier.test_dataset)
                    batch_size (int) : Batch size for the dataLoader 
                    n (int) : index of the class

            Returns:
                    dataloader: PyTorch dataloader corresponding to a subset the dataset provided
    '''
    if seed != None: 
        random.seed(seed)

    temp = []
    for data in dataset: 
        if data[1] == n: 
            temp.append(data)

    testloader1 = DataLoader(temp, batch_size=batch_size, shuffle=False)
    return testloader1


def train_valid_split(dataset, batch_size, train_split, shuffle=True, seed=None): 
    '''
    separate the dataset into training and validation sets and returns the two dataloaders. 

            Parameters:
                    dataset (PyTorch dataset object): dataset to be separated into train and validation partitions 
                    batch_size (int) : Batch size for the dataLoader 
                    train_split (double) : percentage of the dataset for the training split. range = ]0, 1] 
                    shuffle (bool) : shuffle the data when creating the dataloader (default=True)  
                    seed (int) : if you want to fix the seed (default=None)
            Returns:
                    train_dataLoader, validation_dataLoader 
    '''

    random.seed(seed)

    dataset_len = len(dataset) 
    nb_training_examples = round(train_split * dataset_len)

    # create list of indices and shuffle it randomly 
    index_list = list(range(dataset_len))
    random.shuffle(index_list)

    # create training and validation dataLoaders 
    train_index = index_list[:nb_training_examples] 
    validation_index = index_list[nb_training_examples:]

    # sample examples from dataset at corresponding indices form train_index and validation_index
    train_dataset = Subset(dataset, train_index) 
    validation_dataset = Subset(dataset, validation_index) 
    
    # create training and validation dataLoaders
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
    if len(validation_index) == 0: 
        validation_dataloader = None 
    else: 
        validation_dataloader = DataLoader(validation_dataset, shuffle=True, batch_size=batch_size)
  
    return train_dataloader, validation_dataloader

