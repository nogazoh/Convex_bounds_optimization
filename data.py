import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms

import numpy as np
from sklearn.model_selection import train_test_split
import torch.utils.data


log_interval = 100
testing_frequency = 10
K = 50
learning_rate = 1e-3

batch_size = 128
test_batch_size = 32

data_H = 28
data_W = 28

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device = ", device)


# Prepare Data#

class MyDataset(Dataset):
    def __init__(self, samples, labels, transform=None):
        self.samples = torch.tensor(samples, dtype=torch.float32)
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        if len(sample.shape) == 4:
            sample = sample[0]

        if self.transform:
            trans = transforms.ToPILImage()
            sample = trans(sample)
            sample = self.transform(sample)

        if sample.shape[0] > 1:
            sample = transforms.functional.rgb_to_grayscale(sample, 1)

        return sample, self.labels[idx]


def generate_data_loader(X, Y, transform=None, shuffle=True, batch_size=batch_size):
    dataset = MyDataset(X[:, None, :, :], Y, transform)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    return loader


# scale pixels
def prep_pixels(images):
    # convert from integers to floats
    images_norm = images.astype('float32')

    # normalize to range 0-1
    images_norm = images_norm / 255.0

    # return normalized images
    return images_norm


# Load Data#

def get_dataset(data_name, trans):
    train_dataset, test_dataset = [], []
    if data_name == 'MNIST':
        train_dataset = datasets.MNIST('data_MNIST', train=True, download=True, transform=trans)
        test_dataset = datasets.MNIST('data_MNIST', train=False, transform=trans)
    elif data_name == 'SVHN':
        train_dataset = datasets.SVHN('data_SVHN', split='train', transform=trans, target_transform=None, download=True)
        test_dataset = datasets.SVHN('data_SVHN', split='test', transform=trans, target_transform=None, download=True)
    elif data_name == 'USPS':
        train_dataset = datasets.USPS('data_USPS', train=True, transform=trans, target_transform=None, download=True)
        test_dataset = datasets.USPS('data_USPS', train=False, transform=trans, target_transform=None, download=True)
    else:
        print("No such dataset name.")

    return train_dataset, test_dataset


def get_labels(data_name, dataset):
    labels = []
    if data_name == 'MNIST' or data_name == 'USPS':
        labels = dataset.targets
    elif data_name == 'SVHN':
        labels = dataset.labels
    else:
        print("No such dataset name.")

    return labels


def get_data_loaders(data_name, seed=None, num_datapoints=None):
    trans = transforms.Compose([transforms.Resize((data_H, data_W)), transforms.ToTensor()])

    train_dataset, test_dataset = get_dataset(data_name, trans)

    # Convert the pixels from 0-255 to 0-1.
    train_images = prep_pixels(np.array(train_dataset.data))
    test_images = prep_pixels(np.array(test_dataset.data))

    train_labels = np.array(get_labels(data_name, train_dataset))
    test_labels = np.array(get_labels(data_name, test_dataset))

    if num_datapoints is not None:
        np.random.seed(seed)

        train_images, _, train_labels, _ = \
            train_test_split(train_images, train_labels, train_size=num_datapoints, random_state=seed, stratify=train_labels)

        _, test_images, _, test_labels = \
            train_test_split(test_images, test_labels, test_size=num_datapoints, random_state=seed, stratify=test_labels)

    # Generate data loaders.
    train_loader = generate_data_loader(train_images, train_labels, trans, batch_size=batch_size)
    test_loader = generate_data_loader(test_images, test_labels, trans, batch_size=test_batch_size)
    return train_loader, test_loader

