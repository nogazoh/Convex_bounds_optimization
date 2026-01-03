import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import datasets, transforms
import numpy as np
from sklearn.model_selection import train_test_split
import os

# --- Global Configuration ---

# Path to the OfficeHome directory
OFFICE_HOME_PATH = './OfficeHome'

# List of OfficeHome domains (Must match folder names exactly!)
# Note: 'Real World' has a space based on your directory structure.
OFFICE_HOME_DOMAINS = ['Art', 'Clipart', 'Product', 'Real World']

# Specific configurations for each data type
DOMAIN_CONFIGS = {
    'Digits': {
        'size': 28,
        'channels': 1,
        'flat_dim': 28 * 28 * 1
    },
    'OfficeHome': {
        'size': 128,  # Larger size to preserve details (can be changed to 64 or 224)
        'channels': 3,  # RGB
        'flat_dim': 128 * 128 * 3
    }
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device = ", device)


# --- Helper Classes ---

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

        # Note: Removed forced grayscale conversion (rgb_to_grayscale).
        # Channel control is now handled solely via the transform.

        return sample, self.labels[idx]


def generate_data_loader(X, Y, transform=None, shuffle=True, batch_size=128):
    dataset = MyDataset(X[:, None, :, :], Y, transform)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return loader


def prep_pixels(images):
    images_norm = images.astype('float32')
    images_norm = images_norm / 255.0
    return images_norm


def get_config(data_name):
    """Returns size and channel configuration based on the dataset name."""
    if data_name in OFFICE_HOME_DOMAINS:
        conf = DOMAIN_CONFIGS['OfficeHome'].copy()
        # Dynamic calculation in case size was changed manually above
        conf['flat_dim'] = conf['size'] * conf['size'] * conf['channels']
        return conf
    elif data_name in ['MNIST', 'USPS', 'SVHN']:
        return DOMAIN_CONFIGS['Digits']
    else:
        raise ValueError(f"Unknown dataset name: {data_name}")


# --- Data Loading ---

def get_dataset(data_name, trans):
    """
    Loads the raw data.
    For Digits - returns separate Train/Test datasets.
    For OfficeHome - returns a single Dataset (split will be done later).
    """

    # 1. Office-Home
    if data_name in OFFICE_HOME_DOMAINS:
        full_path = os.path.join(OFFICE_HOME_PATH, data_name)
        if not os.path.exists(full_path):
            raise ValueError(f"Path not found: {full_path}. Check OFFICE_HOME_PATH.")

        # Load from folder using ImageFolder
        dataset = datasets.ImageFolder(root=full_path, transform=trans)
        return dataset, None  # No built-in split in these folders

    # 2. Digits
    train_dataset, test_dataset = [], []
    if data_name == 'MNIST':
        train_dataset = datasets.MNIST('data_MNIST', train=True, download=True, transform=trans)
        test_dataset = datasets.MNIST('data_MNIST', train=False, transform=trans)
    elif data_name == 'SVHN':
        train_dataset = datasets.SVHN('data_SVHN', split='train', transform=trans, download=True)
        test_dataset = datasets.SVHN('data_SVHN', split='test', transform=trans, download=True)
    elif data_name == 'USPS':
        train_dataset = datasets.USPS('data_USPS', train=True, transform=trans, download=True)
        test_dataset = datasets.USPS('data_USPS', train=False, transform=trans, download=True)
    else:
        print("No such dataset name.")

    return train_dataset, test_dataset


def get_labels(data_name, dataset):
    # Helper function to extract labels (mainly for Digits)
    if data_name in OFFICE_HOME_DOMAINS:
        return dataset.targets
    elif data_name == 'MNIST' or data_name == 'USPS':
        return dataset.targets
    elif data_name == 'SVHN':
        return dataset.labels
    return []


def get_data_loaders(data_name, seed=None, num_datapoints=None, batch_size=128, test_batch_size=32):
    # 1. Retrieve configuration
    config = get_config(data_name)
    target_size = config['size']

    # 2. Build transforms
    trans_list = [
        transforms.Resize((target_size, target_size)),
        transforms.ToTensor()
    ]

    # If the dataset is defined as single-channel (like MNIST), add grayscale conversion
    if config['channels'] == 1:
        trans_list.append(transforms.Grayscale(num_output_channels=1))

    trans = transforms.Compose(trans_list)

    # 3. Load initial data
    raw_train, raw_test = get_dataset(data_name, trans)

    # --- Split logic based on data type ---

    # Case A: Office-Home (Heavy files, better to use standard DataLoader and avoid Numpy loading)
    if data_name in OFFICE_HOME_DOMAINS:
        dataset = raw_train  # Here we received a single dataset

        # Split to Train/Test (80/20)
        total_len = len(dataset)
        train_len = int(0.8 * total_len)
        test_len = total_len - train_len

        generator = torch.Generator().manual_seed(seed) if seed else None
        train_subset, test_subset = random_split(dataset, [train_len, test_len], generator=generator)

        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_subset, batch_size=test_batch_size, shuffle=False)

        return train_loader, test_loader, config

    # Case B: Digits (Original logic preserved as it works well for small files)
    else:
        # Convert pixels to floats 0-1
        train_images = prep_pixels(np.array(raw_train.data))
        test_images = prep_pixels(np.array(raw_test.data))

        train_labels = np.array(get_labels(data_name, raw_train))
        test_labels = np.array(get_labels(data_name, raw_test))

        if num_datapoints is not None:
            np.random.seed(seed)
            train_images, _, train_labels, _ = \
                train_test_split(train_images, train_labels, train_size=num_datapoints, random_state=seed,
                                 stratify=train_labels)
            _, test_images, _, test_labels = \
                train_test_split(test_images, test_labels, test_size=num_datapoints, random_state=seed,
                                 stratify=test_labels)

        train_loader = generate_data_loader(train_images, train_labels, trans, batch_size=batch_size)
        test_loader = generate_data_loader(test_images, test_labels, trans, batch_size=test_batch_size)

        return train_loader, test_loader, config