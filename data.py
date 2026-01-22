import torch
from torch.utils.data import Dataset, DataLoader, random_split, Subset
from torchvision import datasets, transforms
import numpy as np
from sklearn.model_selection import train_test_split
import os
from collections import defaultdict

# --- Global Configuration ---

OFFICE_HOME_PATH = './OfficeHome'
OFFICE_HOME_DOMAINS = ['Art', 'Clipart', 'Product', 'Real World']

OFFICE_31_PATH = './Office-31'
OFFICE_31_DOMAINS = ['amazon', 'dslr', 'webcam']

DOMAIN_CONFIGS = {
    'Digits': {
        'size': 28,
        'channels': 1,
        'flat_dim': 28 * 28 * 1
    },
    'OfficeHome': {
        'size': 224,
        'channels': 3,
        'flat_dim': 224 * 224 * 3
    },
    'Office31': {
        'size': 224,
        'channels': 3,
        'flat_dim': 224 * 224 * 3
    }
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --- Helper Classes ---

class ApplyTransform(Dataset):
    """
    A utility wrapper to apply specific transformations to a dataset subset.
    This allows us to have different transforms for Train and Test even after a random_split.
    """

    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.subset)


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

        return sample, self.labels[idx]


# --- Processing Helpers ---

def generate_data_loader(X, Y, transform=None, shuffle=True, batch_size=128):
    dataset = MyDataset(X[:, None, :, :], Y, transform)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return loader


def prep_pixels(images):
    images_norm = images.astype('float32')
    images_norm = images_norm / 255.0
    return images_norm


def get_config(data_name):
    if data_name in OFFICE_HOME_DOMAINS:
        conf = DOMAIN_CONFIGS['OfficeHome'].copy()
        conf['flat_dim'] = conf['size'] * conf['size'] * conf['channels']
        return conf
    elif data_name in OFFICE_31_DOMAINS:
        conf = DOMAIN_CONFIGS['Office31'].copy()
        conf['flat_dim'] = conf['size'] * conf['size'] * conf['channels']
        return conf
    elif data_name in ['MNIST', 'USPS', 'SVHN']:
        return DOMAIN_CONFIGS['Digits']
    else:
        raise ValueError(f"Unknown dataset name: {data_name}")


# --- Office-31 Specific Split ---

def get_office31_split(dataset, domain_name, seed=None):
    domain_lower = domain_name.lower()
    if 'amazon' in domain_lower:
        n_samples_per_class = 20
    elif 'dslr' in domain_lower or 'webcam' in domain_lower:
        n_samples_per_class = 7
    else:
        n_samples_per_class = 8

    print(f"--> [Office-31 Split] Domain: {domain_name} | Train Samples/Class: {n_samples_per_class}")

    targets = torch.tensor(dataset.targets)
    class_indices = defaultdict(list)
    for idx, label in enumerate(targets):
        class_indices[label.item()].append(idx)

    train_indices, test_indices = [], []
    rng = np.random.RandomState(seed)

    for label, indices in class_indices.items():
        indices = np.array(indices)
        rng.shuffle(indices)
        curr_n = min(len(indices), n_samples_per_class)
        train_indices.extend(indices[:curr_n])
        test_indices.extend(indices[curr_n:])

    return Subset(dataset, train_indices), Subset(dataset, test_indices)


# --- Data Loading Logic ---

def get_dataset(data_name, trans):
    if data_name in OFFICE_HOME_DOMAINS:
        full_path = os.path.join(OFFICE_HOME_PATH, data_name)
        dataset = datasets.ImageFolder(root=full_path, transform=trans)
        return dataset, None
    elif data_name in OFFICE_31_DOMAINS:
        full_path = os.path.join(OFFICE_31_PATH, data_name)
        dataset = datasets.ImageFolder(root=full_path, transform=trans)
        return dataset, None
    elif data_name == 'MNIST':
        return datasets.MNIST('data_MNIST', train=True, download=True, transform=trans), \
            datasets.MNIST('data_MNIST', train=False, transform=trans)
    elif data_name == 'SVHN':
        return datasets.SVHN('data_SVHN', split='train', transform=trans, download=True), \
            datasets.SVHN('data_SVHN', split='test', transform=trans, download=True)
    elif data_name == 'USPS':
        return datasets.USPS('data_USPS', train=True, transform=trans, download=True), \
            datasets.USPS('data_USPS', train=False, transform=trans, download=True)
    return None, None


def get_labels(data_name, dataset):
    if data_name in OFFICE_HOME_DOMAINS or data_name in OFFICE_31_DOMAINS or data_name in ['MNIST', 'USPS']:
        return dataset.targets
    elif data_name == 'SVHN':
        return dataset.labels
    return []


def get_data_loaders(data_name, seed=None, num_datapoints=None, batch_size=128, test_batch_size=32):
    config = get_config(data_name)
    target_size = config['size']

    # --- 1. Define Transformation Pipelines ---
    # Training: Resize + Augmentation + Normalize
    train_trans = transforms.Compose([
        transforms.Resize((target_size, target_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Testing: Resize + Normalize (No Augmentation)
    test_trans = transforms.Compose([
        transforms.Resize((target_size, target_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # --- 2. Load Data ---
    # Load dataset without fixed transform first (passed as None) to handle splits manually
    raw_train, raw_test = get_dataset(data_name, None)

    # Case A: Office Datasets (RGB with Augmentation)
    if (data_name in OFFICE_HOME_DOMAINS) or (data_name in OFFICE_31_DOMAINS):
        dataset = raw_train
        if data_name in OFFICE_31_DOMAINS:
            train_subset, test_subset = get_office31_split(dataset, data_name, seed)
        else:
            total_len = len(dataset)
            train_len = int(0.8 * total_len)
            generator = torch.Generator().manual_seed(seed) if seed else None
            train_subset, test_subset = random_split(dataset, [train_len, total_len - train_len], generator=generator)

        # Apply different transforms to Train and Test loaders
        train_loader = DataLoader(ApplyTransform(train_subset, train_trans), batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(ApplyTransform(test_subset, test_trans), batch_size=test_batch_size, shuffle=False)
        return train_loader, test_loader, config

    # Case B: Digits (Grayscale, No Augmentation as requested)
    else:
        # Define digit-specific grayscale transform
        digit_trans = transforms.Compose([
            transforms.Resize((target_size, target_size)),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor()
        ])

        train_images = prep_pixels(np.array(raw_train.data))
        test_images = prep_pixels(np.array(raw_test.data))
        train_labels = np.array(get_labels(data_name, raw_train))
        test_labels = np.array(get_labels(data_name, raw_test))

        if num_datapoints is not None:
            train_images, _, train_labels, _ = train_test_split(
                train_images, train_labels, train_size=num_datapoints, random_state=seed, stratify=train_labels
            )
            _, test_images, _, test_labels = train_test_split(
                test_images, test_labels, test_size=num_datapoints, random_state=seed, stratify=test_labels
            )

        return generate_data_loader(train_images, train_labels, digit_trans, batch_size=batch_size), \
            generate_data_loader(test_images, test_labels, digit_trans, batch_size=test_batch_size), \
            config