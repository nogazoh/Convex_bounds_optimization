import torch
from torch.utils.data import Dataset, DataLoader, random_split, Subset
from torchvision import datasets, transforms
import numpy as np
from sklearn.model_selection import train_test_split
import os
from collections import defaultdict

# --- Global Configuration ---

# 1. Existing Office-Home Config (UNCHANGED)
OFFICE_HOME_PATH = './OfficeHome'
OFFICE_HOME_DOMAINS = ['Art', 'Clipart', 'Product', 'Real World']

# 2. NEW: Office-31 Config
OFFICE_31_PATH = './Office-31'
OFFICE_31_DOMAINS = ['amazon', 'dslr', 'webcam']

# Specific configurations for each data type
DOMAIN_CONFIGS = {
    'Digits': {
        'size': 28,
        'channels': 1,
        'flat_dim': 28 * 28 * 1
    },
    'OfficeHome': {
        'size': 128,
        'channels': 3,
        'flat_dim': 128 * 128 * 3
    },
    # NEW: Config for Office-31
    'Office31': {
        'size': 224,  # Standard size for Office-31 (usually used with ResNet)
        'channels': 3,
        'flat_dim': 224 * 224 * 3
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
    # Check Office-Home
    if data_name in OFFICE_HOME_DOMAINS:
        conf = DOMAIN_CONFIGS['OfficeHome'].copy()
        conf['flat_dim'] = conf['size'] * conf['size'] * conf['channels']
        return conf

    # Check Office-31 (NEW)
    elif data_name in OFFICE_31_DOMAINS:
        conf = DOMAIN_CONFIGS['Office31'].copy()
        conf['flat_dim'] = conf['size'] * conf['size'] * conf['channels']
        return conf

    # Check Digits
    elif data_name in ['MNIST', 'USPS', 'SVHN']:
        return DOMAIN_CONFIGS['Digits']
    else:
        raise ValueError(f"Unknown dataset name: {data_name}")


# --- NEW Helper Function for Office-31 Split ---

def get_office31_split(dataset, domain_name, seed=None):
    """
    Creates a train/test split following the specific protocol described in the thesis (Saenko et al.):
    - Amazon: 20 samples per class for training.
    - DSLR/Webcam: 7 samples per class for training.
    The rest go to testing.
    """
    # 1. Determine n_samples based on domain
    domain_lower = domain_name.lower()

    if 'amazon' in domain_lower:
        n_samples_per_class = 20
    elif 'dslr' in domain_lower or 'webcam' in domain_lower:
        n_samples_per_class = 7  # As per the thesis description
    else:
        # Fallback if name is unexpected, though shouldn't happen given the calling logic
        n_samples_per_class = 8

    print(f"--> [Office-31 Split] Domain: {domain_name} | Taking {n_samples_per_class} samples per class for Train.")

    # 2. Organize indices by class
    # ImageFolder datasets store targets in .targets
    targets = torch.tensor(dataset.targets)
    class_indices = defaultdict(list)

    for idx, label in enumerate(targets):
        class_indices[label.item()].append(idx)

    train_indices = []
    test_indices = []

    # Use a local RandomState to ensure reproducibility without affecting global state
    rng = np.random.RandomState(seed)

    # 3. Iterate over each class and sample strictly n_samples
    for label, indices in class_indices.items():
        indices = np.array(indices)

        # Shuffle indices for this class to pick random samples
        rng.shuffle(indices)

        # If a class has fewer samples than required (unlikely in Office31 but good for safety), take all
        curr_n = min(len(indices), n_samples_per_class)

        # Split
        train_indices.extend(indices[:curr_n])
        test_indices.extend(indices[curr_n:])

    # 4. Create Subsets
    train_subset = Subset(dataset, train_indices)
    test_subset = Subset(dataset, test_indices)

    print(f"--> [Office-31 Split] Result: Train={len(train_subset)}, Test={len(test_subset)}")

    return train_subset, test_subset


# --- Data Loading ---

def get_dataset(data_name, trans):
    """
    Loads the raw data.
    """

    # 1. Office-Home (UNCHANGED logic)
    if data_name in OFFICE_HOME_DOMAINS:
        full_path = os.path.join(OFFICE_HOME_PATH, data_name)
        if not os.path.exists(full_path):
            raise ValueError(f"Path not found: {full_path}. Check OFFICE_HOME_PATH.")
        dataset = datasets.ImageFolder(root=full_path, transform=trans)
        return dataset, None

    # 2. Office-31 (NEW logic)
    elif data_name in OFFICE_31_DOMAINS:
        full_path = os.path.join(OFFICE_31_PATH, data_name)
        if not os.path.exists(full_path):
            raise ValueError(f"Path not found: {full_path}. Check OFFICE_31_PATH.")

        dataset = datasets.ImageFolder(root=full_path, transform=trans)
        return dataset, None

    # 3. Digits
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
        print(f"No such dataset name: {data_name}")

    return train_dataset, test_dataset


def get_labels(data_name, dataset):
    # Helper function to extract labels
    if data_name in OFFICE_HOME_DOMAINS:
        return dataset.targets
    elif data_name in OFFICE_31_DOMAINS:  # Added Office-31 check
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

    # --- UPDATED TRANSFORMS ---
    # Basic list of transformations
    trans_steps = [
        transforms.Resize((target_size, target_size)),
        transforms.ToTensor(),
    ]

    # If it is Office-Home or Office-31 (RGB images for ResNet), normalization is mandatory!
    if config['channels'] == 3:
        trans_steps.append(transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                std=[0.229, 0.224, 0.225]))

    # If it is Digits (Grayscale)
    elif config['channels'] == 1:
        trans_steps.append(transforms.Grayscale(num_output_channels=1))
        # Optional: You can add simple normalization for grayscale if desired: transforms.Normalize((0.5,), (0.5,))

    trans = transforms.Compose(trans_steps)
    # ---------------------------

    # 3. Load initial data
    raw_train, raw_test = get_dataset(data_name, trans)

    # --- Split logic based on data type ---

    # Case A: Office-Home AND Office-31
    if (data_name in OFFICE_HOME_DOMAINS) or (data_name in OFFICE_31_DOMAINS):
        dataset = raw_train

        if data_name in OFFICE_31_DOMAINS:
            # Use specific split for Office-31
            train_subset, test_subset = get_office31_split(dataset, data_name, seed)
        else:
            # Standard random split for Office-Home
            total_len = len(dataset)
            train_len = int(0.8 * total_len)
            test_len = total_len - train_len
            generator = torch.Generator().manual_seed(seed) if seed else None
            train_subset, test_subset = random_split(dataset, [train_len, test_len], generator=generator)

        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_subset, batch_size=test_batch_size, shuffle=False)

        return train_loader, test_loader, config

    else:
        # Case B: Digits (Logic remains unchanged)
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