import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from e2cnn import gspaces
from e2cnn import nn as enn
from torch.utils.data import WeightedRandomSampler


# Map failure types to class indices
failure_type_to_idx = {
    'none': 0,
    'Center': 1,
    'Donut': 2,
    'Edge-Loc': 3,
    'Edge-Ring': 4,
    'Loc': 5,
    'Random': 6,
    'Scratch': 7,
    'Near-full': 8
}


class LSWMDDataset(Dataset):
    """
    Dataset class for loading LSWMD (Luggage Surface Wafer Map Dataset) data.
    
    Inherits from torch.utils.data.Dataset and loads data from LSWMD.pkl.
    Divides data into two sets based on whether failureType has labels or not:
    - Labeled: failureType is [['none']] (no failure) or [['failure_type']] (actual failure)
    - Unlabeled: failureType is [] (empty, no label provided)
    
    Args:
        data_path (str): Path to the LSWMD.pkl file. Default: 'data/LSWMD.pkl'
        labeled (bool): If True, return only labeled samples (non-empty failureType).
                       If False, return only unlabeled samples (empty failureType).
                       Default: True
    """
    
    def __init__(self, data_path='data/LSWMD.pkl', labeled=True, split='train'):
        """
        Initialize the dataset by loading pickle file and filtering by label status and split.
        Only keeps filtered samples in memory to save space.
        
        Args:
            data_path (str): Path to the LSWMD.pkl file
            labeled (bool): If True, return only labeled samples
            split (str): Either 'train' or 'test' to filter by trianTestLabel
        """
        full_data = pd.read_pickle(data_path)
        self.labeled = labeled
        self.split = split
        
        # Get indices of samples to keep, then filter and keep only those
        self.indices = self._get_filtered_indices(full_data)
        self.data = full_data.iloc[self.indices].reset_index(drop=True)
        
    def _get_filtered_indices(self, full_data):
        """
        Get indices of samples that match filter criteria.
        Returns indices for labeled/unlabeled samples and train/test split.

        If split is 'test' and n_samples is provided and labeled==True, try to sample
        test examples evenly across all labels to avoid a dominant 'none' class.
        """
        indices = []
        for idx, failure_type in enumerate(full_data['failureType']):
            # Check if failureType is empty (unlabeled)
            is_empty = self._is_empty_label(failure_type)

            # Check if sample matches the requested split
            row = full_data.iloc[idx]
            train_test_label = str(row['trianTestLabel']).lower()
            matches_split = (self.split == 'train' and "train" in train_test_label) or \
                            (self.split == 'test' and "test" in train_test_label)

            if self.labeled and not is_empty and matches_split:
                indices.append(idx)
            elif not self.labeled and is_empty and matches_split:
                indices.append(idx)

        return indices
    
    @staticmethod
    def _get_label_from_failure_type(failure_type):
        """
        Extract a string label from failureType field used in the dataframe.
        Returns 'none' if no label is present.
        """
        if isinstance(failure_type, np.ndarray):
            return failure_type[0][0] if failure_type.size > 0 else 'none'
        elif isinstance(failure_type, list):
            return failure_type[0][0] if len(failure_type) > 0 else 'none'
        else:
            return 'none'

    @staticmethod
    def _is_empty_label(failure_type):
        """
        Check if a failureType value is empty (unlabeled).
        Empty means no label was provided: []
        Labeled means label exists: [['none']] or [['failure_type']]
        """
        if isinstance(failure_type, np.ndarray):
            # Empty array [] has size 0
            return failure_type.size == 0
        elif isinstance(failure_type, list):
            # Empty list []
            return len(failure_type) == 0

        return False
    
    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.data)
    
    def __getitem__(self, idx):
        """
        Get a sample from the dataset.
        
        Args:
            idx (int): Index of the sample to retrieve
            
        Returns:
            dict: A dictionary containing:
                - 'waferMap': The wafer map (2D array)
                - 'dieSize': Die size value
                - 'lotName': Lot name
                - 'waferIndex': Wafer index
                - 'trianTestLabel': Train/test label
                - 'failureType': Failure type label
        """
        row = self.data.iloc[idx]
        
        # Extract failure type label
        failure_type = row['failureType']
        if isinstance(failure_type, np.ndarray):
            label = failure_type[0][0] if failure_type.size > 0 else 'none'
        elif isinstance(failure_type, list):
            label = failure_type[0][0] if len(failure_type) > 0 else 'none'
        else:
            label = 'none'
        
        return {
            'waferMap': np.array(row['waferMap']),
            'dieSize': float(row['dieSize']),
            'lotName': str(row['lotName']),
            'waferIndex': float(row['waferIndex']),
            'trianTestLabel': row['trianTestLabel'],
            'failureType': label
        }
    
    def get_labeled_dataset(self):
        """Return a new instance containing only labeled samples."""
        return LSWMDDataset(labeled=True)
    
    def get_unlabeled_dataset(self):
        """Return a new instance containing only unlabeled samples."""
        return LSWMDDataset(labeled=False)

class CNNBaseline(nn.Module):
    """
    Convolutional Neural Network for wafer map failure type classification.
    
    Architecture:
    - 3 convolutional blocks with ReLU activation and max pooling
    - 2 fully connected layers with dropout
    - Output layer for 9 failure type classes
    
    Args:
        params (dict): Dictionary containing model hyperparameters
    """
    
    def __init__(self, params):
        super().__init__()
        
        # Extract architecture parameters
        arch = params['architecture']
        conv1_cfg = arch['conv1']
        conv2_cfg = arch['conv2']
        conv3_cfg = arch['conv3']
        fc_cfg = arch['fc']
        dropout_rate = arch['dropout_rate']
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=conv1_cfg['out_channels'],
            kernel_size=conv1_cfg['kernel_size'],
            padding=conv1_cfg['padding']
        )
        self.conv2 = nn.Conv2d(
            in_channels=conv1_cfg['out_channels'],
            out_channels=conv2_cfg['out_channels'],
            kernel_size=conv2_cfg['kernel_size'],
            padding=conv2_cfg['padding']
        )
        self.conv3 = nn.Conv2d(
            in_channels=conv2_cfg['out_channels'],
            out_channels=conv3_cfg['out_channels'],
            kernel_size=conv3_cfg['kernel_size'],
            padding=conv3_cfg['padding']
        )
        
        # Max pooling
        self.pool = nn.MaxPool2d(kernel_size=arch['pool_kernel_size'], stride=arch['pool_stride'])
        
        # Dropout
        self.dropout = nn.Dropout(p=dropout_rate)
        
        # Adaptive average pooling to handle variable input sizes
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
        
        # Fully connected layers
        self.fc1 = nn.Linear(
            in_features=conv3_cfg['out_channels'] * 4 * 4,
            out_features=fc_cfg['fc1_hidden']
        )
        self.fc2 = nn.Linear(
            in_features=fc_cfg['fc1_hidden'],
            out_features=fc_cfg['fc2_hidden']
        )
        self.fc3 = nn.Linear(
            in_features=fc_cfg['fc2_hidden'],
            out_features=fc_cfg['num_classes']
        )
        
        # Activation functions
        self.relu = nn.ReLU()
    
    def forward(self, x):
        """
        Forward pass through the CNN.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 1, height, width)
        
        Returns:
            torch.Tensor: Output logits of shape (batch_size, 9)
        """
        # Conv block 1
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.dropout(x)
        
        # Conv block 2
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = self.dropout(x)
        
        # Conv block 3
        x = self.relu(self.conv3(x))
        x = self.pool(x)
        x = self.dropout(x)
        
        # Adaptive pooling for variable input sizes
        x = self.adaptive_pool(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        
        x = self.fc3(x)
        
        return x


def collate_fn(batch):
    """
    Custom collate function to handle variable-sized wafer maps.
    
    Args:
        batch (list): List of samples from the dataset
        
    Returns:
        tuple: (waferMaps, labels) where waferMaps is a batch tensor and labels are indices
    """
    wafer_maps = []
    labels = []



    for sample in batch:
        wafer_map = torch.tensor(sample['waferMap'], dtype=torch.float)
        # Add channel dimension if not present
        if wafer_map.dim() == 2:
            wafer_map = wafer_map.unsqueeze(0)
        wafer_maps.append(wafer_map)

        # Get label index
        failure_type = sample['failureType']
        label_idx = failure_type_to_idx.get(failure_type, 0)
        labels.append(label_idx)

    # Pad wafer maps to same size
    # Find max dimensions
    max_size = 32  # Set a maximum size for padding to avoid excessively large inputs
    max_h = max(w.shape[1] for w in wafer_maps)
    max_w = max(w.shape[2] for w in wafer_maps)
    max_h = min(max_h, max_size)  # Ensure minimum size for CNN
    max_w = min(max_w, max_size)

    padded_maps = []
    for wafer_map in wafer_maps:
        pad_h = max_h - wafer_map.shape[1]
        pad_w = max_w - wafer_map.shape[2]
        if pad_h == 0 and pad_w == 0:
            padded = wafer_map
        elif pad_h > 0 and pad_w > 0:
            padded = torch.nn.functional.pad(wafer_map, (0, pad_w, 0, pad_h), value=0)
        else:
            padded = torch.nn.functional.interpolate(wafer_map.unsqueeze(0),
                                                        size=(max_h, max_w),
                                                        mode='nearest').squeeze(0)
        padded_maps.append(padded)

    # Stack into batch
    wafer_batch = torch.stack(padded_maps)
    label_batch = torch.tensor(labels, dtype=torch.long)

    return wafer_batch, label_batch


class E2CNN(nn.Module):
    def __init__(self, params, N=4):

        super().__init__()
        arch = params["architecture"]
        conv1_cfg = arch["conv1"]
        conv2_cfg = arch["conv2"]
        conv3_cfg = arch["conv3"]
        fc_cfg = arch["fc"]
        dropout_rate = arch["dropout_rate"]
        # Rotation symmetry group
        self.r2_act = gspaces.Rot2dOnR2(N=N)
        # Input type
        self.input_type = enn.FieldType(self.r2_act, [self.r2_act.trivial_repr])
        # Feature types
        feat1_type = enn.FieldType(
            self.r2_act, conv1_cfg["out_channels"] * [self.r2_act.regular_repr]
        )
        feat2_type = enn.FieldType(
            self.r2_act, conv2_cfg["out_channels"] * [self.r2_act.regular_repr]
        )
        feat3_type = enn.FieldType(
            self.r2_act, conv3_cfg["out_channels"] * [self.r2_act.regular_repr]
        )
        # Block 1
        self.block1 = enn.SequentialModule(
            enn.R2Conv(
                self.input_type,
                feat1_type,
                kernel_size=conv1_cfg["kernel_size"],
                padding=conv1_cfg["padding"],
            ),
            enn.ReLU(feat1_type),
            enn.PointwiseMaxPool(
                feat1_type,
                kernel_size=arch["pool_kernel_size"],
                stride=arch["pool_stride"],
            ),
            enn.FieldDropout(feat1_type, p=dropout_rate),
        )
        # Block 2
        self.block2 = enn.SequentialModule(
            enn.R2Conv(
                feat1_type,
                feat2_type,
                kernel_size=conv2_cfg["kernel_size"],
                padding=conv2_cfg["padding"],
            ),
            enn.ReLU(feat2_type),
            enn.PointwiseMaxPool(
                feat2_type,
                kernel_size=arch["pool_kernel_size"],
                stride=arch["pool_stride"],
            ),
            enn.FieldDropout(feat2_type, p=dropout_rate),
        )
        # Block 3
        self.block3 = enn.SequentialModule(
            enn.R2Conv(
                feat2_type,
                feat3_type,
                kernel_size=conv3_cfg["kernel_size"],
                padding=conv3_cfg["padding"],
            ),
            enn.ReLU(feat3_type),
            enn.PointwiseMaxPool(
                feat3_type,
                kernel_size=arch["pool_kernel_size"],
                stride=arch["pool_stride"],
            ),
            enn.FieldDropout(feat3_type, p=dropout_rate),
        )
        # Rotation-invariant pooling
        self.gpool = enn.GroupPooling(feat3_type)
        invariant_channels = self.gpool.out_type.size
        # Ordinary PyTorch layers
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
        self.dropout = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(invariant_channels * 4 * 4, fc_cfg["fc1_hidden"])
        self.fc2 = nn.Linear(fc_cfg["fc1_hidden"], fc_cfg["fc2_hidden"])
        self.fc3 = nn.Linear(fc_cfg["fc2_hidden"], fc_cfg["num_classes"])
        self.relu = nn.ReLU()

    def forward(self, x):
        # Convert Tensor -> GeometricTensor
        x = enn.GeometricTensor(x, self.input_type)
        # Equivariant feature extraction
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        # Rotation invariant
        x = self.gpool(x)
        # Back to ordinary tensor
        x = x.tensor
        # Global features
        x = self.adaptive_pool(x)
        x = torch.flatten(x, start_dim=1)
        # Classifier
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        logits = self.fc3(x)
        return logits

class CostSensitiveLoss(nn.Module):
    def __init__(self, cost_matrix, device):
        super().__init__()
        self.cost_matrix = cost_matrix.to(device)
    
    def forward(self, logits, targets):
        probs = torch.softmax(logits, dim=1)
        target_costs = self.cost_matrix[targets]
        expected_cost = (probs * target_costs).sum(dim=1)
        return expected_cost.mean()

def init_weighted_random_sampler(dataset, num_classes):
    """
    Initialize a WeightedRandomSampler to handle class imbalance in the dataset.
    
    Args:
        dataset (LSWMDDataset): The dataset to sample from
        num_classes (int): Number of classes in the dataset
    
    Returns:
        WeightedRandomSampler: A sampler that can be used in DataLoader
    """
    # Count the number of samples per class
    class_counts = np.zeros(num_classes)
    for sample in dataset:
        label = failure_type_to_idx[sample['failureType']]
        class_counts[label] += 1

    # Calculate weights for each class (inverse of counts)
    class_weights = 1.0 / class_counts

    sample_weights = np.array([class_weights[failure_type_to_idx[sample["failureType"]]] for sample in dataset])

    # Create a WeightedRandomSampler
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
    
    return sampler


def get_cost_matrix():
    cost_matrix = torch.ones(9, 9)
    cost_matrix.fill_diagonal_(0)

    defect_classes = [1, 2, 3, 4, 5, 6, 7, 8]
    for c in defect_classes:
        cost_matrix[c, 0] = 2.0

    severe_classes = [4, 8]
    for c in severe_classes:
        cost_matrix[c, 0] = 3.0

    return cost_matrix

