import tables
import torch
import os
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset
from torchvision.datasets import DatasetFolder

def get_leaf_paths(hdf5_file, table_path):
    with tables.open_file(hdf5_file, mode='r') as file:
        group = file.get_node(table_path)
        leaf_paths = []

        if isinstance(group, tables.Leaf):
            leaf_paths.append(table_path)
        elif isinstance(group, tables.Group):
            for node in file.walk_nodes(group, classname='Leaf'):
                leaf_paths.append(node._v_pathname)
                
        return leaf_paths
    
class ImageDataset(DatasetFolder):
    def __init__(self, paths, label, transform=None):
        """
        Generic dataset for combining folders dynamically for any class.

        Args:
            paths (list): List of folder paths to include in the dataset.
                          The label is inferred from the last subfolder (e.g., '.../0', '.../1').
            label: the numeric label to use for all these folders
            transform (callable, optional): Transformation to apply to the input data.
        """
        self.transform = transform
        # Collect all files and their inferred labels
        self.samples = self._make_dataset(paths, label)

        if not self.samples:
            raise ValueError(f"No valid samples found in provided paths: {paths}")
        
    def _make_dataset(self, paths, label):
        samples = []
        for folder_path in paths:

            if not os.path.exists(folder_path):
                raise FileNotFoundError(f"Dataset path does not exist: {folder_path}")

            if not os.path.isdir(folder_path):
                raise NotADirectoryError(f"Expected a directory but got a file: {folder_path}")

            # Collect all valid files
            if os.path.isdir(folder_path):
                # We already know the label from `label` param
                for root, _, filenames in os.walk(folder_path):
                    for filename in filenames:
                        path = os.path.join(root, filename)
                        samples.append((path, label))
        return samples

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = Image.open(path)  # Adjust if not working with images
        if self.transform:
            sample = self.transform(sample)
        return sample, target

    def __len__(self):
        return len(self.samples)


class HDF5Dataset(Dataset):
    def __init__(self, hdf5_file, table_name, transform=None):
        self.hdf5_file = hdf5_file
        self.table_name = table_name
        self.transform = transform
        self.file = tables.open_file(hdf5_file, mode='r')
        self.table = self.file.get_node(table_name)

        # Retrieve min and max values from table attributes
        self.min_value = self.table.attrs.min_value if hasattr(self.table.attrs, 'min_value') else None
        self.max_value = self.table.attrs.max_value if hasattr(self.table.attrs, 'max_value') else None
        self.mean_value = self.table.attrs.mean_value if hasattr(self.table.attrs, 'mean_value') else None
        self.std_value = self.table.attrs.std_value if hasattr(self.table.attrs, 'std_value') else None

    def __len__(self):
        return self.table.nrows

    def __getitem__(self, idx):
        row = self.table[idx]  # Fetch the row dynamically
        spectrogram = row['data']
        label = row['label']
        
        # Convert to tensor
        spectrogram = torch.tensor(spectrogram, dtype=torch.float32).unsqueeze(0)

        if self.transform:
            spectrogram = self.transform(spectrogram)

        return spectrogram, torch.tensor(label, dtype=torch.long)

    def __del__(self):
        self.file.close()

    def set_transform(self, transform):
        self.transform = transform
    
class Subset(Dataset):
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]

class NormalizeToRange:
    def __init__(self, min_value=None, max_value=None, new_min=0, new_max=1):
        self.min_value = min_value
        self.max_value = max_value
        self.new_min = new_min
        self.new_max = new_max

    def __call__(self, tensor):
        # If min_value or max_value is not provided, use the tensor's min and max
        min_value = self.min_value if self.min_value is not None else tensor.min()
        max_value = self.max_value if self.max_value is not None else tensor.max()
        range_value = max_value - min_value
        
        # Scale tensor to [0, 1]
        tensor = (tensor - min_value) / range_value
        # Scale to [new_min, new_max]
        tensor = tensor * (self.new_max - self.new_min) + self.new_min
        return tensor

class Standardize:
    def __init__(self, mean=None, std=None):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        mean = self.mean if self.mean is not None else tensor.mean()
        std = self.std if self.std is not None else tensor.std()
        tensor = (tensor - mean) / std
        return tensor

class MedianNormalize:
    def __init__(self, axis=None):
        """
        Initialize the MedianNormalize class.

        Args:
            axis (int or None): The axis along which to compute the median.
                                - If axis=0, normalize by column medians.
                                - If axis=1, normalize by row medians.
                                - If axis=None, normalize globally by the median.
        """
        self.axis = axis

    def __call__(self, tensor):
        if self.axis is not None:
            median = torch.median(tensor, dim=self.axis, keepdim=True).values
        else:
            median = torch.median(tensor)
        
        tensor = tensor - median
        return tensor

class ConditionalResize:
    def __init__(self, target_size):
        self.target_size = target_size
        self.resize_transform = transforms.Resize(target_size)

    def __call__(self, image):
        if image.size != self.target_size:
            return self.resize_transform(image)
        return image