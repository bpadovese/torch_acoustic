import tables
import torch
from torch.utils.data import Dataset

class HDF5Dataset(Dataset):
    def __init__(self, hdf5_file, table_name, transform=None):
        self.hdf5_file = hdf5_file
        self.table_name = table_name
        self.transform = transform
        self.file = tables.open_file(hdf5_file, mode='r')
        self.table = self.file.get_node(f'{table_name}/data')
        self.data = self.table.col('data')
        self.labels = self.table.col('label')

        # Retrieve min and max values from table attributes
        self.min_value = self.table.attrs.min_value
        self.max_value = self.table.attrs.max_value
        self.mean_value = self.table.attrs.mean_value
        self.std_value = self.table.attrs.std_value

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        spectrogram = self.data[idx]
        label = self.labels[idx]
        
        # Convert to tensor
        spectrogram = torch.tensor(spectrogram, dtype=torch.float32).unsqueeze(0)

        if self.transform:
            spectrogram = self.transform(spectrogram)

        return spectrogram, torch.tensor(label, dtype=torch.long)

    def __del__(self):
        self.file.close()

    def set_transform(self, transform):
        self.transform = transform

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