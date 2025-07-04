import time
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import re
import random
import numpy as np
import os
import itertools
from math import floor
from dev_utils.nn import resnet18_for_single_channel, resnet50_for_single_channel, AdvPropResNet
from data_handling.dataset import HDF5Dataset, NormalizeToRange, ImageDataset, MedianNormalize, ConditionalResize, get_leaf_paths
from torch.utils.data import DataLoader, ConcatDataset, SubsetRandomSampler
from pathlib import Path
from collections import defaultdict
from diffusers.optimization import get_cosine_schedule_with_warmup 
from lightning.fabric import Fabric
from lightning.fabric.loggers import CSVLogger
import torcheval.metrics as metrics


def train(fabric, model, optimizer, loss_func, train_loader_real_0, train_loader_aug_0, train_loader_real_1, train_loader_aug_1, val_loader=None, lr_scheduler=None, num_epochs=20, num_classes=2):
    print("Training starting...")
    for epoch in range(num_epochs):
        start_time = time.time()  # Start timer
        model.train()

        # Training step
        train_loss, train_accuracy, train_precision, train_recall = train_epoch(
            fabric, model, optimizer, loss_func, train_loader_real_0, train_loader_aug_0, train_loader_real_1, train_loader_aug_1, lr_scheduler=lr_scheduler, num_classes=num_classes
        )

        # Validation step
        avg_val_loss, val_accuracy, val_precision, val_recall = validate(
            fabric, model, val_loader, loss_func, num_classes
        )

        end_time = time.time()  # End timer
        epoch_duration = end_time - start_time

        print(f'Epoch {epoch + 1}, Training Loss: {train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}')
        print(f'Training Accuracy: {train_accuracy:.2f}%, Training Precision: {train_precision:.4f}, Training Recall: {train_recall:.4f}')
        print(f'Validation Accuracy: {val_accuracy:.2f}%, Validation Precision: {val_precision:.4f}, Validation Recall: {val_recall:.4f}')
        print(f'Epoch Duration: {epoch_duration:.2f} seconds')
        print()

        # Log metrics
        fabric.log_dict({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_accuracy': train_accuracy,
            'train_precision': train_precision,
            'train_recall': train_recall,
            'validation_loss': avg_val_loss,
            'validation_accuracy': val_accuracy,
            'validation_precision': val_precision,
            'validation_recall': val_recall,
        })

def train_epoch(fabric, model, optimizer, loss_func, train_loader_real_0, train_loader_aug_0, train_loader_real_1, train_loader_aug_1, lr_scheduler=None, num_classes=2):
    running_loss = 0.0

    # Initialize metrics
    precision_metric = metrics.MulticlassPrecision(num_classes=num_classes, average=None).to(fabric.device)
    recall_metric = metrics.MulticlassRecall(num_classes=num_classes, average=None).to(fabric.device)
    accuracy_metric = metrics.MulticlassAccuracy(num_classes=num_classes, average='macro').to(fabric.device)
    
    # Create iterators to cycle through the loaders
    iterator_real_0 = itertools.cycle(train_loader_real_0)
    iterator_aug_0 = itertools.cycle(train_loader_aug_0)
    iterator_real_1 = itertools.cycle(train_loader_real_1)
    iterator_aug_1 = itertools.cycle(train_loader_aug_1)
    
    for _ in range(len(train_loader_real_0)):  # Loop over class 0 batches
    # for clean_inputs, clean_labels in train_loader:
        optimizer.zero_grad()

        # Get proportionally sampled batches
        inputs_real_0, labels_real_0 = next(iterator_real_0)
        inputs_aug_0, labels_aug_0 = next(iterator_aug_0)
        inputs_real_1, labels_real_1 = next(iterator_real_1)
        inputs_aug_1, labels_aug_1 = next(iterator_aug_1)

        # Combine real samples
        inputs_class_0 = torch.cat([inputs_real_0, inputs_aug_0], dim=0)
        labels_class_0 = torch.cat([labels_real_0, labels_aug_0], dim=0)

        inputs_class_1 = torch.cat([inputs_real_1, inputs_aug_1], dim=0)
        labels_class_1 = torch.cat([labels_real_1, labels_aug_1], dim=0)

        # Combine real class 0 and class 1
        inputs_clean = torch.cat([inputs_real_0, inputs_real_1], dim=0)
        labels_clean = torch.cat([labels_real_0, labels_real_1], dim=0)

        # Combine augmented class 0 and class 1 for adversarial BN
        inputs_adv = torch.cat([inputs_aug_0, inputs_aug_1], dim=0)
        labels_adv = torch.cat([labels_aug_0, labels_aug_1], dim=0)
        # unique_classes = torch.unique(labels, return_counts=True)
        # print(f"Batch classes: {unique_classes}")

        # Step 1: Forward pass for clean data (Main BN)
        outputs_clean = model(inputs_clean, is_adv=False)
        loss_clean = loss_func(outputs_clean, labels_clean)

        # Step 2: Forward pass for adversarial data (Auxiliary BN)
        outputs_adv = model(inputs_adv, is_adv=True)  # Adversarial BN for both classes
        loss_adv = loss_func(outputs_adv, labels_adv)

        # Step 3: Compute the total loss (average of both losses)
        loss = loss_clean + loss_adv
        # alpha = 0.5  # Tune this value
        # loss = (1 - alpha) * loss_clean + alpha * loss_adv
        # print(f"Loss Clean: {loss_clean.item()} | Loss Adv: {loss_adv.item()}")

        # Step 4: Backpropagation
        fabric.backward(loss)
        optimizer.step()

        if lr_scheduler:
            lr_scheduler.step()

        running_loss += loss.item()
        precision_metric.update(outputs_clean, labels_clean)
        recall_metric.update(outputs_clean, labels_clean)
        accuracy_metric.update(outputs_clean, labels_clean)
    
    avg_loss = running_loss / len(train_loader_real_0)

    # Compute metrics at the end of the epoch
    precision = precision_metric.compute()[1].item()
    recall = recall_metric.compute()[1].item()
    accuracy = accuracy_metric.compute().item()

    return avg_loss, accuracy, precision, recall


@torch.no_grad()
def validate(fabric, model, dataloader, loss_func, num_classes):
    model.eval()
    test_loss = 0

    precision_metric = metrics.MulticlassPrecision(num_classes=num_classes, average=None).to(fabric.device)
    recall_metric = metrics.MulticlassRecall(num_classes=num_classes, average=None).to(fabric.device)
    accuracy_metric = metrics.MulticlassAccuracy(num_classes=num_classes, average='macro').to(fabric.device)

    for inputs, labels in dataloader:
        outputs = model(inputs)
        test_loss += loss_func(outputs, labels).item()

        precision_metric.update(outputs, labels)
        recall_metric.update(outputs, labels)
        accuracy_metric.update(outputs, labels)

    # avg_val_loss = test_loss / len(dataloader.dataset)
    avg_val_loss = test_loss / len(dataloader)
    precision = precision_metric.compute()
    recall = recall_metric.compute()
    accuracy = accuracy_metric.compute().item()
    
    precision = precision[1].item()
    recall = recall[1].item()
    return avg_val_loss, accuracy, precision, recall

def select_transform(norm_type, input_shape):
    match norm_type:
        case 0:
            # No operation (no-op): Do not apply any transformation
            return None
        case 1:
            # Only resize
            return transforms.Compose([
                ConditionalResize(input_shape)  # Only resize
            ])
        case 2:
            # Sample-wise normalization to [0, 1]
            return transforms.Compose([
                ConditionalResize(input_shape),
                NormalizeToRange(new_min=0, new_max=1)  # Normalize sample-wise
            ])
        case 3:
            return transforms.Compose([
                ConditionalResize(input_shape),
                transforms.ToTensor(),
            ])
        case 6:
            # Median normalization along the specified axis
            return transforms.Compose([
                ConditionalResize(input_shape),
                MedianNormalize(axis=1)  # Example: row-wise median normalization
            ])
        case _:
            return None  # Fallback: no operation (same as case 0)

def get_next_version(model_name, output_folder):
    """
    Finds the highest version of the model in the output folder and returns the next version number.
    """
    # Using a regular expression to match model versions like 'model_vX.pt'
    version_pattern = re.compile(rf"{model_name}_v(\d+)\.pt")

    # List all model files in the output folder
    version_numbers = []
    for file in output_folder.iterdir():
        match = version_pattern.match(file.name)
        if match:
            version_numbers.append(int(match.group(1)))

    # Return the next version number or 0
    return max(version_numbers, default=-1) + 1

def create_balanced_indices_across_datasets(train_dataset, aug_dataset):
    """
    Creates balanced indices across both train and augmented datasets.
    Ensures the final number of samples per class is equal.

    Args:
        train_dataset: Main training dataset.
        aug_dataset: Augmented dataset.

    Returns:
        train_indices: Indices for the balanced train dataset.
        aug_indices: Indices for the balanced augmented dataset.
    """
    # Extract labels for both datasets
    train_labels = [train_dataset[idx][1] for idx in range(len(train_dataset))]
    aug_labels = [aug_dataset[idx][1] for idx in range(len(aug_dataset))] if aug_dataset else []

    
    # saving the indexes of each class occurrence in a dict for each class
    train_class_counts = {}
    for idx, label in enumerate(train_labels):
        train_class_counts[label] = train_class_counts.get(label, []) + [idx]
 
    aug_class_counts = {}
    for idx, label in enumerate(aug_labels):
        aug_class_counts[label] = aug_class_counts.get(label, []) + [idx]

    # Determine the final number of samples per class
    final_class_counts = {}
    for cls in train_class_counts.keys():
        train_count = len(train_class_counts.get(cls, []))
        aug_count = len(aug_class_counts.get(cls, []))
        # The total number of samples per class should be the minimum available count across all classes
        # total_class_count = min(, min(train_count, aug_count) * 2)
        final_class_counts[cls] = train_count + aug_count

    min_count = min(final_class_counts.values())

    # Create balanced indices
    train_indices, aug_indices = [], []
    for cls, _ in final_class_counts.items():

        train_available = train_class_counts.get(cls, [])
        aug_available = aug_class_counts.get(cls, [])

        # Use all augmented samples if available
        aug_sampled = aug_available[:min(len(aug_available), min_count)]

        # Take the remaining from train dataset
        remaining_count = min_count - len(aug_sampled)
        train_sampled = torch.randperm(len(train_available))[:remaining_count].tolist()

        # Store final indices
        train_indices.extend([train_available[i] for i in train_sampled])
        aug_indices.extend(aug_sampled)

    return train_indices, aug_indices

def create_balanced_indices(labels):
    """
    Create balanced indices for each label.

    Args:
        labels: List of labels.

    Returns:
        List of balanced indices.
    """
    # Gather all indices by class
    class_indices = {}
    for idx, label in enumerate(labels):
        class_indices.setdefault(label, []).append(idx)

    # Determine the minimum class size
    min_count = min(len(indices) for indices in class_indices.values())

    # Randomly sample min_count indices from each class
    balanced_indices = []
    for cls, indices in class_indices.items():
        # randperm will create a tensor length indices with random integers from 0 to len(indices).
        # From this tensor we will take the first min_count indices which effectively corresponds to
        # sampling min_count indicies (similar behaviour to numpy.random.choice)
        sampled_relative_indices = torch.randperm(len(indices))[:min_count].tolist()
        sampled_indices = [indices[i] for i in sampled_relative_indices]  # Map to global indices
        balanced_indices.extend(sampled_indices)

    return balanced_indices

def count_samples_by_class(dataset):
    """
    Count the number of samples per class in a dataset.

    Args:
        dataset: The dataset to count samples from.

    Returns:
        A dictionary with class labels as keys and sample counts as values.
    """
    from collections import defaultdict
    class_counts = defaultdict(int)
    for idx in range(len(dataset)):
        _, label = dataset[idx]  # Ensure label is correctly accessed
        class_counts[label] += 1
    return dict(class_counts)

def create_per_class_datasets(paths, transform=None):
    """
    Returns a dict of {class_label: ImageDataset}
    """
    paths_by_class = defaultdict(list)
    for folder_path in paths:
        if not os.path.exists(folder_path):
                raise FileNotFoundError(f"Dataset path does not exist: {folder_path}")

        if not os.path.isdir(folder_path):
            raise NotADirectoryError(f"Expected a directory but got a file: {folder_path}")
        
        # Infer the label from the last subfolder
        label = int(os.path.basename(folder_path))
        paths_by_class[label].append(folder_path)

    datasets = []
    for class_label, folder_paths in paths_by_class.items():
        datasets.append(ImageDataset(
            paths=folder_paths, 
            label=class_label, 
            transform=transform
        ))
    return datasets 

def main(dataset, mode='img', train_set='/train', aug_set=None, val_set='/test', output_folder=None, 
         train_batch_size=32, input_shape=(128,128), eval_batch_size=32, num_epochs=20, model_name='my_model', norm_type=2, norm_type_aug=0, 
         versioning=False, seed=None):
    
    if seed:
        # Set seeds for reproducibility
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

    if output_folder is None:
        output_folder = Path('.').resolve()
    else:
        output_folder = Path(output_folder).resolve()

    output_folder.mkdir(parents=True, exist_ok=True)

    # Initialize CSV logger
    logger = CSVLogger(output_folder, name='logs')

    fabric = Fabric(loggers=logger)
    num_classes = 2

    # Model file naming logic based on versioning parameter
    if versioning:
        # If versioning is True, version the model by checking existing files
        version = get_next_version(model_name, output_folder)

        model_path = output_folder / f"{model_name}_v{version}.pt"
    else:
        # If versioning is False, always use the same model name
        model_path = output_folder / f"{model_name}.pt"


    train_transform = select_transform(norm_type, input_shape)

    train_set = [os.path.normpath(table).lstrip(os.sep) for table in train_set]
    train_paths = [os.path.join(dataset, table) for table in train_set]

    train_datasets = create_per_class_datasets(train_paths, transform=train_transform)
    train_dataset = ConcatDataset(train_datasets) if len(train_datasets) > 1 else train_datasets[0]
    
    if aug_set:
        aug_set = [os.path.normpath(table).lstrip(os.sep) for table in aug_set]
        aug_paths = [os.path.join(dataset, table) for table in aug_set]

        aug_datasets = create_per_class_datasets(aug_paths, transform=train_transform)
        aug_dataset = ConcatDataset(aug_datasets) if len(aug_datasets) > 1 else aug_datasets[0]
    else:
        aug_dataset = None  # No adversarial dataset


    val_set = [os.path.normpath(table).lstrip(os.sep) for table in val_set]
    val_paths = [os.path.join(dataset, table) for table in val_set]
    # Load images from the folder

    test_datasets = create_per_class_datasets(val_paths, transform=train_transform)
    test_dataset = ConcatDataset(test_datasets) if len(test_datasets) > 1 else test_datasets[0]

    # labels = [train_dataset[idx][1] for idx in range(len(train_dataset))]  # Extract labels

    # Create balanced indices
    print("Creating balanced subset for training dataset...")
    # balanced_indices = create_balanced_indices(labels)  # Generate balanced indices
    train_indices, aug_indices = create_balanced_indices_across_datasets(train_dataset, aug_dataset)

    # Creating indices for separate class samplers
    indices_real_0 = [i for i in train_indices if train_dataset[i][1] == 0]
    indices_real_1 = [i for i in train_indices if train_dataset[i][1] == 1]

    indices_aug_0 = [i for i in aug_indices if aug_dataset[i][1] == 0]
    indices_aug_1 = [i for i in aug_indices if aug_dataset[i][1] == 1]

    # Get the actual counts of samples
    num_real_1 = len(indices_real_1)
    num_aug_1 = len(indices_aug_1)
    total_class_1 = num_real_1 + num_aug_1

    # Calculate dynamic proportions
    p_real = num_real_1 / total_class_1
    p_aug = num_aug_1 / total_class_1

    # Compute the batch sizes based on proportions
    batch_size_real_1 = int((train_batch_size // 2) * p_real)
    batch_size_aug_1 = int((train_batch_size // 2) * p_aug)

    # Ensure batch sizes add up correctly (adjust if rounding errors)
    remaining = (train_batch_size // 2) - (batch_size_real_1 + batch_size_aug_1)
    batch_size_real_1 += remaining  # Adjust real class 1 batch size to compensate
    
    # Get the actual counts of samples for class 0
    num_real_0 = len(indices_real_0)
    num_aug_0 = len(indices_aug_0)
    total_class_0 = num_real_0 + num_aug_0

    # Calculate dynamic proportions for class 0
    p_real_0 = num_real_0 / total_class_0
    p_aug_0 = num_aug_0 / total_class_0

    # Compute batch sizes for class 0
    batch_size_real_0 = int((train_batch_size // 2) * p_real_0)
    batch_size_aug_0 = int((train_batch_size // 2) * p_aug_0)

    # Ensure batch sizes add up correctly (adjust for rounding errors)
    remaining_0 = (train_batch_size // 2) - (batch_size_real_0 + batch_size_aug_0)
    batch_size_real_0 += remaining_0  # Adjust real class 0 batch size to compensate
    
    
    
    print("Setting up Fabric...")

    # Create DataLoaders for each class
    train_loader_real_0 = DataLoader(
        train_dataset,
        batch_size=batch_size_real_0,  # Half batch from real class 0
        sampler=SubsetRandomSampler(indices_real_0),
        shuffle=False,
        num_workers=4
    )

    train_loader_aug_0 = DataLoader(
        train_dataset,
        batch_size=batch_size_aug_0,  # Half batch from real class 0
        sampler=SubsetRandomSampler(indices_real_0),
        shuffle=False,
        num_workers=4
    )

    train_loader_real_1 = DataLoader(
        train_dataset,
        batch_size=batch_size_real_1,  # Quarter batch from real class 1
        sampler=SubsetRandomSampler(indices_real_1),
        shuffle=False,
        num_workers=4
    )

    train_loader_aug_1 = DataLoader(
        aug_dataset,
        batch_size=batch_size_aug_1,  # Quarter batch from augmented class 1
        sampler=SubsetRandomSampler(aug_indices),
        shuffle=False,
        num_workers=4
    )



    test_loader = DataLoader(test_dataset, batch_size=eval_batch_size, shuffle=False, num_workers=4)

    print("Setting up DataLoaders...")
    train_loader_real_0, train_loader_aug_0, test_loader, train_loader_real_1, train_loader_aug_1 = fabric.setup_dataloaders(train_loader_real_0, train_loader_aug_0, test_loader, train_loader_real_1, train_loader_aug_1)   

    base_model = resnet18_for_single_channel() # usual resnet modified for singlechannel
    model = AdvPropResNet(base_model) #aux batch norm
    
    loss_func = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.999), weight_decay=0)

    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=500,
        num_training_steps=(len(train_loader_real_0) * num_epochs),
    )
    # lr_scheduler = None
    model, optimizer, lr_scheduler = fabric.setup(model, optimizer, lr_scheduler)
    # model, optimizer = fabric.setup(model, optimizer)
    
    train(fabric, model, optimizer, loss_func, train_loader_real_0, train_loader_aug_0, train_loader_real_1, train_loader_aug_1, val_loader=test_loader, lr_scheduler=lr_scheduler, num_epochs=num_epochs, num_classes=num_classes)

    logger.finalize("success")
    
    state = { # fabric will automatically unwrap the state_dict() when necessary
        "model": model, #necessary
        "optimizer": optimizer, 
        "lr_scheduler": lr_scheduler, 
        "epoch": num_epochs
    }
    fabric.save(model_path, state)
    # torch.save(model.state_dict(), model_path)
    
    print('Finished Training')

# Example usage
if __name__ == "__main__":
    import argparse

    def boolean_string(s):
        if s not in {'False', 'True'}:
            raise ValueError('Not a valid boolean string')
        return s == 'True'
    
    parser = argparse.ArgumentParser(description='Train a model on HDF5 dataset.')
    parser.add_argument('dataset', help='Path to the HDF5 dataset file, or root image folder')
    parser.add_argument('--train_set', type=str, nargs='+', default=None, help=(
        'For HDF5 mode: table name(s) for training data (e.g., /train).\n'
        'For img mode: path(s) to training image folders.'))
    parser.add_argument('--aug_set', type=str, nargs='+', default=None, help=(
        'For HDF5 mode: table name for augmented data (e.g., /aug).\n'
        'For img mode: path to folder containing augmented image data.'))
    parser.add_argument('--val_set', type=str, nargs='+', default=None, help=(
        'For HDF5 mode: table name(s) for validation data (e.g., /val).\n'
        'For img mode: path(s) to validation image folders.'))
    parser.add_argument('--output_folder', default=None, type=str, help='Folder to save the trained model.')
    parser.add_argument('--train_batch_size', type=int, default=32, help='Batch size for training.')
    parser.add_argument('--eval_batch_size', type=int, default=32, help='Batch size for evaluation.')
    parser.add_argument('--norm_type', type=int, default=2, help=(
        'Type of normalization/standardization to apply. Default is 2. Options are:\n'
        '0 - No operation.\n'
        '1 - Only resize the images without normalization or standardization.\n'
        '2 - Normalize each sample individually to the range [0, 1].\n'
        '3 - Normalize across the entire dataset to the range [0, 1].\n'
        '4 - Standardize each sample individually to have zero mean and unit variance.\n'
        '5 - Standardize across the entire dataset using dataset-wide mean and standard deviation values.'
    ))
    parser.add_argument('--norm_type_aug', type=int, default=0, help='Norm that will apply for the augmetned data')
    parser.add_argument('--num_epochs', type=int, default=20, help='Number of training epochs.')
    parser.add_argument('--input_shape', type=int, nargs='+', default=[128, 128], help='Input shape as width and height (e.g., --input_shape 128 128).')
    parser.add_argument('--model_name', type=str, default='my_model', help='name of the model')
    parser.add_argument('--versioning', type=boolean_string, default='False', help='If True, the model name will be versioned (e.g., v_0, v_1, etc.) based on the models already saved in the output path.')
    parser.add_argument("--seed", type=int, default=None, help="Random seed.")

    args = parser.parse_args()

    if len(args.input_shape) == 1:
        input_shape = (args.input_shape[0], args.input_shape[0])
    elif len(args.input_shape) == 2:
        input_shape = tuple(args.input_shape) #convert to tuple
    else:
        parser.error("--input_shape must be one or two integers.")

    main(args.dataset, train_set=args.train_set, aug_set=args.aug_set, val_set=args.val_set, 
         output_folder=args.output_folder, train_batch_size=args.train_batch_size, eval_batch_size=args.eval_batch_size, 
         norm_type=args.norm_type, norm_type_aug=args.norm_type_aug, num_epochs=args.num_epochs, input_shape=input_shape,
         model_name=args.model_name, versioning=args.versioning, seed=args.seed)
