import time
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import re
import numpy as np
from data_handling.dataset import HDF5Dataset, NormalizeToRange, Standardize, MedianNormalize, ConditionalResize, get_leaf_paths
from torch.utils.data import DataLoader, ConcatDataset, WeightedRandomSampler
from pathlib import Path
from collections import defaultdict
from torchinfo import summary
from diffusers.optimization import get_cosine_schedule_with_warmup 
from lightning.fabric import Fabric
from lightning.fabric.loggers import CSVLogger
import torcheval.metrics as metrics
from torch.optim.lr_scheduler import CosineAnnealingLR
# from torchtune.modules import get_cosine_schedule_with_warmup


def resnet18_for_single_channel():
    model = models.resnet18(weights=None)

    # Modifying the input layer
    model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

    # Modifying the fully connected layer
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)

    return model

def compute_class_counts(concat_dataset):
    """
    Computes the class counts across multiple HDF5Dataset objects by 
    reading only the first label from each table and using the `nrows` attribute.

    Note: Assumes that each table corresponds to a single label.
    
    Args:
        datasets: ConcatDatset of HDF5Dataset objects.
        
    Returns:
        label list
        Dictionary mapping class labels to their respective counts.
    """
    labels = []
    class_counts = defaultdict(int)

    for dataset in concat_dataset.datasets:
        # Read the first label to determine the class
        first_label = dataset[0][1].item()  # Fetch the label of the first sample
        
        # Use the `nrows` attribute to determine the number of samples in the table
        table_size = dataset.table.nrows
        labels.extend([first_label] * table_size)
        # Add the number of samples to the appropriate class
        class_counts[first_label] += table_size
    
    return labels, class_counts

def create_weighted_sampler(datasets):
    """
    Create a WeightedRandomSampler based on class counts.

    Args:
        class_counts: Dictionary of class counts.

    Returns:
        WeightedRandomSampler object.
    """

    labels, class_counts = compute_class_counts(datasets)

    # Calculate class weights (inverse of frequency)
    total_samples = sum(class_counts.values())
    class_weights = {cls: total_samples / count for cls, count in class_counts.items()}

    # Assign sample weights based on class weights
    sample_weights = torch.tensor(
        [class_weights[label] for label in labels], dtype=torch.float
    )

    # Calculate num_samples: minimum class count times the number of classes
    min_class_count = min(class_counts.values())
    num_samples = int(min_class_count * len(class_counts))   # Total samples to draw per epoch

    # Create the WeightedRandomSampler
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=num_samples,  # Total number of samples
        replacement=False
    )

    return sampler

def train(fabric, model, optimizer, loss_func, train_loader, lr_scheduler=None, val_loader=None, num_epochs=20, num_classes=2):
    print("Training starting...")
    for epoch in range(num_epochs):
        start_time = time.time()  # Start timer
        model.train()

        # Training step
        train_loss, train_accuracy, train_precision, train_recall = train_epoch(
            fabric, model, optimizer, loss_func, train_loader, epoch, lr_scheduler=lr_scheduler, num_classes=num_classes
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

def train_epoch(fabric, model, optimizer, loss_func, train_loader, epoch, lr_scheduler=None, num_classes=2):
    running_loss = 0.0

    # Initialize metrics
    precision_metric = metrics.MulticlassPrecision(num_classes=num_classes, average=None).to(fabric.device)
    recall_metric = metrics.MulticlassRecall(num_classes=num_classes, average=None).to(fabric.device)
    accuracy_metric = metrics.MulticlassAccuracy(num_classes=num_classes, average='macro').to(fabric.device)

    for batch_idx, (inputs,labels) in enumerate(train_loader, 0):
        # Count labels for each class
        # class_counts = {cls: (labels == cls).sum().item() for cls in range(num_classes)} # DEBUGGING
        # print(f"Batch {batch_idx} class counts: {class_counts}")
        # Zero the parameter gradients
        optimizer.zero_grad()
        # Forward pass
        outputs = model(inputs)
        # Compute loss
        loss = loss_func(outputs, labels)
        fabric.backward(loss)
        # Step optimizer
        optimizer.step()
        # Step learning rate scheduler if provided
        if lr_scheduler:
            lr_scheduler.step()
        # Update running loss
        running_loss += loss.item()

        # Update metrics
        precision_metric.update(outputs, labels)
        recall_metric.update(outputs, labels)
        accuracy_metric.update(outputs, labels)

    # Compute average loss for the epoch
    avg_loss = running_loss / len(train_loader)

    # Compute metrics for the entire epoch
    precision = precision_metric.compute()
    recall = recall_metric.compute()
    accuracy = accuracy_metric.compute().item()

    precision = precision[1].item()  # Precision for class 1 (KW)
    recall = recall[1].item()  # Recall for class 1 (KW)
    
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

def select_transform(norm_type, dataset_min=None, dataset_max=None, dataset_mean=None, dataset_std=None):
    match norm_type:
        case 0:
            # No operation (no-op): Do not apply any transformation
            return None
        case 1:
            # Only resize
            return transforms.Compose([
                ConditionalResize((128, 128))  # Only resize
            ])
        case 2:
            # Sample-wise normalization to [0, 1]
            return transforms.Compose([
                ConditionalResize((128, 128)),
                NormalizeToRange(new_min=0, new_max=1)  # Normalize sample-wise
            ])
        case 3:
            # Feature-wise normalization to [0, 1]
            return transforms.Compose([
                ConditionalResize((128, 128)),
                NormalizeToRange(min_value=dataset_min, max_value=dataset_max, new_min=0, new_max=1)  # Normalize feature-wise
            ])
        case 4:
            # Sample-wise standardization
            return transforms.Compose([
                ConditionalResize((128, 128)),
                Standardize()  # Standardize sample-wise
            ])
        case 5:
            # Feature-wise standardization
            return transforms.Compose([
                ConditionalResize((128, 128)),
                Standardize(mean=dataset_mean, std=dataset_std)  # Standardize feature-wise
            ])
        case 6:
            # Median normalization along the specified axis
            return transforms.Compose([
                ConditionalResize((128, 128)),
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

def main(dataset, train_table='/train', aug_table=None, val_table='/test', output_folder=None, train_batch_size=32, eval_batch_size=32, num_epochs=20, model_name='my_model', norm_type=2, norm_type_aug=0, versioning=False, dataset_stats=None):
    
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
   
    dataset_min = None
    dataset_max = None
    dataset_mean = None
    dataset_std = None

    # Select transforms for the training dataset
    train_transform = select_transform(norm_type, dataset_min, dataset_max, dataset_mean, dataset_std)

    print("Loading train dataset...")
    train_datasets = []
    # Handle train_table argument (single path or list of paths)
    if isinstance(train_table, str):
        train_table = [train_table]  # Convert to list if it's a single path
    for path in train_table:
        leaf_paths = get_leaf_paths(dataset, path)
        for leaf_path in leaf_paths:
            train_ds = HDF5Dataset(dataset, leaf_path, transform=None) 
            train_ds.set_transform(train_transform)
            train_datasets.append(train_ds)
    # train_dataset = ConcatDataset(train_datasets) if len(train_datasets) > 1 else train_datasets[0]

    print("Loading val dataset...")
    val_datasets = []
    # Handle val_table argument (single path or list of paths)
    if isinstance(val_table, str):
        val_table = [val_table]  # Convert to list if it's a single path
    for path in val_table:
        leaf_paths = get_leaf_paths(dataset, path)
        for leaf_path in leaf_paths:
            val_ds = HDF5Dataset(dataset, leaf_path, transform=None) 
            val_ds.set_transform(train_transform)
            val_datasets.append(val_ds)
    test_dataset = ConcatDataset(val_datasets) if len(val_datasets) > 1 else val_datasets[0]

    print("Loading augmentation dataset...")
    # Optional: Handle augmented dataset and apply transforms
    aug_dataset = None
    if aug_table:
        aug_dataset = HDF5Dataset(dataset, aug_table + '/data', transform=None)
        # Select the transforms for the augmented set
        aug_transform = select_transform(norm_type_aug, dataset_min, dataset_max, dataset_mean, dataset_std)
        aug_dataset.set_transform(aug_transform)
        train_datasets.append(aug_dataset)
    

    # Concating all train datasets together.
    train_dataset = ConcatDataset(train_datasets) if len(train_datasets) > 1 else train_datasets[0]
    
    # Create a WeightedRandomSampler for balanced sampling
    train_sampler = create_weighted_sampler(train_dataset)

    print("Setting up Fabric...")
    train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=False, sampler=train_sampler, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=eval_batch_size, shuffle=False, num_workers=4)

    print("Setting up DataLoaders...")
    train_loader, test_loader = fabric.setup_dataloaders(train_loader, test_loader)
    

    # Modify the ResNet-18 model for single-channel input
    model = resnet18_for_single_channel()
    loss_func = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.999), weight_decay=0)

    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=500,
        num_training_steps=(len(train_loader) * num_epochs),
    )
    # lr_scheduler = None
    model, optimizer, lr_scheduler = fabric.setup(model, optimizer, lr_scheduler)
    # model, optimizer = fabric.setup(model, optimizer)
    
    train(fabric, model, optimizer, loss_func, train_loader, val_loader=test_loader, lr_scheduler=lr_scheduler, num_epochs=num_epochs, num_classes=num_classes)

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
    parser.add_argument('dataset', help='Path to the HDF5 dataset file.')
    parser.add_argument('--train_table', type=str, nargs='+', default='/train', help='HDF5 table name for training data.')
    parser.add_argument('--aug_table', type=str, default=None, help='HDF5 table name for augmented data, this assumes a different set of transformations will apply.')
    parser.add_argument('--val_table', type=str, nargs='+', default='/test', help='HDF5 table name for validation data.')
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
    parser.add_argument('--model_name', type=str, default='my_model', help='name of the model')
    parser.add_argument('--versioning', type=boolean_string, default='False', help='If True, the model name will be versioned (e.g., v_0, v_1, etc.) based on the models already saved in the output path.')

    args = parser.parse_args()
    main(args.dataset, train_table=args.train_table, aug_table=args.aug_table, val_table=args.val_table, output_folder=args.output_folder, 
         train_batch_size=args.train_batch_size, eval_batch_size=args.eval_batch_size, norm_type=args.norm_type, 
         norm_type_aug=args.norm_type_aug, num_epochs=args.num_epochs, model_name=args.model_name, versioning=args.versioning)
