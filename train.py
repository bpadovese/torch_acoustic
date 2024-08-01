import time
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from data_handling.dataset import HDF5Dataset, NormalizeToRange, Standardize
from torch.utils.data import DataLoader 
from pathlib import Path
from torchinfo import summary
from lightning.fabric import Fabric
from lightning.fabric.loggers import CSVLogger
import torcheval.metrics as metrics
from torch.optim.lr_scheduler import CosineAnnealingLR


def resnet18_for_single_channel():
    model = models.resnet18(weights=None)

    # Modifying the input layer
    model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

    # Modifying the fully connected layer
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)

    return model

def train(fabric, model, optimizer, loss_func, train_loader, val_loader=None, num_epochs=20, num_classes=2):
    print("Training starting...")
    for epoch in range(num_epochs):
        start_time = time.time()  # Start timer
        model.train()
        train_loss = train_epoch(fabric, model, optimizer, loss_func, train_loader, epoch)
        # Validation step
        avg_val_loss, val_accuracy, val_precision, val_recall = validate(fabric, model, val_loader, loss_func, num_classes)
        end_time = time.time()  # End timer
        epoch_duration = end_time - start_time
        print(f'Epoch {epoch + 1}, Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%, Validation Precision: {val_precision:.4f}, Validation Recall: {val_recall:.4f}, Duration: {epoch_duration:.2f} seconds')

        # Log metrics
        fabric.log_dict({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'validation_loss': avg_val_loss,
            'validation_accuracy': val_accuracy,
            'validation_precision': val_precision,
            'validation_recall': val_recall,
        })

def train_epoch(fabric, model, optimizer, loss_func, train_loader, epoch):
    running_loss = 0.0
    for batch_idx, (inputs,labels) in enumerate(train_loader, 0):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_func(outputs, labels)
        fabric.backward(loss)
        optimizer.step()
        running_loss += loss.item()
    avg_loss = running_loss / len(train_loader)
    print(f'Epoch {epoch + 1}, Loss: {avg_loss}')
    return avg_loss 


@torch.no_grad()
def validate(fabric, model, dataloader, loss_func, num_classes):
    model.eval()
    test_loss = 0

    precision_metric = metrics.MulticlassPrecision(num_classes=num_classes, average='micro').to(fabric.device)
    recall_metric = metrics.MulticlassRecall(num_classes=num_classes, average='micro').to(fabric.device)
    accuracy_metric = metrics.MulticlassAccuracy(num_classes=num_classes, average='micro').to(fabric.device)

    for inputs, labels in dataloader:
        outputs = model(inputs)
        test_loss += loss_func(outputs, labels).item()

        precision_metric.update(outputs, labels)
        recall_metric.update(outputs, labels)
        accuracy_metric.update(outputs, labels)

    avg_val_loss = test_loss / len(dataloader.dataset)
    precision = precision_metric.compute().item()
    recall = recall_metric.compute().item()
    accuracy = accuracy_metric.compute().item()

    return avg_val_loss, accuracy, precision, recall

def main(dataset, train_table='/train', val_table='/test', output_folder=None, train_batch_size=32, eval_batch_size=32, num_epochs=20, model_name='my_model', norm_type=1):
    
    if output_folder is None:
        output_folder = Path('.').resolve()
    else:
        output_folder = Path(output_folder).resolve()

    output_folder.mkdir(parents=True, exist_ok=True)

    # Initialize CSV logger
    logger = CSVLogger(output_folder, name='logs')

    fabric = Fabric(loggers=logger)
    num_classes = 2

    if model_name is None:
        model_name = "my_model.pt"
    
    model_path = output_folder / model_name
    if not model_path.suffix:
        model_path = model_path.with_suffix('.pt')

    train_dataset = HDF5Dataset(dataset, train_table, transform=None)
    test_dataset = HDF5Dataset(dataset, val_table, transform=None)

     # Get the dataset-level statistics for feature-wise normalization/standardization
    dataset_min = train_dataset.min_value
    dataset_max = train_dataset.max_value
    dataset_mean = train_dataset.mean_value
    dataset_std = train_dataset.std_value

    transform = None
    match norm_type:
        case 0:
            # Default: Only resize
            transform = transforms.Compose([
                transforms.Resize((128, 128))  # Only resize
            ])
        case 1:
            # Sample-wise normalization to [0, 1]
            transform = transforms.Compose([
                transforms.Resize((128, 128)),
                NormalizeToRange(new_min=0, new_max=1)  # Normalize sample-wise
            ])
        case 2:
            # Feature-wise normalization to [0, 1]
            transform = transforms.Compose([
                transforms.Resize((128, 128)),
                NormalizeToRange(min_value=dataset_min, max_value=dataset_max, new_min=0, new_max=1)  # Normalize feature-wise
            ])
        case 3:
            # Sample-wise standardization
            transform = transforms.Compose([
                transforms.Resize((128, 128)),
                Standardize()  # Standardize sample-wise
            ])
        case 4:
            # Feature-wise standardization
            transform = transforms.Compose([
                transforms.Resize((128, 128)),
                Standardize(mean=dataset_mean, std=dataset_std)  # Standardize feature-wise
            ])
        case _:
            raise ValueError("Invalid norm_type value. It should be between 0 and 4.")

    train_dataset.set_transform(transform)
    test_dataset.set_transform(transform)
    
    train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=eval_batch_size, shuffle=False)

    train_loader, test_loader = fabric.setup_dataloaders(train_loader, test_loader)
    

    # Modify the ResNet-18 model for single-channel input
    model = resnet18_for_single_channel()
    loss_func = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), weight_decay=0)

    model, optimizer = fabric.setup(model, optimizer)

    train(fabric, model, optimizer, loss_func, train_loader, val_loader=test_loader, num_epochs=num_epochs, num_classes=num_classes)

    logger.finalize("success")
    torch.save(model.state_dict(), model_path)
    
    print('Finished Training')

# Example usage
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Train a model on HDF5 dataset.')
    parser.add_argument('dataset', help='Path to the HDF5 dataset file.')
    parser.add_argument('--train_table', default='/train', help='HDF5 table name for training data.')
    parser.add_argument('--val_table', default='/test', help='HDF5 table name for validation data.')
    parser.add_argument('--output_folder', default=None, type=str, help='Folder to save the trained model.')
    parser.add_argument('--train_batch_size', type=int, default=32, help='Batch size for training.')
    parser.add_argument('--eval_batch_size', type=int, default=32, help='Batch size for evaluation.')
    parser.add_argument('--norm_type', type=int, default=1, help=(
        'Type of normalization/standardization to apply. Default is 0. Options are:\n'
        '0 - Only resize the images without normalization or standardization.\n'
        '1 - Normalize each sample individually to the range [0, 1].\n'
        '2 - Normalize across the entire dataset to the range [0, 1].\n'
        '3 - Standardize each sample individually to have zero mean and unit variance.\n'
        '4 - Standardize across the entire dataset using dataset-wide mean and standard deviation values.'
    ))
    parser.add_argument('--num_epochs', type=int, default=20, help='Number of training epochs.')
    parser.add_argument('--model_name', type=str, default='my_model', help='name of the model')

    args = parser.parse_args()
    main(args.dataset, args.train_table, args.val_table, args.output_folder, args.train_batch_size, args.eval_batch_size, args.num_epochs, args.model_name)
