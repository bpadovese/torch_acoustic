import torch
import torch.nn as nn
import torcheval.metrics as metrics
import torchvision.models as models
import pandas as pd
import os
from lightning.fabric import Fabric
from data_handling.dataset import HDF5Dataset, NormalizeToRange, Standardize, MedianNormalize, ConditionalResize, get_leaf_paths
from torch.utils.data import DataLoader, ConcatDataset
from torchvision.transforms import transforms
from pathlib import Path
from dev_utils.nn import resnet18_for_single_channel, resnet50_for_single_channel

def select_transform(norm_type, dataset_min=None, dataset_max=None, dataset_mean=None, dataset_std=None):
    match norm_type:
        case 0:
            return None
        case 1:
            return transforms.Compose([
                ConditionalResize((128, 128))
            ])
        case 2:
            return transforms.Compose([
                ConditionalResize((128, 128)),
                NormalizeToRange(new_min=0, new_max=1)
            ])
        case 3:
            return transforms.Compose([
                ConditionalResize((128, 128)),
                NormalizeToRange(min_value=dataset_min, max_value=dataset_max, new_min=0, new_max=1)
            ])
        case 4:
            return transforms.Compose([
                ConditionalResize((128, 128)),
                Standardize()
            ])
        case 5:
            return transforms.Compose([
                ConditionalResize((128, 128)),
                Standardize(mean=dataset_mean, std=dataset_std)
            ])
        case 6:
            return transforms.Compose([
                ConditionalResize((128, 128)),
                MedianNormalize(axis=1)
            ])
        case _:
            return None

@torch.no_grad()
def evaluate(fabric, model, dataloader, loss_func, num_classes):
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

    avg_test_loss = test_loss / len(dataloader)
    precision = precision_metric.compute()[1].item()
    recall = recall_metric.compute()[1].item()
    accuracy = accuracy_metric.compute().item()

    return avg_test_loss, accuracy, precision, recall

def main(dataset, model_path, val_table='/test', eval_batch_size=32, norm_type=2, output_csv='result.csv', overwrite=True):
    fabric = Fabric()
    state = fabric.load(model_path)
    model = resnet18_for_single_channel()
    model.load_state_dict(state["model"])

    loss_func = nn.CrossEntropyLoss()
    dataset_min = None
    dataset_max = None
    dataset_mean = None
    dataset_std = None

    val_transform = select_transform(norm_type, dataset_min, dataset_max, dataset_mean, dataset_std)

    val_datasets = []
    if isinstance(val_table, str):
        val_table = [val_table]
    for path in val_table:
        leaf_paths = get_leaf_paths(dataset, path)
        for leaf_path in leaf_paths:
            val_ds = HDF5Dataset(dataset, leaf_path, transform=None)
            val_ds.set_transform(val_transform)
            val_datasets.append(val_ds)

    test_dataset = ConcatDataset(val_datasets) if len(val_datasets) > 1 else val_datasets[0]
    test_loader = DataLoader(test_dataset, batch_size=eval_batch_size, shuffle=False)

    test_loader = fabric.setup_dataloaders(test_loader)
    model = fabric.setup(model)

    avg_test_loss, accuracy, precision, recall = evaluate(fabric, model, test_loader, loss_func, num_classes=2)

    print(f'Test Loss: {avg_test_loss:.4f}')
    print(f'Test Accuracy: {accuracy:.2f}%')
    print(f'Test Precision: {precision:.4f}')
    print(f'Test Recall: {recall:.4f}')

    # Extract model name from the model path
    model_name = Path(model_path).stem


    # Save results to CSV
    results = {
        'Model Name': model_name,
        'Test Loss': avg_test_loss,
        'Test Accuracy': accuracy,
        'Test Precision': precision,
        'Test Recall': recall
    }

    df = pd.DataFrame([results])

    mode='a'
    if overwrite:
        mode='w'
        
    # Check if file exists
    file_exists = os.path.exists(output_csv)

    # Write to file
    df.to_csv(output_csv, index=False, mode=mode, header=not file_exists)
    print(f"Results saved to {output_csv}")


# Example usage
if __name__ == "__main__":
    import argparse
    
    def boolean_string(s):
        if s not in {'False', 'True'}:
            raise ValueError('Not a valid boolean string')
        return s == 'True'
    
    parser = argparse.ArgumentParser(description='Evaluate a trained model on a new dataset.')
    parser.add_argument('dataset', help='Path to the HDF5 dataset file.')
    parser.add_argument('model_path', help='Path to the trained model file.')
    parser.add_argument('--val_table', type=str, nargs='+', default='/test', help='HDF5 table name for evaluation data.')
    parser.add_argument('--eval_batch_size', type=int, default=32, help='Batch size for evaluation.')
    parser.add_argument('--norm_type', type=int, default=2, help=(
        'Type of normalization/standardization to apply. Default is 2. Options are:\n'
        '0 - No operation.\n'
        '1 - Only resize.\n'
        '2 - Normalize each sample individually to the range [0, 1].\n'
        '3 - Normalize across the entire dataset to the range [0, 1].\n'
        '4 - Standardize each sample individually.\n'
        '5 - Standardize across the entire dataset.'
    ))
    parser.add_argument('--output_csv', type=str, default='results.csv', help='Path to save the results as a CSV file.')
    parser.add_argument('--overwrite', type=boolean_string, default=True, help='Overwrites the results or append to it.')

    args = parser.parse_args()
    main(args.dataset, args.model_path, val_table=args.val_table, eval_batch_size=args.eval_batch_size, norm_type=args.norm_type, output_csv=args.output_csv, overwrite=args.overwrite)
