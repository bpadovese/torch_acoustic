import time
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import random
import sys
import numpy as np
import os
from torch.utils.data import DataLoader, ConcatDataset, Subset
from pathlib import Path
from lightning.fabric import Fabric
from lightning.fabric.loggers import CSVLogger
from diffusers.optimization import get_cosine_schedule_with_warmup
from dev_utils.nn import resnet18_for_single_channel
from data_handling.dataset import ImageDataset, ConditionalResize, NormalizeToRange, MedianNormalize
import torcheval.metrics as metrics


def select_transform(norm_type, input_shape):
    match norm_type:
        case 1:
            return transforms.Compose([ConditionalResize(input_shape)])
        case 2:
            return transforms.Compose([ConditionalResize(input_shape), NormalizeToRange(0, 1)])
        case 3:
            return transforms.Compose([ConditionalResize(input_shape), transforms.ToTensor()])
        case 6:
            return transforms.Compose([ConditionalResize(input_shape), MedianNormalize(axis=1)])
        case _:
            return None

def create_datasets(paths, transform, num_samples):
    datasets = []
    for i, path in enumerate(paths):
        label = int(os.path.basename(path))
        dataset = ImageDataset([path], label=label, transform=transform)
        if num_samples[i] is not None:
            sampled_indices = random.sample(range(len(dataset)), min(num_samples[i], len(dataset)))
            dataset = Subset(dataset, sampled_indices)
        datasets.append(dataset)
    return ConcatDataset(datasets)

def train_epoch(model, loader, loss_func, optimizer, fabric):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    precision_metric = metrics.MulticlassPrecision(num_classes=2, average=None).to(fabric.device)
    recall_metric = metrics.MulticlassRecall(num_classes=2, average=None).to(fabric.device)
    for inputs, labels in loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_func(outputs, labels)
        fabric.backward(loss)
        optimizer.step()
        running_loss += loss.item()
        preds = outputs.argmax(1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        precision_metric.update(outputs, labels)
        recall_metric.update(outputs, labels)
    precision = precision_metric.compute()[1].item()
    recall = recall_metric.compute()[1].item()
    return running_loss / len(loader), 100.0 * correct / total, precision, recall

def validate(model, loader, loss_func, fabric):
    model.eval()
    loss_total, correct, total = 0.0, 0, 0
    precision_metric = metrics.MulticlassPrecision(num_classes=2, average=None).to(fabric.device)
    recall_metric = metrics.MulticlassRecall(num_classes=2, average=None).to(fabric.device)
    with torch.no_grad():
        for inputs, labels in loader:
            outputs = model(inputs)
            loss_total += loss_func(outputs, labels).item()
            preds = outputs.argmax(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            precision_metric.update(outputs, labels)
            recall_metric.update(outputs, labels)
    precision = precision_metric.compute()[1].item()
    recall = recall_metric.compute()[1].item()
    return loss_total / len(loader), 100.0 * correct / total, precision, recall

def main(dataset_root, train_dirs, val_dirs, num_samples, output_folder, input_shape=(128,128),
         norm_type=2, batch_size=32, num_epochs=20, model_name="model", seed=None):

    if seed:
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

    output_folder = Path(output_folder or '.').resolve()
    output_folder.mkdir(parents=True, exist_ok=True)

    logger = CSVLogger(output_folder, name='logs')
    fabric = Fabric(loggers=logger)

    transform = select_transform(norm_type, input_shape)
    train_paths = [os.path.join(dataset_root, p) for p in train_dirs]
    val_paths = [os.path.join(dataset_root, p) for p in val_dirs]

    train_dataset = create_datasets(train_paths, transform, num_samples)
    val_dataset = create_datasets(val_paths, transform, [None]*len(val_paths))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    train_loader, val_loader = fabric.setup_dataloaders(train_loader, val_loader)

    model = resnet18_for_single_channel()
    loss_func = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = get_cosine_schedule_with_warmup(optimizer, 500, len(train_loader)*num_epochs)
    model, optimizer, scheduler = fabric.setup(model, optimizer, scheduler)

    for epoch in range(num_epochs):
        start = time.time()
        train_loss, train_acc, train_prec, train_rec = train_epoch(model, train_loader, loss_func, optimizer, fabric)
        val_loss, val_acc, val_prec, val_rec = validate(model, val_loader, loss_func, fabric)
        scheduler.step()

        fabric.log_dict({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_accuracy': train_acc,
            'train_precision': train_prec,
            'train_recall': train_rec,
            'val_loss': val_loss,
            'val_accuracy': val_acc,
            'val_precision': val_prec,
            'val_recall': val_rec,
        })

        print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%, Prec: {train_prec:.4f}, Rec: {train_rec:.4f} | Val Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%, Prec: {val_prec:.4f}, Rec: {val_rec:.4f} | Time: {time.time()-start:.2f}s")

    fabric.save(output_folder / f"{model_name}.pt", {"model": model})
    logger.finalize("success")
    print("Training complete.")

def save_command(output_folder):
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    command_str = ' '.join(sys.argv)
    with open(output_folder / "command.txt", 'w') as f:
        f.write(command_str + '\n')

if __name__ == "__main__":
    import argparse

    def parse_num_samples(value):
        return None if value.lower() == 'none' else int(value)

    parser = argparse.ArgumentParser(description='Train a model on image folders.')
    parser.add_argument('dataset_root', help='Root folder containing class-labeled subfolders')
    parser.add_argument('--train_dirs', type=str, nargs='+', required=True, help='List of training subfolder names')
    parser.add_argument('--val_dirs', type=str, nargs='+', required=True, help='List of validation subfolder names')
    parser.add_argument('--num_samples', type=parse_num_samples, nargs='+', required=True, help='Number of samples to load per training folder')
    parser.add_argument('--output_folder', type=str, default='.', help='Where to save the model and logs')
    parser.add_argument('--input_shape', type=int, nargs=2, default=[128, 128], help='Input shape of the images')
    parser.add_argument('--norm_type', type=int, default=2, help='Normalization type (0=none, 1=resize, 2=normalize, 3=ToTensor, 6=MedianNormalize)')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training and validation')
    parser.add_argument('--num_epochs', type=int, default=20, help='Number of training epochs')
    parser.add_argument('--model_name', type=str, default='model', help='Model name for saving')
    parser.add_argument('--seed', type=int, default=None, help='Random seed')

    args = parser.parse_args()

    save_command(args.output_folder)

    main(
        dataset_root=args.dataset_root,
        train_dirs=args.train_dirs,
        val_dirs=args.val_dirs,
        num_samples=args.num_samples,
        output_folder=args.output_folder,
        input_shape=tuple(args.input_shape),
        norm_type=args.norm_type,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        model_name=args.model_name,
        seed=args.seed
    )
    