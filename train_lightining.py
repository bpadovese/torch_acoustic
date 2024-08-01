import os
import torch
from torch import nn
import torchvision.models as models
import torch.nn.functional as F
from dev_misc.classifier.dataset import HDF5Dataset, NormalizeToRange
from torchvision import transforms
from torch.utils.data import DataLoader
from pathlib import Path
import torcheval.metrics as metrics
import lightning as L

class ResNet18SingleChannel(L.LightningModule):
    def __init__(self, num_classes=2, lr=0.001):
        super(ResNet18SingleChannel, self).__init__()
        self.save_hyperparameters()
        self.model = self._build_model(num_classes)
        self.loss_func = nn.CrossEntropyLoss()
        self.precision_metric = metrics.MulticlassPrecision(num_classes=num_classes, average='micro')
        self.recall_metric = metrics.MulticlassRecall(num_classes=num_classes, average='micro')
        self.accuracy_metric = metrics.MulticlassAccuracy(num_classes=num_classes, average='micro')

    def _build_model(self, num_classes):
        model = models.resnet18(weights=None)
        model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
        return model

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        loss = self.loss_func(outputs, labels)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        loss = self.loss_func(outputs, labels)
        self.precision_metric(outputs, labels)
        self.recall_metric(outputs, labels)
        self.accuracy_metric(outputs, labels)
        self.log('val_loss', loss)
        self.log('val_precision', self.precision_metric, on_step=False, on_epoch=True)
        self.log('val_recall', self.recall_metric, on_step=False, on_epoch=True)
        self.log('val_accuracy', self.accuracy_metric, on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        return optimizer

def main(dataset, train_table='/train', val_table='/test', output_folder=None, train_batch_size=32, eval_batch_size=32, num_epochs=20):
    # Define the transformations
    transform = transforms.Compose([
        transforms.Resize((128, 128)), # Resize to 128x128
        NormalizeToRange(new_min=0, new_max=1)  # Normalize to range [0, 1]
    ])

    train_dataset = HDF5Dataset(dataset, train_table, transform=transform)
    val_dataset = HDF5Dataset(dataset, val_table, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=eval_batch_size, shuffle=False)
    
    # Instantiate the model
    model = ResNet18SingleChannel()

    # Instantiate the trainer
    trainer = L.Trainer(max_epochs=num_epochs, default_root_dir=output_folder)
    
    # Train the model
    trainer.fit(model, train_loader, val_loader)

    # Save the trained model
    if output_folder is not None:
        output_path = Path(output_folder) / 'resnet18_single_channel.pth'
        torch.save(model.state_dict(), output_path)
        print(f'Model saved to {output_path}')

# Example usage
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Train a model on HDF5 dataset.')
    parser.add_argument('dataset', help='Path to the HDF5 dataset file.')
    parser.add_argument('--train_table', default='/train', help='HDF5 table name for training data.')
    parser.add_argument('--val_table', default='/test', help='HDF5 table name for validation data.')
    parser.add_argument('--output_folder', default=None, help='Folder to save the trained model.')
    parser.add_argument('--train_batch_size', type=int, default=32, help='Batch size for training.')
    parser.add_argument('--eval_batch_size', type=int, default=32, help='Batch size for evaluation.')
    parser.add_argument('--num_epochs', type=int, default=20, help='Number of training epochs.')

    args = parser.parse_args()
    main(args.dataset, args.train_table, args.val_table, args.output_folder, args.train_batch_size, args.eval_batch_size, args.num_epochs)