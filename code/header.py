## header.py
# Holds all the classes and functions that are used in train.py

import os
import json
import torch
import timm
import pydicom
import typing
import random

import albumentations as A
import pytorch_lightning as pl

from torch.utils.data import Dataset, DataLoader
from torchmetrics.classification import Accuracy, AUROC, F1Score
from albumentations.pytorch import ToTensorV2

class DICOMDataset(Dataset):
    def __init__(self, json_data: typing.List[dict], transform=None):
        self.dicom_files = json_data
        self.transform = transform

    def __len__(self):
        return len(self.dicom_files)

    def __getitem__(self, idx):
        dicom_file = self.dicom_files[idx]["dcmpath"]
        label = self.dicom_files[idx]["Binary"]
        dicom_data = pydicom.dcmread(os.path.join(os.path.dirname(__file__), "..", dicom_file))
        image = dicom_data.pixel_array

        if len(image.shape) > 2:  # Some US are doppler, so have 3 channels
            image = image[:, :, 0]

        cropping_rules = {
            (720, 960): ((110, 600), (150, 700)),
            (768, 1024): ((200, 650), (170, 900)),
            (873, 1164): ((125, 775), (0, 1000)),
            (960, 1280): ((165, 770), (230, 1050)),
        }
        if image.shape in cropping_rules:
            y_slice, x_slice = cropping_rules[image.shape]
            image = image[y_slice[0]:y_slice[1], x_slice[0]:x_slice[1]]

        if self.transform is not None:
            image = self.transform(image=image)["image"]

        return image, label

class DICOMDataModule(pl.LightningDataModule):
    def __init__(self, path_to_json: str, batch_size=32, random_state=1026):
        super().__init__()
        self.json_data = json.load(open(path_to_json)) 
        self.random_state = random_state
        self.batch_size = batch_size

    def setup(self, stage=None):
        random.seed(self.random_state)

        # Generate train/val/test split
        test = [x for x in self.json_data if x["Site"] == "BLM"]
        temp = [x for x in self.json_data if x not in test]
        random.shuffle(temp)
        split_idx = int(len(temp) * 0.8)
        train = temp[:split_idx]
        val = temp[split_idx:]

        # Define transforms
        train_transform = A.Compose([
            A.Resize(256, 256),
            A.Normalize(mean=0.223, std=0.203),  # mean and std of RadImageNet
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.5),
            ToTensorV2(),
        ])

        val_transform = A.Compose([
            A.Resize(256, 256),
            A.Normalize(mean=0.223, std=0.203),  # mean and std of RadImageNet
            ToTensorV2(),
        ])

        self.train_dataset = DICOMDataset(train, transform=train_transform)
        self.val_dataset = DICOMDataset(val, transform=val_transform)
        self.test_dataset = DICOMDataset(test, transform=val_transform)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size)

class ResNet50Module(pl.LightningModule):
    def __init__(self, num_classes=1, max_epochs=100):
        super().__init__()

        # Define hyperparameters
        self.num_classes = num_classes
        self.lr = 1e-3
        self.max_epochs = max_epochs

        # Define model
        self.model = timm.create_model('resnet50', pretrained=False, in_chans=1)
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, self.num_classes)
        weights_path = os.path.join(os.path.dirname(__file__), "../misc/RadImageNet-ResNet50_notop_torch.pth")
        rin_state_dict = torch.load(weights_path)

        # For conv1.weight, reduce it down to 1 channel by taking the mean across the 3 channels
        rin_state_dict["conv1.weight"] = rin_state_dict["conv1.weight"].mean(dim=1, keepdim=True)

        self.model.load_state_dict(rin_state_dict, strict=False)

        # Define metrics
        if self.num_classes == 1:
            self.accuracy = Accuracy(task="binary")
            self.AUROC = AUROC(task="binary")
            self.f1 = F1Score(task="binary")
            self.loss = torch.nn.BCEWithLogitsLoss()
        else:
            self.accuracy = Accuracy(task="multiclass", num_classes=self.num_classes)
            self.AUROC = AUROC(task="multiclass", num_classes=self.num_classes)
            self.f1 = F1Score(task="multiclass", num_classes=self.num_classes)
            self.loss = torch.nn.CrossEntropyLoss()

        self.save_hyperparameters()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_float = y.float()
        y_hat = torch.squeeze(self.forward(x), dim=1)
        loss = self.loss(y_hat, y_float)
        self.log('train_loss', loss)
        self.log("train_acc", self.accuracy(y_hat, y), on_step=False, on_epoch=True, sync_dist=True)
        self.log("train_auroc", self.AUROC(y_hat, y), on_step=False, on_epoch=True, sync_dist=True)
        self.log("train_f1", self.f1(y_hat, y), on_step=False, on_epoch=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_float = y.float()
        y_hat = torch.squeeze(self.forward(x), dim=1)
        loss = self.loss(y_hat, y_float)
        self.log('val_loss', loss)
        self.log("val_acc", self.accuracy(y_hat, y), on_step=False, on_epoch=True, sync_dist=True)
        self.log("val_auroc", self.AUROC(y_hat, y), on_step=False, on_epoch=True, sync_dist=True)
        self.log("val_f1", self.f1(y_hat, y), on_step=False, on_epoch=True, sync_dist=True)
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        y_float = y.float()
        y_hat = torch.squeeze(self.forward(x), dim=1)
        loss = self.loss(y_hat, y_float)
        test_acc = self.accuracy(y_hat, y)
        test_auroc = self.AUROC(y_hat, y)
        test_f1 = self.f1(y_hat, y)
        self.log_dict({
            "test_loss": loss,
            "test_acc": test_acc,
            "test_auroc": test_auroc,
            "test_f1": test_f1,
        })

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=self.lr)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.max_epochs)
        return {
            "optimizer": opt,
            "lr_scheduler": {
                "scheduler": sch,
                "monitor": "val_loss",
                "interval": "epoch",
                "frequency": 1,
            }
        }