import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim
import models
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from pathlib import Path

output_root = Path("Combined_Dataset")

def main():

    IMG_SIZE = 224
    BATCH_SIZE = 32
    RESULTS_ROOT = "results"
    

    train_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    train_dataset = datasets.ImageFolder(
        root=output_root / "train",
        transform=train_transform
    )

    val_dataset = datasets.ImageFolder(
        root=output_root / "val",
        transform=eval_transform
    )

    test_dataset = datasets.ImageFolder(
        root=output_root / "test",
        transform=eval_transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2
    )

    print("\nClass mapping used by ImageFolder:")
    print(train_dataset.class_to_idx)

    print(f"\nNumber of training images: {len(train_dataset)}")
    print(f"Number of validation images: {len(val_dataset)}")
    print(f"Number of test images: {len(test_dataset)}")

    NUM_CLASSES = len(train_loader.dataset.classes)
    print(f"Number of classes: {NUM_CLASSES}")


    baselineANN = models.BaselineANN(NUM_CLASSES, IMG_SIZE)
    primaryCNN = models.PrimaryCNN(NUM_CLASSES, IMG_SIZE)
    transfer_model = models.TransferResNet18(
            num_classes=NUM_CLASSES,
            freeze_backbone=True
    )

    transferResults = models.train(
        model=transfer_model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=20,
        learning_rate=0.001,
        results_root=RESULTS_ROOT,
        save_results=False,
        show_plots=True
    )

    # baselineResults = models.train(baselineANN, 
    #                                train_loader=train_loader, 
    #                                val_loader=val_loader, 
    #                                num_epochs=30,
    #                                learning_rate=0.001,
    #                                results_root=RESULTS_ROOT,
    #                                save_results=True,
    #                                show_plots=True)
    
    run_dir = transferResults[-1]
    
    #models.test(transferResults, test_loader=test_loader,save_results=True, run_dir=run_dir)
    




if __name__ == "__main__":
    main()

