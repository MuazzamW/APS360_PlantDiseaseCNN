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
        transforms.RandomResizedCrop(224, scale=(0.6, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(25),
        transforms.ColorJitter(
            brightness=0.3,
            contrast=0.3,
            saturation=0.3,
            hue=0.05
        ),
        transforms.GaussianBlur(kernel_size=3),
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    baselineANN = models.BaselineANN(NUM_CLASSES, IMG_SIZE)
    primaryCNN = models.PrimaryCNN(NUM_CLASSES, IMG_SIZE)
    transfer_model = models.TransferResNet18(
            num_classes=NUM_CLASSES,
            freeze_backbone=True
    )

    # transferResults = models.train(
    #     model=transfer_model,
    #     train_loader=train_loader,
    #     val_loader=val_loader,
    #     num_epochs=20,
    #     learning_rate=0.001,
    #     results_root=RESULTS_ROOT,
    #     save_results=True,
    #     show_plots=True
    # )

    # baselineResults = models.train(baselineANN, 
    #                                train_loader=train_loader, 
    #                                val_loader=val_loader, 
    #                                num_epochs=30,
    #                                learning_rate=0.001,
    #                                results_root=RESULTS_ROOT,
    #                                save_results=True,
    #                                show_plots=True)

    # primaryCNN.load_state_dict(
    #     torch.load("results/primary_cnn_v2_20260805_085409_epochs20_lr0.001_batch32/model_state_dict.pt",
    #                map_location=device)
    # )

    transfer_model.load_state_dict(
        torch.load("results/transfer_resnet18_20260805_210956_epochs20_lr0.001_batch32/model_state_dict.pt")
    )

    # primaryCNN = primaryCNN.to(device)
    # primaryCNN.eval()

    transfer_model = transfer_model.to(device)
    transfer_model.eval()

    run_dir = Path("results/transfer_resnet18_20260805_210956_epochs20_lr0.001_batch32/")
    
    #models.test(primaryCNN, test_loader=test_loader, save_results=True, run_dir=run_dir)
    
    models.test(transfer_model, test_loader=test_loader,save_results=True, run_dir=run_dir)
    




if __name__ == "__main__":
    main()

