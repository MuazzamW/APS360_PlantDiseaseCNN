import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

num_classes = 27
IMG_SIZE = 224


#utilities
def plot_training_curve(train_loss, train_accuracy, val_loss, val_accuracy):

    num_epochs = len(train_loss)
    epochs = range(1, num_epochs + 1)

    # Accuracy plot
    plt.figure()

    plt.plot(
        epochs,
        train_accuracy,
        label="Training Accuracy"
    )

    plt.plot(
        epochs,
        val_accuracy,
        label="Validation Accuracy"
    )

    plt.title("Training vs Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid()
    plt.show()

    # Loss plot
    plt.figure()

    plt.plot(
        epochs,
        train_loss,
        label="Training Loss"
    )

    plt.plot(
        epochs,
        val_loss,
        label="Validation Loss"
    )

    plt.title("Training vs Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid()
    plt.show()


class BaselineANN(nn.Module):
    def __init__(self, num_classes=27, img_size=IMG_SIZE):
        super(BaselineANN, self).__init__()

        self.name = "baseline_ann"

        input_size = 3 * img_size * img_size

        self.fc1 = nn.Linear(input_size, 512)
        self.dropout1 = nn.Dropout(0.3)

        self.fc2 = nn.Linear(512, 128)
        self.dropout2 = nn.Dropout(0.3)

        self.fc3 = nn.Linear(128, num_classes)

    def forward(self, x):
        # x shape: [batch_size, 3, 224, 224]

        x = x.view(x.size(0), -1)
        # x shape: [batch_size, 150528]

        x = F.relu(self.fc1(x))
        x = self.dropout1(x)

        x = F.relu(self.fc2(x))
        x = self.dropout2(x)

        x = self.fc3(x)
        # x shape: [batch_size, num_classes]

        return x
    
class PrimaryCNN(nn.Module):
    def __init__(
        self,
        num_classes=27,
        img_size=224
    ):
        super(PrimaryCNN, self).__init__()

        self.name = "primary_cnn"

        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=32,
            kernel_size=3,
            padding=1
        )

        self.conv2 = nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=3,
            padding=1
        )

        self.conv3 = nn.Conv2d(
            in_channels=64,
            out_channels=128,
            kernel_size=3,
            padding=1
        )

        self.pool = nn.MaxPool2d(2, 2)

        self.dropout = nn.Dropout(0.3)

        flattened_size = self._get_flattened_size(img_size)

        self.fc1 = nn.Linear(flattened_size, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def _get_flattened_size(self, img_size):
        with torch.no_grad():
            dummy = torch.zeros(1, 3, img_size, img_size)

            x = self.pool(F.relu(self.conv1(dummy)))
            x = self.pool(F.relu(self.conv2(x)))
            x = self.pool(F.relu(self.conv3(x)))

            x = x.view(x.size(0), -1)

            return x.shape[1]

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))

        x = x.view(x.size(0), -1)

        x = F.relu(self.fc1(x))
        x = self.dropout(x)

        x = self.fc2(x)

        return x


def train_one_epoch(model, train_loader, criterion, optimizer, device):
    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

        predicted = outputs.argmax(dim=1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total

    return epoch_loss, epoch_acc


def evaluate(model, data_loader, criterion, device):
    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)

            predicted = outputs.argmax(dim=1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total

    return epoch_loss, epoch_acc

def train_baseline(train_loader, val_loader):
    NUM_EPOCHS = 50

    train_loss_array = np.zeros(NUM_EPOCHS)
    train_accuracy_array = np.zeros(NUM_EPOCHS)

    val_loss_array = np.zeros(NUM_EPOCHS)
    val_accuracy_array = np.zeros(NUM_EPOCHS)


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    num_classes = 27

    model = BaselineANN(
        num_classes=num_classes,
        img_size=IMG_SIZE
    ).to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(
        model.parameters(),
        lr=0.001
    )

    for epoch in range(NUM_EPOCHS):
        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device
        )

        val_loss, val_acc = evaluate(
            model,
            val_loader,
            criterion,
            device
        )

        print(
            f"Epoch [{epoch + 1}/{NUM_EPOCHS}] "
            f"Train Loss: {train_loss:.4f}, "
            f"Train Acc: {train_acc:.4f}, "
            f"Val Loss: {val_loss:.4f}, "
            f"Val Acc: {val_acc:.4f}"
        )

        train_loss_array[epoch] = train_loss
        train_accuracy_array[epoch] = train_acc
        
        val_loss_array[epoch] = val_loss
        val_accuracy_array[epoch] = val_acc
    
    plot_training_curve(train_loss_array,train_accuracy_array,val_loss_array,val_accuracy_array)


def train_primary(train_loader, val_loader):
    NUM_EPOCHS = 10

    train_loss_array = np.zeros(NUM_EPOCHS)
    train_accuracy_array = np.zeros(NUM_EPOCHS)

    val_loss_array = np.zeros(NUM_EPOCHS)
    val_accuracy_array = np.zeros(NUM_EPOCHS)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    num_classes = 27

    model = PrimaryCNN(
        num_classes=num_classes,
        img_size=IMG_SIZE
    ).to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(
        model.parameters(),
        lr=0.001
    )

    for epoch in range(NUM_EPOCHS):
        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device
        )

        val_loss, val_acc = evaluate(
            model,
            val_loader,
            criterion,
            device
        )

        print(
            f"Epoch [{epoch + 1}/{NUM_EPOCHS}] "
            f"Train Loss: {train_loss:.4f}, "
            f"Train Acc: {train_acc:.4f}, "
            f"Val Loss: {val_loss:.4f}, "
            f"Val Acc: {val_acc:.4f}"
        )

        train_loss_array[epoch] = train_loss
        train_accuracy_array[epoch] = train_acc

        val_loss_array[epoch] = val_loss
        val_accuracy_array[epoch] = val_acc

    plot_training_curve(
        train_loss_array,
        train_accuracy_array,
        val_loss_array,
        val_accuracy_array
    )

    return model, train_loss_array, train_accuracy_array, val_loss_array, val_accuracy_array
        