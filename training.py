import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import argparse

# define the OCR dataset
class OCRDataset(Dataset):
    def __init__(self, file_list, transform=None):
        self.file_list = file_list
        self.transform = transform

        self.file_paths = [] # a list of all image file paths
        self.file_labels = [] # a list of all image label paths
        with open(file_list, 'r') as f:
            for line in f:
                path = line.strip()
                # get the label aka the number in the file name
                label = self.extract_label(path)
                self.file_paths.append(path)
                self.file_labels.append(label)

        # get unique labels and create a label-idx and a idx-label mapping
        self.label_to_idx = {label: idx for idx, label in enumerate(sorted(set(self.file_labels)))}
        self.idx_to_label = {idx: label for label, idx in self.label_to_idx.items()}
        print("length of label to idx:",len(self.label_to_idx),"length of idx to label:", len(self.idx_to_label))


    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        label = self.file_labels[idx]
        image = Image.open(file_path).convert('1') # Convert to 1-bit black and white

        if self.transform:
            image = self.transform(image)

        return image, self.label_to_idx[label]

        
    def extract_label(self, path):
        # e.g. ../../../../../scratch/lt2326-2926-h24/ThaiOCR/ThaiOCR-TrainigSet/Thai/269/200/normal/KTCB211_200_10_12_269.bmp
        path_parts = path.split(os.sep)
        label = path_parts[-4]
        return label

# define a CNN model
class OCRrecognizer(nn.Module):
    def __init__(self, num_classes):
        super(OCRrecognizer, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 32 * 32, 128)  # 64 channels with 32x32 image after pooling
        self.fc2 = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        print(f"Input to model: {x.shape}")
        x = self.relu(self.conv1(x))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 64 * 32 * 32)  # To flatten the image
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Training
def train_model(model, train_loader, valid_loader, criterion, optimizer, num_epochs, device):
    model.to(device)
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            # print(f"Batch images shape: {images.shape}, Batch labels shape: {labels.shape}")
            images, labels = images.to(device), labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(images)
            # print(f'Outputs shape: {outputs.shape}, Labels shape: {labels.shape}')

            # Check for size mismatch
            # if outputs.size(0) != labels.size(0):
            #     print(f'Output size: {outputs.size(0)}, Label size: {labels.size(0)}')
            #     continue  # Skip this iteration

            loss = criterion(outputs, labels)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}')

        # validation
        if valid_loader:
            validate_model(model, valid_loader, criterion, device)

# Validation
def validate_model(model, valid_loader, criterion, device):
    model.eval()
    valid_loss = 0.0
    with torch.no_grad():
        for images, labels in valid_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            valid_loss += loss.item()

    print(f'Validation Loss: {valid_loss/len(valid_loader):.4f}')



def main():
    parser = argparse.ArgumentParser(description="Train OCR Model on Thai/English Characters")
    parser.add_argument('--train-file', type=str, required=True, help="Path of the train.txt file")
    parser.add_argument('--valid-file', type=str, help="Path of the valid.txt file (optional)")
    parser.add_argument('--batch-size', type=int, default=32, help="Batch size for training")
    parser.add_argument('--epochs', type=int, default=10, help="Number of epochs")
    parser.add_argument('--lr', type=float, default=0.001, help="Learning rate")
    parser.add_argument('--output-model', type=str, required=True, help="Path to save the trained model")
    parser.add_argument('--device', type=str, default="cuda", help="Device to use for training (cuda or cpu)")

    args = parser.parse_args()

    # Data transforms: resizing and normalization
    transform = transforms.Compose([
        transforms.Resize((64, 64)),  # Resize images to 64x64
        transforms.ToTensor(),  # Convert image to tensor
        transforms.Normalize((0.5,), (0.5,))  # Normalize to [-1, 1]
    ])

    # Load datasets
    train_dataset = OCRDataset(file_list=args.train_file, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    valid_loader = None
    if args.valid_file:
        valid_dataset = OCRDataset(file_list=args.valid_file, transform=transform)
        valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False)

    # Initialize the model, loss function, and optimizer
    num_classes = len(train_dataset.idx_to_label)  # Number of unique characters (labels)
    model = OCRrecognizer(num_classes=num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Train the model
    train_model(model, train_loader, valid_loader, criterion, optimizer, num_epochs=args.epochs, device=args.device)

    # Save the trained model
    torch.save(model.state_dict(), args.output_model)
    print(f"Model saved to {args.output_model}")


if __name__ == '__main__':
    main()