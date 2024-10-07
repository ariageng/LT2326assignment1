import torch
from sklearn.metrics import classification_report, accuracy_score
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import argparse
from training import OCRrecognizer, OCRDataset

def evaluate_model(model, test_loader, device):
    model.eval() 
    all_preds = []
    all_labels = []

    with torch.no_grad(): # disable gradient calculation for efficiency during the evaluation
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    print(f"Accuracy score: {accuracy_score(all_labels, all_preds)}")
    print(classification_report(all_labels, all_preds))


def main(model_path, test_data_dir, train_data_dir, batch_size):
    device = torch.device("cuda:0") # if torch.cuda.is_available() else "cpu"
    
    # Data loading and transformations
    transform = transforms.Compose([
        transforms.Grayscale(),  # Convert to grayscale (as your dataset is black and white)
        transforms.Resize((64, 64)),  # Resize to match your model input size (64x64)
        transforms.ToTensor(),  # Convert the image to a tensor
        transforms.Normalize((0.5,), (0.5,))  # Normalize the image to [-1, 1]
    ])
    
    test_dataset = OCRDataset(test_data_dir, transform=transform)  # Load the test dataset
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)  # No need to shuffle for testing
    
    # Load the training dataset to infer the number of classes
    train_dataset = OCRDataset(train_data_dir, transform=transform)
    num_classes = len(train_dataset.label_to_idx)  # Get the number of unique classes from the label mapping
    print(f"Number of classes inferred from training dataset: {num_classes}")
    
    # Load the model
    model = OCRrecognizer(num_classes).to(device)  # Change to match your model name
    model.load_state_dict(torch.load(model_path, map_location=device))  # Load the trained model parameters
    
    # Evaluate the model
    evaluate_model(model, test_loader, device)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the OCR model using the test split.")
    parser.add_argument('--model_path', type=str, required=True, help="Path to the saved model")
    parser.add_argument('--test_data_dir', type=str, required=True, help="Path to the test split txt file")
    parser.add_argument('--train_data_dir', type=str, required=True, help="Path to the train split txt file")
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size")
    # parser.add_argument('--num_classes', type=int, required=True, help="Number of classes (Thai/English characters)")
    
    args = parser.parse_args()
    main(args.model_path, args.test_data_dir, args.train_data_dir, args.batch_size)
