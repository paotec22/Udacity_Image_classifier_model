import argparse
import torch
from torchvision import models
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from collections import OrderedDict

# Define the training function
def train_model(data_dir, save_dir, arch, learning_rate, hidden_units, epochs, gpu):
    # Load and preprocess data
    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    valid_test_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Load the datasets
    train_data = datasets.ImageFolder(data_dir + '/train', transform=train_transforms)
    valid_data = datasets.ImageFolder(data_dir + '/valid', transform=valid_test_transforms)
    
    trainloader = DataLoader(train_data, batch_size=64, shuffle=True)
    validloader = DataLoader(valid_data, batch_size=64)

    # Define model architecture (VGG or other)
    if arch == 'vgg16':
        model = models.vgg16(pretrained=True)
    elif arch == 'vgg13':
        model = models.vgg13(pretrained=True)
    else:
        raise ValueError("Unsupported architecture. Use 'vgg16' or 'vgg13'.")
    
    for param in model.parameters():
        param.requires_grad = False
    
    # Build a new classifier
    classifier = nn.Sequential(
        nn.Linear(model.classifier[0].in_features, hidden_units),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(hidden_units, 102),
        nn.LogSoftmax(dim=1)
    )
    
    model.classifier = classifier
    
    # Define the loss function and optimizer
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)

    # Train the model
    device = torch.device("cuda" if gpu and torch.cuda.is_available() else "cpu")
    model.to(device)

    for epoch in range(epochs):
        model.train()
        running_loss = 0
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        # Validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in validloader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        validation_accuracy = correct / total
        print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(trainloader)}, Validation Accuracy: {validation_accuracy}")

    # Save the checkpoint
    checkpoint = {
        'arch': arch,
        'state_dict': model.state_dict(),
        'class_to_idx': train_data.class_to_idx,
        'optimizer': optimizer.state_dict(),
        'epochs': epochs,
        'learning_rate': learning_rate
    }

    torch.save(checkpoint, save_dir + '/checkpoint.pth')
    print(f"Model saved to {save_dir}/checkpoint.pth")

# Parse command-line arguments
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a model on the flower dataset")
    parser.add_argument('data_dir', type=str, help='Directory with the flower data')
    parser.add_argument('--save_dir', type=str, default='.', help='Directory to save the checkpoint')
    parser.add_argument('--arch', type=str, choices=['vgg16', 'vgg13'], default='vgg16', help='Model architecture')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--hidden_units', type=int, default=512, help='Number of hidden units in the classifier')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs to train')
    parser.add_argument('--gpu', action='store_true', help='Use GPU for training')

    args = parser.parse_args()
    
    train_model(args.data_dir, args.save_dir, args.arch, args.learning_rate, args.hidden_units, args.epochs, args.gpu)