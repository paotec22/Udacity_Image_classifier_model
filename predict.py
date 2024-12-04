import argparse
import torch
from torchvision import models, transforms
from PIL import Image
import numpy as np
import json

# Define image processing function
def process_image(image_path):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model, returns a Numpy array '''
    
    # Load the image
    pil_image = Image.open(image_path)
    
    # Resize the image
    pil_image.thumbnail((256, 256))
    
    # Crop the center 224x224 region
    width, height = pil_image.size
    left = (width - 224) / 2
    top = (height - 224) / 2
    right = (width + 224) / 2
    bottom = (height + 224) / 2
    pil_image = pil_image.crop((left, top, right, bottom))
    
    # Convert image to numpy array and normalize
    np_image = np.array(pil_image) / 255.
    
    # Normalize the image using ImageNet means and std
    means = np.array([0.485, 0.456, 0.406])
    stds = np.array([0.229, 0.224, 0.225])
    np_image = (np_image - means) / stds
    
    # Convert image to PyTorch tensor and change dimensions
    tensor_image = torch.from_numpy(np_image).float()
    tensor_image = tensor_image.permute(2, 0, 1)  # Convert HWC to CHW
    
    return tensor_image.unsqueeze(0)

# Define the prediction function
def predict(image_path, checkpoint_path, top_k=5, category_names=None, gpu=False):
    ''' Predict the class of an image using a trained model '''
    
    # Load the checkpoint
    checkpoint = torch.load(checkpoint_path)
    
    # Load the model
    model = models.__dict__[checkpoint['arch']](pretrained=True)
    model.classifier = checkpoint['state_dict']['classifier']
    model.load_state_dict(checkpoint['state_dict'])
    
    model.eval()
    device = torch.device("cuda" if gpu and torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Process the image
    image = process_image(image_path).to(device)
    
    # Forward pass
    with torch.no_grad():
        outputs = model(image)
    
    # Get the top K predictions
    probs, indices = torch.topk(torch.exp(outputs), top_k)
    probs = probs.cpu().numpy()[0]
    indices = indices.cpu().numpy()[0]
    
    # Get class labels
    class_to_idx = checkpoint['class_to_idx']
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    classes = [idx_to_class[i] for i in indices]
    
    # Optionally map class indices to human-readable labels
    if category_names:
        with open(category_names, 'r') as f:
            cat_to_name = json.load(f)
        classes = [cat_to_name[str(cls)] for cls in classes]
    
    return probs, classes

# Parse command-line arguments
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Predict the class of an image using a trained model")
    parser.add_argument('image_path', type=str, help='Path to the input image')
    parser.add_argument('checkpoint', type=str, help='Path to the model checkpoint')
    parser.add_argument('--top_k', type=int, default=5, help='Return the top K most likely classes')
    parser.add_argument('--category_names', type=str, help='Path to the category names file')
    parser.add_argument('--gpu', action='store_true', help='Use GPU for inference')

    args = parser.parse_args()

    probs, classes = predict(args.image_path, args.checkpoint, args.top_k, args.category_names, args.gpu)
    
    print(f"Top {args.top_k} Classes: {classes}")
    print(f"Probabilities: {probs}")