import os
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import SegformerForSemanticSegmentation, SegformerFeatureExtractor
from torch.optim import AdamW
from PIL import Image
import argparse

class SemanticSegmentationDataset(Dataset):
    def __init__(self, root_dir, feature_extractor):
        self.root_dir = root_dir
        self.feature_extractor = feature_extractor
        self.images = sorted([f for f in os.listdir(os.path.join(root_dir, 'images')) if f.endswith('.jpg')])
        self.masks = sorted([f for f in os.listdir(os.path.join(root_dir, 'masks')) if f.endswith('.png')])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, 'images', self.images[idx])
        mask_path = os.path.join(self.root_dir, 'masks', self.masks[idx])
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)
        encoded_inputs = self.feature_extractor(images=image, masks=mask, return_tensors="pt")
        for k, v in encoded_inputs.items():
            encoded_inputs[k].squeeze_()
        return encoded_inputs

def train(args):
    # Initialize the feature extractor
    feature_extractor = SegformerFeatureExtractor.from_pretrained(args.model_name)

    # Create the dataset and dataloader
    train_dataset = SemanticSegmentationDataset(root_dir=args.dataset, feature_extractor=feature_extractor)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    # Initialize the model
    model = SegformerForSemanticSegmentation.from_pretrained(args.model_name, num_labels=args.num_classes)
    model.to(args.device)

    # Optimizer
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    
    # Training loop
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            outputs = model(pixel_values=batch['pixel_values'].to(args.device), labels=batch['labels'].to(args.device))
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            print(f"Epoch {epoch + 1}/{args.epochs}, Loss: {loss.item()}")
        
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch + 1}/{args.epochs} completed. Average Loss: {avg_loss:.4f}")
    
    # Save the model
    model.save_pretrained(args.output_dir)
    feature_extractor.save_pretrained(args.output_dir)

def parse_args():
    parser = argparse.ArgumentParser(description="Train Segformer model for semantic segmentation.")
    parser.add_argument("--dataset", type=str, required=True, help="Path to the dataset directory.")
    parser.add_argument("--model_name", type=str, default="nvidia/segformer-b0-finetuned-ade-512-512", help="Model name or path.")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for training.")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs to train.")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate.")
    parser.add_argument("--num_classes", type=int, default=2, help="Number of classes.")
    parser.add_argument("--output_dir", type=str, default="./output", help="Directory to save the trained model.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use for training.")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    train(args)