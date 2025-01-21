import argparse
import torch
from torch.utils.data import DataLoader
from transformers import SegformerForSemanticSegmentation, SegformerFeatureExtractor
from datasets import load_dataset
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Segformer model on a dataset")
    parser.add_argument("--model_name_or_path", type=str, required=True, help="Path to the pretrained model")
    parser.add_argument("--dataset_name", type=str, required=True, help="Name of the dataset to evaluate on")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for evaluation")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to run evaluation on")
    return parser.parse_args()

def load_data(dataset_name, feature_extractor, batch_size):
    dataset = load_dataset(dataset_name)
    
    def preprocess(example):
        inputs = feature_extractor(example['image'], return_tensors="pt")
        inputs['labels'] = torch.tensor(example['label'])
        return inputs
    
    dataset = dataset.map(preprocess, batched=True)
    dataloader = DataLoader(dataset['test'], batch_size=batch_size)
    return dataloader

def evaluate(model, dataloader, device):
    model.eval()
    model.to(device)
    total_loss = 0
    criterion = torch.nn.CrossEntropyLoss()

    with torch.no_grad():
        for batch in dataloader:
            inputs = batch['pixel_values'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(pixel_values=inputs, labels=labels)
            loss = criterion(outputs.logits, labels)
            total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    print(f"Average Loss: {avg_loss:.4f}")

def main():
    args = parse_args()
    
    # Load the feature extractor and model
    feature_extractor = SegformerFeatureExtractor.from_pretrained(args.model_name_or_path)
    model = SegformerForSemanticSegmentation.from_pretrained(args.model_name_or_path)
    
    # Load the data
    dataloader = load_data(args.dataset_name, feature_extractor, args.batch_size)
    
    # Evaluate the model
    evaluate(model, dataloader, args.device)

if __name__ == "__main__":
    main()