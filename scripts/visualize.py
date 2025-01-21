import os
import torch
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from transformers import SegformerForSemanticSegmentation, SegformerFeatureExtractor
from PIL import Image
import argparse
import cv2

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

def visualize_predictions(model, feature_extractor, dataset, idx, output_path=None):
    model.eval()
    with torch.no_grad():
        sample = dataset[idx]
        inputs = {k: v.unsqueeze(0).to(model.device) for k, v in sample.items() if k in feature_extractor.model_input_names}
        outputs = model(**inputs)
        logits = outputs.logits
        upsampled_logits = torch.nn.functional.interpolate(logits, size=sample['pixel_values'].shape[-2:], mode='bilinear', align_corners=False)
        predicted_mask = upsampled_logits.argmax(dim=1).squeeze().cpu().numpy()

    image = sample['pixel_values'].permute(1, 2, 0).cpu().numpy()
    mask = sample['labels'].cpu().numpy()

    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[0].imshow(image)
    ax[0].set_title('Image')
    ax[1].imshow(mask, cmap='gray')
    ax[1].set_title('Ground Truth Mask')
    ax[2].imshow(predicted_mask, cmap='gray')
    ax[2].set_title('Predicted Mask')
    plt.show()

    # Save the result if output_path is provided
    if output_path:
        blended = cv2.addWeighted(image, 0.7, predicted_mask, 0.3, 0)
        cv2.imwrite(output_path, blended)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize segmentation mask on the original image.")
    parser.add_argument("image_path", type=str, help="Path to the original image.")
    parser.add_argument("mask_path", type=str, help="Path to the segmentation mask.")
    parser.add_argument("--output_path", type=str, default=None, help="Path to save the visualized image.")

    args = parser.parse_args()

    # Load the feature extractor and model
    feature_extractor = SegformerFeatureExtractor.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")
    model = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512", num_labels=2)
    model.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

    # Create a dummy dataset with the provided image and mask paths
    class DummyDataset(Dataset):
        def __init__(self, image_path, mask_path, feature_extractor):
            self.image_path = image_path
            self.mask_path = mask_path
            self.feature_extractor = feature_extractor

        def __len__(self):
            return 1

        def __getitem__(self, idx):
            image = Image.open(self.image_path).convert("RGB")
            mask = Image.open(self.mask_path)
            encoded_inputs = self.feature_extractor(images=image, masks=mask, return_tensors="pt")
            for k, v in encoded_inputs.items():
                encoded_inputs[k].squeeze_()
            return encoded_inputs

    dataset = DummyDataset(args.image_path, args.mask_path, feature_extractor)

    # Visualize predictions
    visualize_predictions(model, feature_extractor, dataset, idx=0, output_path=args.output_path)