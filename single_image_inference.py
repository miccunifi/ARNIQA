import torch
from PIL import Image
from torchvision import transforms
from argparse import ArgumentParser

from utils.utils_data import center_corners_crop


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--img_path", type=str, required=True, help="Path to the image to be evaluated")
    parser.add_argument("--regressor_dataset", type=str, default="kadid10k", choices=["live", "csiq",
                                     "tid2013", "kadid10k", "flive", "spaq", "clive", "koniq10k"], help="Dataset used to train the regressor")
    args = parser.parse_args()

    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

    # Load the model
    model = torch.hub.load(repo_or_dir="miccunifi/ARNIQA", source="github", model="ARNIQA",
                           regressor_dataset=args.regressor_dataset)
    model.eval().to(device)

    # Define the normalization transform
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    # Load the full-scale image
    img = Image.open(args.img_path).convert("RGB")
    # Get the half-scale image
    img_ds = transforms.Resize((img.size[1] // 2, img.size[0] // 2))(img)

    # Get the center and corners crops
    img = center_corners_crop(img, crop_size=224)
    img_ds = center_corners_crop(img_ds, crop_size=224)

    # Preprocess the images
    img = [transforms.ToTensor()(crop) for crop in img]
    img = torch.stack(img, dim=0)
    img = normalize(img).to(device)
    img_ds = [transforms.ToTensor()(crop) for crop in img_ds]
    img_ds = torch.stack(img_ds, dim=0)
    img_ds = normalize(img_ds).to(device)

    # Compute the quality score
    with torch.no_grad(), torch.cuda.amp.autocast():
        score = model(img, img_ds, return_embedding=False, scale_score=True)
        # Compute the average score over the crops
        score = score.mean(0)

    print(f"Image {args.img_path} quality score: {score.item()}")
