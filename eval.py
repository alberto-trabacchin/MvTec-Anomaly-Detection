from pathlib import Path
import torch
from torchvision.transforms import transforms
from models import ConvAutoencoder
from data import get_datasets, get_loaders
import argparse
from tqdm import tqdm


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', type=str, required=True)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    return parser


def evaluate_model(args, model, dataloader, criterion):
    model.eval()
    eval_loss = 0.0
    with torch.inference_mode():
        for batch in tqdm(dataloader):
            images, _ = batch
            images = images.to(args.device)
            outputs = model(images)
            loss = criterion(outputs, images)
            eval_loss += loss.item()
    print(f"Loss: {eval_loss}")
        


if __name__ == "__main__":
    args = get_parser().parse_args()
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor()
    ])
    train_ds, test_ds = get_datasets(args.data_path, transform)
    train_loader, test_loader = get_loaders(train_ds, test_ds, args)
    model = ConvAutoencoder(embedding_dim=1024)
    model.load_state_dict(torch.load("checkpoints/model.pth"))
    model.to(args.device)
    criterion = torch.nn.MSELoss()
    evaluate_model(args, model, test_loader, criterion)
