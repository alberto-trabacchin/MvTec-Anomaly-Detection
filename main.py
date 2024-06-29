import argparse
import models
import data
import torch
from matplotlib import pyplot as plt
from torchvision.transforms import v2
from tqdm import tqdm


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', type=str, required=True)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    return parser


def train_loop(args, model, criterion, optimizer, train_loader):
    train_loss = 0.0
    model.train()
    pbar = tqdm(total=len(train_loader))
    for i in range(args.epochs):
        for batch in train_loader:
            images, _ = batch
            images = images.to(args.device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, images)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            pbar.update(1)
        pbar.close()
        print(f"Epoch {i+1}/{args.epochs}, Loss: {train_loss/len(train_loader)}")
    


if __name__ == "__main__":
    args = get_parser().parse_args()
    transform = v2.Compose([
        v2.ToImage(), 
        v2.Resize((64, 64)),
        v2.ToDtype(torch.float32, scale=True)
    ])
    train_ds, test_ds = data.get_datasets(args.data_path, transform)
    train_loader, test_loader = data.get_loaders(train_ds, test_ds, args)
    model = models.ConvAutoencoder(embedding_dim=128)
    model.to(args.device)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    train_loop(args, model, criterion, optimizer, train_loader)