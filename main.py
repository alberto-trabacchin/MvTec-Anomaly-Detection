import argparse
import models
import data
import torch
from matplotlib import pyplot as plt
from torchvision.transforms import transforms
from tqdm import tqdm
from pathlib import Path

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', type=str, required=True)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    return parser

def plot_init(num_classes, num_samples):
    fig, axs = plt.subplots(num_classes, num_samples * 2, figsize=(20, 2 * num_classes))
    for ax in axs.flatten():
        ax.axis('off')
    return fig, axs

def clear_axes(axs):
    for ax in axs.flatten():
        ax.clear()
        ax.axis('off')

def train_loop(args, model, criterion, optimizer, train_loader):
    num_classes = len(set(train_loader.dataset.targets_name))
    fig, axs = plot_init(num_classes, num_samples=5)
    for i in range(args.epochs):
        model.train()
        train_loss = 0.0
        pbar = tqdm(total=len(train_loader))
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
        print(f"Epoch {i+1}/{args.epochs}, Loss: {train_loss/len(train_loader):.4E}")

        # Evaluate model and plot results
        clear_axes(axs)
        fig, ax = evaluate_model(model, train_loader.dataset, fig, axs)
        plt.savefig(f"outputs/epoch_{i+1}.png", bbox_inches='tight', dpi=60)
        
    return model

def evaluate_model(model, dataset, fig, axs, num_samples=5):
    model.eval()
    class_names = list(set(dataset.targets_name))    
    with torch.inference_mode():
        for class_idx, class_name in enumerate(class_names):
            class_samples = [i for i, name in enumerate(dataset.targets_name) if name == class_name]
            selected_samples = class_samples[:num_samples]
            
            for i, sample_idx in enumerate(selected_samples):
                image, _ = dataset[sample_idx]
                image = image.to(args.device).unsqueeze(0)
                output = model(image)
                image = image.squeeze().permute(1, 2, 0).cpu().numpy()
                output = output.squeeze().permute(1, 2, 0).cpu().numpy()
                image = (image - image.min()) / (image.max() - image.min())
                output = (output - output.min()) / (output.max() - output.min())
                axs[class_idx, i * 2].imshow(image)
                axs[class_idx, i * 2 + 1].imshow(output)
                axs[class_idx, i * 2].axis('off')
                axs[class_idx, i * 2 + 1].axis('off')
                
                if i == 0:  # Label the first column with the class name
                    axs[class_idx, 0].set_ylabel(class_name, rotation=0, size='large', labelpad=60)
    return fig, axs

if __name__ == "__main__":
    Path("outputs").mkdir(exist_ok=True, parents=True)
    args = get_parser().parse_args()
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor()
    ])
    train_ds, test_ds = data.get_datasets(args.data_path, transform)
    train_loader, test_loader = data.get_loaders(train_ds, test_ds, args)
    model = models.ConvAutoencoder(embedding_dim=128)
    model.to(args.device)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    train_loop(args, model, criterion, optimizer, train_loader)
