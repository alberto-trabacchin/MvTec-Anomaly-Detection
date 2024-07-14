import argparse
import models
import data
import torch
from matplotlib import pyplot as plt
from torchvision.transforms import transforms
from tqdm import tqdm
from pathlib import Path
import os

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', type=str, required=True)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--n-save', type=int, default=5)
    parser.add_argument('--emb-dim', type=int, default=1024)
    return parser

def save_images(output_dir, epoch, class_name, sample_idx, image, output):
    class_dir = os.path.join(output_dir, class_name)
    os.makedirs(class_dir, exist_ok=True)
    epoch_dir = os.path.join(class_dir, str(epoch))
    os.makedirs(epoch_dir, exist_ok=True)
    if epoch == 1:
        original_dir = os.path.join(class_dir, "original")
        os.makedirs(original_dir, exist_ok=True)
        original_path = os.path.join(original_dir, f"{sample_idx}.png")
        plt.imsave(original_path, image)
    output_path = os.path.join(epoch_dir, f"{sample_idx}.png")
    plt.imsave(output_path, output)

def train_loop(args, model, criterion, optimizer, train_loader, test_loader):
    output_dir = "outputs"
    min_loss = float('inf')
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        pbar = tqdm(total=len(train_loader) + len(test_loader), desc=f"Epoch {epoch+1 :4d}/{args.epochs}")

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

        train_loss /= len(train_loader)
        eval_loss = evaluate_model(args, model, test_loader, criterion, pbar)
        
        pbar.set_postfix({"train/loss": f"{train_loss:.2E}", "eval/loss": f" {eval_loss:.2E}"})
        pbar.set_description("Saving results")
        save_train_res(args, model, train_loader.dataset, epoch, output_dir, args.n_save)
        pbar.set_description(f"Epoch {epoch+1 :4d}/{args.epochs}")
        pbar.close()

        if train_loss < min_loss:
            min_loss = train_loss
            Path("checkpoints").mkdir(exist_ok=True)
            torch.save(model.state_dict(), "checkpoints/model.pth")
    return model

def evaluate_model(args, model, dataloader, criterion, pbar):
    model.eval()
    eval_loss = 0.0
    pbar.set_description("Evaluating")
    with torch.inference_mode():
        for batch in dataloader:
            images, _ = batch
            images = images.to(args.device)
            outputs = model(images)
            loss = criterion(outputs, images)
            eval_loss += loss.item()
            pbar.update(1)
    eval_loss /= len(dataloader)
    return eval_loss

def save_train_res(args, model, dataset, epoch, output_dir, num_samples):
    model.eval()
    class_names = list(set(dataset.targets_name))    
    with torch.inference_mode():
        for class_idx, class_name in enumerate(class_names):
            class_samples = [i for i, name in enumerate(dataset.targets_name) if name == class_name]
            selected_samples = class_samples[:num_samples]
            
            for sample_idx in selected_samples:
                image, _ = dataset[sample_idx]
                image = image.to(args.device).unsqueeze(0)
                output = model(image)
                image = image.squeeze().permute(1, 2, 0).cpu().numpy()
                output = output.squeeze().permute(1, 2, 0).cpu().numpy()
                image = (image - image.min()) / (image.max() - image.min())
                output = (output - output.min()) / (output.max() - output.min())
                
                save_images(output_dir, epoch + 1, class_name, sample_idx, image, output)

if __name__ == "__main__":
    Path("outputs").mkdir(exist_ok=True, parents=True)
    args = get_parser().parse_args()
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor()
    ])
    train_ds, test_ds = data.get_datasets(args.data_path, transform)
    train_loader, test_loader = data.get_loaders(train_ds, test_ds, args)
    model = models.ConvAutoencoder(embedding_dim=args.emb_dim)
    model.to(args.device)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    train_loop(args, model, criterion, optimizer, train_loader, test_loader)
