from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from PIL import Image
import torch
from torchvision.transforms import v2


class MvTecDS(Dataset):
    def __init__(self, data_path, train=True, transform=None):
        self.data_path = Path(data_path)
        self.classes = [d.name for d in self.data_path.iterdir() if d.is_dir()]
        self.train = train
        self.transform = transform
        self.data = []
        self.targets = []
        self.targets_name = []
        if self.train:
            for i, c in enumerate(self.classes):
                for img in (self.data_path / c / 'train'/'good').iterdir():
                    self.data.append(img)
                    self.targets.append(i)
                    self.targets_name.append(c)
        else:
            for i, c in enumerate(self.classes):
                for defect in (self.data_path / c / 'test').iterdir():
                    images = [img for img in defect.iterdir()]
                    self.data.extend(images)
                    self.targets.extend([i] * len(images))
                    self.targets_name.extend([c] * len(images))

    def __getitem__(self, index):
        image = Image.open(self.data[index]).convert('RGB')
        target = self.targets[index]
        if self.transform:
            image = self.transform(image)
        return image, target
    
    def __len__(self):
        return len(self.data)
            

def get_datasets(data_path, transform=None):
    train_ds = MvTecDS(data_path, train=True, transform=transform)
    test_ds = MvTecDS(data_path, train=False, transform=transform)
    return train_ds, test_ds


def get_loaders(train_ds, test_ds, args):
    train_loader = DataLoader(
        train_ds, 
        batch_size=args.batch_size, 
        shuffle=True,
        num_workers=args.num_workers
    )
    test_loader = DataLoader(
        test_ds, 
        batch_size=args.batch_size, 
        shuffle=False,
        num_workers=args.num_workers
    )
    return train_loader, test_loader