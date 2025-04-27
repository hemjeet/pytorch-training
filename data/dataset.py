from torchvision import datasets, transforms, models
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import os


class Dataset:
    def __init__(self, image_paths):
        self.image_paths = image_paths
        self.transform = self._get_transforms()

    def _get_transforms(self):
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize(
                mean = [0.485, 0.456, 0.406],
                std = [0.229, 0.224, 0.225]
            )

        ])
        return transform

    def create_dataloader(self):
        train_data = ImageFolder(root=os.path.join(self.image_paths, 'train'), transform = self.transform)
        test_data = ImageFolder(root=os.path.join(self.image_paths, 'test'), transform = self.transform)
        print('Lebth of train data:', len(train_data))
        print('Lebth of test data:', len(test_data))
        train_loader = DataLoader(train_data, batch_size = 8, shuffle = True)
        test_loader = DataLoader(test_data, batch_size = 4, shuffle = False)
        return train_loader, test_loader, train_data.class_to_idx
