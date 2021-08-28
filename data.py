import os
from glob import glob
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.transforms import transforms as T
from torchvision.transforms.functional import to_pil_image
from PIL import Image


class TrashDataset(Dataset):
    def __init__(self, dataset_dir, transform=None):
        # Formatting of dataset_dir: trashnet-data/{class}/{image-name}.jpg
        self.img_filenames = glob(os.path.join(dataset_dir, "*/*.jpg"))

        # Str to number label conversion of data
        self.label_dict = {
            "cardboard": 0,
            "glass": 1,
            "metal": 2,
            "paper": 3,
            "plastic": 4,
            "trash": 5,
        }

        self.transform = transform

    def __len__(self):
        return len(self.img_filenames)

    def __getitem__(self, idx):
        filename = self.img_filenames[idx]
        image = Image.open(filename)
        label = self.label_dict[os.path.basename(os.path.dirname(filename))]

        if self.transform:
            image = self.transform(image)

        return image, label


def get_img_transforms(input_size):
    img_transforms = T.Compose(
        [
            T.Resize((input_size, input_size)),
            T.RandomApply([T.RandomResizedCrop((input_size, input_size))], p=0.5),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.5),
            T.ToTensor(),
        ]
    )
    return img_transforms


def get_dataloaders(args):
    # Define Transforms
    img_transforms = get_img_transforms(args.input_size)

    # Define Dataset
    dataset = TrashDataset(args.data, transform=img_transforms)

    # split dataset
    train_count = int(len(dataset) * 0.9)
    val_count = len(dataset) - train_count
    train_set, val_set = random_split(dataset, [train_count, val_count])

    train_dataloader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_dataloader = DataLoader(val_set, batch_size=args.batch_size, shuffle=True)

    return train_dataloader, val_dataloader
