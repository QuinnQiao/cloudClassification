import torch
from torchvision import transforms as T
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import csv, os
from PIL import Image

class CloudDataSL(Dataset):
    '''
    Single-label, copy image if multi-label

    '''
    def __init__(self, if_label, imgs_folder, csv_file, transform=None):
        super(CloudDataSL, self).__init__()
        self.transform = transform
        self.imgs = []
        if if_label:
            self.labels = []
            self._get_imgs_labels(imgs_folder, csv_file)
        else:
            self.labels = None
            self._get_imgs(imgs_folder, csv_file)

    def _get_imgs_labels(self, imgs_folder, csv_file):
        with open(csv_file, mode='r', encoding='utf-8') as f:
            lines = csv.reader(f)
            tot = 0
            for line in lines:
                tot += 1
                if tot == 1:            
                    continue
                name = line[0]
                label = line[1].split(';')
                for _l in label:
                    self.imgs.append(os.path.join(imgs_folder, name))
                    self.labels.append(int(_l)-1)
                tot += len(label)-1
        # check correctness          
        assert tot == len(self.labels)+1

    def _get_imgs(self, imgs_folder, csv_file):
        with open(csv_file, mode='r', encoding='utf-8') as f:
            lines = csv.reader(f)
            tot = 0
            for line in lines:
                tot += 1
                if tot == 1:
                    continue
                self.imgs.append(os.path.join(imgs_folder, line[0]))
        # check correctness          
        assert tot == len(self.imgs)+1
    
    def _crop(self, img):
        # drop the lower part if the image is "vertical"
        w, h = img.size
        if float(h) > w * 1.5:
            img = img.crop((0, 0, w, int(h * 0.666667))) # left, upper, right, lower
        return img

    def __getitem__(self, index):
        img = self._crop(Image.open(self.imgs[index]).convert('RGB'))
        if self.transform is not None:
            img = self.transform(img)
        if self.labels is not None:
            return img, self.labels[index]
        else:
            return img

    def __len__(self):
        return len(self.imgs)

def get_transform(resize_size, crop_size, is_train):
    to_tensor = [T.ToTensor(),
                 T.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))]
    if is_train:
        # resize_size is not used
        transform = [T.RandomRotation(degrees=20), 
                     T.RandomHorizontalFlip(0.5),
                     T.RandomResizedCrop(size=crop_size, scale=(0.5, 1.0))
                    ] + to_tensor
    else:
        transform = [T.Resize(resize_size),
                     T.FiveCrop(crop_size),
                     T.Lambda(lambda crops: torch.stack([T.Compose(to_tensor)(crop) for crop in crops]))
                    ]
    return T.Compose(transform)

def get_loader(if_label, imgs_folder, csv_file, resize_size, 
                crop_size, is_train, batch_size, num_workers):
    transform = get_transform(resize_size, crop_size, is_train)
    dataset = CloudDataSL(if_label, imgs_folder, csv_file, transform)
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers,
                        shuffle=is_train, pin_memory=True, drop_last=is_train)
    if if_label:
        return loader, len(dataset)
    else:
        return loader, dataset.imgs
