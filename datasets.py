import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import os
from imageio import imread
from PIL import Image
class CelebAMaskHQ(Dataset):
    def __init__(self, root, transform):
        self.img_path = root
        self.transform_img = transform
        self.train_dataset = []
        for i in [name for name in os.listdir(self.img_path) if os.path.isfile(os.path.join(self.img_path, name))]:
            img_path = os.path.join(self.img_path, str(i))
            self.train_dataset.append(img_path) 
        self.num_images = len(self.train_dataset)

    def __getitem__(self, index):

        dataset = self.train_dataset
        img_path = dataset[index]
        image = Image.fromarray(imread(img_path))
        return self.transform_img(image),self.transform_img(image)

    def __len__(self):
        """Return the number of images."""
        return self.num_images

class ImageDataset(object):
    def __init__(self, args):
        if args.dataset.lower() == 'cifar10':
            Dt = datasets.CIFAR10
            transform = transforms.Compose([
                transforms.Resize(args.img_size),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])
            args.n_classes = 10
        elif args.dataset.lower() == 'stl10':
            Dt = datasets.STL10
            transform = transforms.Compose([
                transforms.Resize(args.img_size),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])
        elif args.dataset.lower() == 'celeba':
            Dt = datasets.CelebA
            transform = transforms.Compose([
                transforms.Resize(args.img_size),
                transforms.CenterCrop(args.img_size),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])
        elif args.dataset.lower() == 'celebahq':
            Dt = CelebAMaskHQ
            transform = transforms.Compose([
                transforms.Resize(args.img_size),
                transforms.CenterCrop(args.img_size),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])
        elif args.dataset.lower() == 'lsun_church':
            Dt = datasets.LSUN
            transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.Resize(args.img_size),
                transforms.CenterCrop(args.img_size),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])
        else:
            raise NotImplementedError('Unknown dataset: {}'.format(args.dataset))

        if args.dataset.lower() == 'stl10':
            self.train = torch.utils.data.DataLoader(
                Dt(root=args.data_path, split='train+unlabeled', transform=transform, download=True),
                batch_size=args.dis_batch_size, shuffle=True,
                num_workers=args.num_workers, pin_memory=True)

            self.valid = torch.utils.data.DataLoader(
                Dt(root=args.data_path, split='test', transform=transform),
                batch_size=args.dis_batch_size, shuffle=False,
                num_workers=args.num_workers, pin_memory=True)

            self.test = self.valid
        elif args.dataset.lower() == 'celeba':
            self.train = torch.utils.data.DataLoader(
                Dt(root=args.data_path, split='train', transform=transform, download=True),
                batch_size=args.dis_batch_size, shuffle=True,
                num_workers=args.num_workers, pin_memory=True)

            self.valid = torch.utils.data.DataLoader(
                Dt(root=args.data_path, split='valid', transform=transform),
                batch_size=args.dis_batch_size, shuffle=False,
                num_workers=args.num_workers, pin_memory=True)

            self.test = self.valid
        elif args.dataset.lower() == 'celebahq':
            self.train = torch.utils.data.DataLoader(
                Dt(root=args.data_path+'/CelebAMask-HQ/CelebA-HQ-img/',transform=transform),
                batch_size=args.dis_batch_size, shuffle=True,
                num_workers=args.num_workers, pin_memory=True)

            self.valid = torch.utils.data.DataLoader(
                Dt(root=args.data_path, transform=transform),
                batch_size=args.dis_batch_size, shuffle=True,
                num_workers=args.num_workers, pin_memory=True)

            self.test = self.valid
        elif args.dataset.lower() == 'cifar10':
            self.train = torch.utils.data.DataLoader(
                Dt(root=args.data_path, train=True, transform=transform, download=True),
                batch_size=args.dis_batch_size, shuffle=True,
                num_workers=args.num_workers, pin_memory=True)

            self.valid = torch.utils.data.DataLoader(
                Dt(root=args.data_path, train=False, transform=transform),
                batch_size=args.dis_batch_size, shuffle=False,
                num_workers=args.num_workers, pin_memory=True)

            self.test = self.valid
        elif args.dataset.lower() == 'lsun_church':
            self.train = torch.utils.data.DataLoader(
                Dt(root=args.data_path, classes=['church_outdoor_train'], transform=transform),
                batch_size=args.dis_batch_size, shuffle=True,
                num_workers=args.num_workers, pin_memory=True)

            self.valid = torch.utils.data.DataLoader(
                Dt(root=args.data_path, classes=['church_outdoor_train'], transform=transform),
                batch_size=args.dis_batch_size, shuffle=True,
                num_workers=args.num_workers, pin_memory=True)

            self.test = self.valid
