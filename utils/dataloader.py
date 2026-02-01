import os
import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from utils.data_augmentation import cv_random_flip, randomCrop, randomRotation, randomPeper, colorEnhance
import cv2


class LabeledTrainDataset(Dataset):
    def __init__(self, l_image_root,  # file path: string, Original RGB Images
                       l_gt_root,  # file path: string, GT Annotations
                       l_txt_root,  # txt file path: string, Sampled Images
                       l_train_size,
                       rVFlip=False,
                       rCrop=False,
                       rRotate=False,
                       colorEnhance=False,
                       rPeper=False):

        self.l_train_size = l_train_size
        self.patch_size = 14

        # data augmentation
        self.rVFlip = rVFlip
        self.rCrop = rCrop
        self.rRotate = rRotate
        self.colorEnhance = colorEnhance
        self.rPeper = rPeper

        # load labeled data
        self.l_images, self.l_gts = [], []
        if l_txt_root is not None:
            with open(l_txt_root, 'r') as f:
                img_names = f.readlines()
            for img_name in img_names:
                img_name = img_name.strip()
                self.l_images.append(os.path.join(l_image_root, img_name + '.jpg'))
                self.l_gts.append(os.path.join(l_gt_root, img_name + '.png'))
        else:
            for img_name in os.listdir(l_image_root):
                if img_name.endswith('.jpg') or img_name.endswith('.png'):
                    self.l_images.append(os.path.join(l_image_root, img_name))
                    self.l_gts.append(os.path.join(l_gt_root, img_name[:-4] + '.png'))


        # sorted files
        self.l_images = sorted(self.l_images)
        self.l_gts = sorted(self.l_gts)

        assert len(self.l_images) == len(self.l_gts), '>>> Number of labeled images and gts do not match.'

        self.size = len(self.l_images)

        for i in range(self.size):
            assert self.l_images[i].split('/')[-1][:-4] == self.l_gts[i].split('/')[-1][:-4], '>>> File name mismatch.'

        # transforms
        self.img_transforms = transforms.Compose([
            transforms.Resize((self.l_train_size, self.l_train_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        self.gt_transforms = transforms.Compose([
            transforms.Resize((self.l_train_size, self.l_train_size)),
            transforms.ToTensor()])

        print(f'>>> Training/Validating with {self.size} samples')

    def __getitem__(self, index):
        l_image = self.rgb_loader(self.l_images[index])
        l_gt = self.binary_loader(self.l_gts[index])

        # data augmentation
        if self.rVFlip:
            l_image, l_gt = cv_random_flip([l_image, l_gt])
        if self.rCrop:
            l_image, l_gt = randomCrop([l_image, l_gt])
        if self.rRotate:
            l_image, l_gt = randomRotation([l_image, l_gt])
        if self.colorEnhance:
            l_image = colorEnhance(l_image)
        if self.rPeper:
            l_gt = randomPeper(l_gt)

        ori_img = self.gt_transforms(l_image)  # used for seg evaluate

        # labeled data processing
        l_image = self.img_transforms(l_image)
        # l_image_fold = self.image_fold(l_image)
        l_gt = self.gt_transforms(l_gt)

        return ori_img, l_image, l_gt

    def __len__(self):
        return self.size

    def rgb_loader(self, path):
        img = cv2.imread(path, flags=cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, code=cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img, mode='RGB')
        return img

    def binary_loader(self, path):
        img = cv2.imread(path, flags=cv2.IMREAD_GRAYSCALE)
        img = Image.fromarray(img, mode='L')
        return img


class UnlabeledTrainDataset(Dataset):
    def __init__(self, u_image_root,  # file path: string
                       u_gt_root,  # file path: string
                       sampled_txt,
                       u_train_size,
                       rVFlip=True,
                       rCrop=True,
                       rRotate=False,
                       colorEnhance=True,
                       rPeper=False):

        self.u_train_size = u_train_size

        self.patch_size = 14

        # data augmentation
        self.rVFlip = rVFlip
        self.rCrop = rCrop
        self.rRotate = rRotate
        self.colorEnhance = colorEnhance
        self.rPeper = rPeper

        # unlabeled data
        self.u_images, self.u_gts = [], []
        self.u_images.extend([os.path.join(u_image_root, f) for f in os.listdir(u_image_root) if f.endswith('.jpg') or f.endswith('.png')])
        self.u_gts.extend([os.path.join(u_gt_root, f) for f in os.listdir(u_gt_root) if f.endswith('.jpg') or f.endswith('.png')])

        self.sampled = []
        with open(sampled_txt, 'r') as f:
            img_names = f.readlines()
        for img_name in img_names:
            img_name = img_name.strip()
            self.sampled.append(img_name)

        # remove labeled data from unlabeled data
        self.u_images = [f for f in self.u_images if f.split('/')[-1][:-4] not in self.sampled]
        self.u_gts = [f for f in self.u_gts if f.split('/')[-1][:-4] not in self.sampled]
        
        # sorted files
        self.u_images = sorted(self.u_images)
        self.u_gts = sorted(self.u_gts)

        assert len(self.u_images) == len(self.u_gts), '>>> Number of unlabeled images and gts do not match.'

        self.size = len(self.u_images)

        for i in range(self.size):
            assert self.u_images[i].split('/')[-1][:-4] == self.u_gts[i].split('/')[-1][:-4], '>>> Unlabeled image and gt do not match.'

        # transforms
        self.img_transforms = transforms.Compose([
            transforms.Resize((self.u_train_size, self.u_train_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        self.gt_transforms = transforms.Compose([
            transforms.Resize((self.u_train_size, self.u_train_size)),
            transforms.ToTensor()])

        print(f'>>> Training/Validating with {self.size} samples')

    def __getitem__(self, index):
        u_image = self.rgb_loader(self.u_images[index])
        u_gt = self.binary_loader(self.u_gts[index])

        # data augmentation
        if self.rVFlip:
            u_image, u_gt = cv_random_flip([u_image, u_gt])
        if self.rCrop:
            u_image, u_gt = randomCrop([u_image, u_gt])
        if self.rRotate:
            u_image, u_gt = randomRotation([u_image, u_gt])
        if self.colorEnhance:
            u_image = colorEnhance(u_image)
        if self.rPeper:
            u_gt = randomPeper(u_gt)

        # unlabeled data processing
        u_image = self.img_transforms(u_image)
        u_gt = self.gt_transforms(u_gt)

        return u_image, u_gt

    def __len__(self):
        return self.size

    def rgb_loader(self, path):
        img = cv2.imread(path, flags=cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, code=cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img, mode='RGB')
        return img

    def binary_loader(self, path):
        img = cv2.imread(path, flags=cv2.IMREAD_GRAYSCALE)
        img = Image.fromarray(img, mode='L')
        return img


class TestDataset(Dataset):
    def __init__(self, image_root,  # file path: string
                       gt_root,  # file path: string
                       test_size):
        
        self.test_size = test_size

        self.images, self.gts = [], []
        self.images.extend([os.path.join(image_root, f) for f in os.listdir(image_root) if f.endswith('.jpg') or f.endswith('.png')])
        self.gts.extend([os.path.join(gt_root, f) for f in os.listdir(gt_root) if f.endswith('.jpg') or f.endswith('.png')])
            
        # sorted files
        self.images = sorted(self.images)
        self.gts = sorted(self.gts)

        assert len(self.images) == len(self.gts), '>>> Number of labeled images and gts do not match.'

        self.size = len(self.images)

        for i in range(self.size):
            assert self.images[i].split('/')[-1][:-4] == self.gts[i].split('/')[-1][:-4], '>>> File name mismatch.'


        # transforms
        self.img_transforms = transforms.Compose([
            transforms.Resize((self.test_size, self.test_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        self.gt_transforms = transforms.Compose([
            transforms.Resize((self.test_size, self.test_size)),
            transforms.ToTensor()])

        print(f'>>> Testing/Validating with {self.size} samples')
        
    def __getitem__(self, index):
        image = self.rgb_loader(self.images[index])
        gt = self.binary_loader(self.gts[index])

        ori_img = self.gt_transforms(image)
        image = self.img_transforms(image)
        ori_gt = transforms.PILToTensor()(gt)
        gt = self.gt_transforms(gt)

        name = self.images[index].split('/')[-1]
        if '\\' in name:
            name = name.split('\\')[-1]
        if name.endswith('.jpg'):
            name = name.split('.jpg')[0] + '.png'

        return ori_img, ori_gt, name, image, gt

    def __len__(self):
        return len(self.images)

    def rgb_loader(self, path):
        img = cv2.imread(path, flags=cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, code=cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img, mode='RGB')
        return img

    def binary_loader(self, path):
        img = cv2.imread(path, flags=cv2.IMREAD_GRAYSCALE)
        img = Image.fromarray(img, mode='L')
        return img
    