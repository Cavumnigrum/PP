import torch
import cv2
from albumentations.pytorch import ToTensorV2
from torchvision.utils import draw_segmentation_masks
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp
import albumentations as A
from EyeDataset import EyeDataset, DatasetPart
from sklearn.model_selection import train_test_split
import numpy as np
import random

if __name__ == "__main__":
    size = 1024

    device = "cuda" if torch.cuda.is_available() else 'cpu'

    size = 1024
    train_list = [A.LongestMaxSize(size, interpolation=cv2.INTER_CUBIC),
                  A.PadIfNeeded(size, size),
                  ToTensorV2(transpose_mask=True)]
    eval_list = [A.LongestMaxSize(size, interpolation=cv2.INTER_CUBIC),
                A.PadIfNeeded(size, size),
                ToTensorV2(transpose_mask=True)]

    transforms = {'train': A.Compose(train_list), 'test': A.Compose(eval_list)}

    test_dataset = EyeDataset("./Dataset/eye_test", transform=transforms["test"])
    dataset = EyeDataset("./Dataset/eye_train")

    train_indices, test_indices = train_test_split(range(len(dataset)), test_size=0.25)
    train_dataset = DatasetPart(dataset, train_indices, transform=transforms['train'])
    valid_dataset = DatasetPart(dataset, test_indices, transform=transforms['test'])

    model = smp.Unet('resnet50', activation='logsoftmax', classes=2).cuda()

    # default augmentation weights
    # model.load_state_dict(torch.load("./model/weights"))

    # excessive augmentation weights
    model.load_state_dict(torch.load("./model/man"))

    # additional augmentation weights
    # model.load_state_dict(torch.load("./model/weights_additional_aug"))
    for _ in range(5):
        fig, axs = plt.subplots(2, 4, figsize=(16, 8))
        fig.suptitle(f"model predictions{' '*105} tested photos", fontsize=14)

        for i in range(4):
            sample = test_dataset[random.randint(0,len(test_dataset)-1)]
            image = sample["image"].to(device)
            true_mask = sample['mask'].to(device)
            prediction = model.eval()(image.unsqueeze(dim=0))

            img = (image.cpu() * 255).type(torch.uint8)
            pred_mask = (torch.exp(prediction[0]) > 0.5).cpu()
            image_with_mask = draw_segmentation_masks(img, pred_mask)
            image_with_mask = np.moveaxis(image_with_mask.cpu().numpy(), 0, -1)
            axs[i // 2, (i % 2)].imshow(image_with_mask)
            axs[i // 2, (i % 2)].axis('off')

            image_with_mask = draw_segmentation_masks(img, true_mask.type(torch.bool))
            image_with_mask = np.moveaxis(image_with_mask.cpu().numpy(), 0, -1)
            axs[i // 2, (i % 2)+2].imshow(image_with_mask)
            axs[i // 2, (i % 2)+2].axis('off')

        fig.tight_layout()
        fig.subplots_adjust(top=0.88)
        plt.show()

    for _ in range(5):
        fig, axs = plt.subplots(2, 4, figsize=(16, 8))
        fig.suptitle(f"model predictions{' ' * 105} original masks", fontsize=14)

        for i in range(4):
            sample = valid_dataset[random.randint(0, len(valid_dataset))]
            image = sample["image"].to(device)
            true_mask = sample['mask'].to(device)
            prediction = model.eval()(image.unsqueeze(dim=0))

            img = (image.cpu() * 255).type(torch.uint8)
            pred_mask = (torch.exp(prediction[0]) > 0.5).cpu()
            image_with_mask = draw_segmentation_masks(img, pred_mask)
            image_with_mask = np.moveaxis(image_with_mask.cpu().numpy(), 0, -1)
            axs[i // 2, (i % 2)].imshow(image_with_mask)
            axs[i // 2, (i % 2)].axis('off')

            image_with_mask = draw_segmentation_masks(img, true_mask.type(torch.bool))
            image_with_mask = np.moveaxis(image_with_mask.cpu().numpy(), 0, -1)
            axs[i // 2, (i % 2) + 2].imshow(image_with_mask)
            axs[i // 2, (i % 2) + 2].axis('off')

        fig.tight_layout()
        fig.subplots_adjust(top=0.88)
        plt.show()
