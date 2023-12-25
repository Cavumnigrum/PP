import albumentations as A
from albumentations.pytorch import ToTensorV2

import numpy as np
import cv2
from sklearn.model_selection import train_test_split
import segmentation_models_pytorch as smp

import torch
from torch.utils.data import Dataset

import matplotlib.pyplot as plt

from UnetTrainer import UnetTrainer
from metrics import SoftDice, make_metrics
from EyeDataset import EyeDataset, DatasetPart

import warnings

warnings.filterwarnings("ignore")


def plot_history(train_history, val_history, title='loss'):
    plt.figure()
    plt.title('{}'.format(title))
    plt.plot(train_history, label='train', zorder=1)

    points = np.array(val_history)
    steps = list(range(0, len(train_history) + 1, int(len(train_history) / len(val_history))))[1:]

    plt.scatter(steps, val_history, marker='+', s=180, c='orange', label='val', zorder=2)
    plt.xlabel('train steps')

    plt.legend(loc='best')
    plt.grid()

    plt.show()


def make_criterion():
    soft_dice = SoftDice()

    def exp_dice(pred, target):
        return 1 - soft_dice(torch.exp(pred[:, 1:]), target[:, 1:])

    return exp_dice


if __name__ == "__main__":

    print(torch.cuda.is_available())
    device = "cuda" if torch.cuda.is_available() else 'cpu'

    size = 1024
    # excessive augmentation
    # train_transform = A.Compose([A.LongestMaxSize(size, interpolation=cv2.INTER_CUBIC),
    #               A.PadIfNeeded(size, size),
    #               A.Rotate(limit=40),
    #               A.RandomBrightness(limit=0.1),
    #               A.JpegCompression(quality_lower=85, quality_upper=100, p=0.5),
    #               A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5),
    #               A.RandomContrast(limit=0.2, p=0.5),
    #               A.HorizontalFlip(),
    #               ToTensorV2(transpose_mask=True),
    #               ])
    # eval_transform = A.Compose([A.LongestMaxSize(size, interpolation=cv2.INTER_CUBIC),
    #               A.PadIfNeeded(size, size),
    #               A.Rotate(limit=40),
    #               A.RandomBrightness(limit=0.1),
    #               A.JpegCompression(quality_lower=85, quality_upper=100, p=0.5),
    #               A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5),
    #               A.RandomContrast(limit=0.2, p=0.5),
    #               A.HorizontalFlip(),
    #               ToTensorV2(transpose_mask=True),])

    # additional augmentation
    train_list = [A.LongestMaxSize(size, interpolation=cv2.INTER_CUBIC),
                  A.PadIfNeeded(size, size),
                  A.RandomContrast(limit=0.05, p=0.5),
                  A.HorizontalFlip(),
                  A.VerticalFlip(),
                  A.RandomBrightness(limit=0.05),
                  ToTensorV2(transpose_mask=True),
                  ]
    eval_list = [A.LongestMaxSize(size, interpolation=cv2.INTER_CUBIC),
                 A.PadIfNeeded(size, size),
                 ToTensorV2(transpose_mask=True)]

    # default augmentation
    # train_list = [A.LongestMaxSize(size, interpolation=cv2.INTER_CUBIC),
    #               A.PadIfNeeded(size, size),
    #               ToTensorV2(transpose_mask=True),
    #               ]
    # eval_list = [A.LongestMaxSize(size, interpolation=cv2.INTER_CUBIC),
    #              A.PadIfNeeded(size, size),
    #              ToTensorV2(transpose_mask=True)]

    transforms = {'train': A.Compose(train_list), 'test': A.Compose(eval_list)}

    # Инициализируем датасет
    dataset = EyeDataset("./Dataset/eye_train")

    # Проверим состояние загруженного датасета
    for msg in dataset.make_report():
        print(msg)

    # разделим датасет на тренировочный и валидационный, чтобы смотреть на качество
    train_indices, test_indices = train_test_split(range(len(dataset)), test_size=0.25)

    print(f"Разбиение на train/test : {len(train_indices)}/{len(test_indices)}")

    # Разбиваем объект датасета на тренировачный и валидационный
    train_dataset = DatasetPart(dataset, train_indices, transform=transforms['train'])
    valid_dataset = DatasetPart(dataset, test_indices, transform=transforms['test'])

    train_loader = torch.utils.data.DataLoader(train_dataset, 1,
                                               num_workers=4,
                                               shuffle=True, drop_last=True)

    valid_loader = torch.utils.data.DataLoader(valid_dataset, 1,
                                               num_workers=4,
                                               shuffle=True, drop_last=True)

    torch.cuda.empty_cache()
    # Подргружаем модель и задаём функцию потерь
    model = smp.Unet('resnet50', activation='logsoftmax', classes=2).cuda()

    # Критерии
    criterion = make_criterion()

    optimizer = torch.optim.Adam(model.parameters(), 0.0001)

    # Обучение модели
    trainer = UnetTrainer(model, optimizer, criterion, device, metric_functions=make_metrics())
    summary = trainer.fit(train_loader, 20, val_loader=valid_loader)

    print(summary)

    plot_history(summary['loss_train'], summary['loss_test'])
    plot_history(summary['accuracy_train'], summary['accuracy_test'], "accuracy")
    plot_history(summary['recall_train'], summary['recall_test'], "recall")

    torch.save(model.state_dict(), "./model/weights_additional_aug")
