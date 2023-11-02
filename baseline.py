# pylint: disable=line-too-long
"""
baseline version of the model

Jakob Johnson
10/1/2023
"""

import os
import random

import configargparse
import namegenerator
import numpy as np
import torch
import yaml
from torch import nn
from torch.backends import cudnn
from torch.utils.data import DataLoader
from torchmetrics.classification import MulticlassAccuracy
from torchvision import datasets, models, transforms
from tqdm import tqdm


def train_loop(dataloader, model, optimizer):
    """Train the model for one epoch."""
    # set model to train mode
    model.train()

    # set up loss and metrics
    loss_fn = nn.CrossEntropyLoss()
    accuracy = MulticlassAccuracy(num_classes=NUM_CLASSES).to(configs.device)

    num_batches = len(dataloader)
    train_loss = 0

    tbar_loader = tqdm(dataloader, dynamic_ncols=True)
    tbar_loader.set_description("train")

    for images, labels in tbar_loader:
        # move images to GPU if needed
        images, labels = (
            images.to(configs.device),
            labels.to(configs.device),
        )

        # zero gradients from previous step
        optimizer.zero_grad()

        # compute prediction and loss
        preds = model(images)
        loss = loss_fn(preds, labels)
        train_loss += loss.item()

        # backpropagation
        loss.backward()
        optimizer.step()
        scheduler.step()

        # update metrics
        accuracy.update(preds, labels)

    return {
        "train_acc": float(accuracy.compute()),
        "train_loss": train_loss / num_batches,
        "learning_rate": scheduler.get_last_lr()[0],
    }


def val_loop(dataloader, model):
    """Validate the model for one epoch."""
    # set model to eval mode
    model.eval()

    # set up loss and metrics
    loss_fn = nn.CrossEntropyLoss()
    accuracy = MulticlassAccuracy(num_classes=NUM_CLASSES).to(configs.device)

    num_batches = len(dataloader)
    val_loss = 0

    with torch.no_grad():
        tbar_loader = tqdm(dataloader, dynamic_ncols=True)
        tbar_loader.set_description("val")

        for images, labels in tbar_loader:
            # move images to GPU if needed
            images, labels = (images.to(configs.device), labels.to(configs.device))

            # compute prediction and loss
            preds = model(images)
            val_loss += loss_fn(preds, labels).item()

            # update metrics
            accuracy.update(preds, labels)

    return {
        "val_acc": float(accuracy.compute()),
        "val_loss": val_loss / num_batches,
    }


def cosine_annealing(step, total_steps, lr_max, lr_min):
    return lr_min + (lr_max - lr_min) * 0.5 * (1 + np.cos(step / total_steps * np.pi))


if __name__ == "__main__":
    # parse args/config file
    parser = configargparse.ArgParser(default_config_files=["./baseline.yml"])
    parser.add_argument("-c", "--config", is_config_file=True, default="./baseline.yml", help="config file location")
    parser.add_argument("-e", "--epochs", type=int, default=1, help="number of epochs to train for")
    parser.add_argument("--arch", type=str, default="resnet18", help="model architecture")
    parser.add_argument("--pretrained", action="store_true", help="use pretrained model")
    parser.add_argument("--name", type=str, default="random", help="run name")
    parser.add_argument("--dataset", type=str, default="cifar10", help="dataset name")
    parser.add_argument("-r", "--dataset-root", type=str, default="./data/", help="dataset filepath")
    parser.add_argument("--batch-size", type=int, default=8, help="batch size")
    parser.add_argument("--workers", type=int, default=2, help="dataloader worker threads")
    parser.add_argument("--lr", type=float, default=0.1, help="learning rate")
    parser.add_argument("--momentum", type=float, default=0.9, help="optimizer momentum")
    parser.add_argument("--weight-decay", type=float, default=0.0005, help="optimizer weight decay")
    parser.add_argument("-S", "--seed", type=int, default=-1, help="random seed, -1 for random")
    parser.add_argument("--device", type=str, default="cuda", help="gpu(s) to use")
    parser.add_argument("--root", type=str, default="runs", help="root of folder to save runs in")
    configs, _ = parser.parse_known_args()

    #########################################
    ## SET UP SEEDS AND PRE-TRAINING FILES ##
    #########################################
    if configs.name == "random":
        configs.name = namegenerator.gen()
    else:
        configs.name = configs.name

    if configs.seed != -1:
        random.seed(configs.seed)
        torch.manual_seed(configs.seed)
        cudnn.deterministic = True

    print(f"Run name: {configs.name}")
    try:
        os.makedirs(f"{configs.root}/{configs.name}", exist_ok=True)
    except FileExistsError as error:
        pass
    configs.root = f"{configs.root}/{configs.name}"

    # save configs object as yaml
    with open(os.path.join(configs.root, "baseline.yaml"), "w", encoding="utf-8") as file:
        yaml.dump(vars(configs), file)

    ####################
    ## SET UP DATASET ##
    ####################

    match configs.dataset.lower():
        case "imagenet":
            NUM_CLASSES = 1000

            transform = transforms.Compose(
                [
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomVerticalFlip(),
                    # transforms.ColorJitter(brightness=0.5),
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    # lambda x: x / 255.0,  # match ToTensor()'s conversion to [0,1]
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),  # imagenet1k mean and sd
                ]
            )

            training_data = datasets.ImageNet(root=configs.dataset_root, split="train", transform=transform)
            val_data = datasets.ImageNet(root=configs.dataset_root, split="val", transform=transform)

        case "cifar10":
            NUM_CLASSES = 10

            train_transform = transforms.Compose(
                [
                    lambda x: x.convert("RGB"),
                    transforms.Resize(32),
                    transforms.CenterCrop(32),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomCrop(32, padding=4),
                    transforms.ToTensor(),
                    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616]),
                ]
            )
            val_transform = transforms.Compose(
                [
                    lambda x: x.convert("RGB"),
                    transforms.ToTensor(),
                    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616]),
                ]
            )

            training_data = datasets.CIFAR10(
                root=configs.dataset_root,
                train=True,
                transform=train_transform,
                download=True,
            )
            val_data = datasets.CIFAR10(
                root=configs.dataset_root,
                train=False,
                transform=val_transform,
                download=True,
            )

        case "cifar100":
            NUM_CLASSES = 100

            train_transform = transforms.Compose(
                [
                    lambda x: x.convert("RGB"),
                    transforms.Resize(32),
                    transforms.CenterCrop(32),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomCrop(32, padding=4),
                    transforms.ToTensor(),
                    transforms.Normalize([0.5071, 0.4867, 0.4408], [0.2675, 0.2565, 0.2761]),  # cifar100 mean and sd
                ]
            )
            val_transform = transforms.Compose(
                [
                    lambda x: x.convert("RGB"),
                    transforms.ToTensor(),
                    transforms.Normalize([0.5071, 0.4867, 0.4408], [0.2675, 0.2565, 0.2761]),  # cifar100 mean and sd
                ]
            )

            training_data = datasets.CIFAR100(
                root=configs.dataset_root,
                train=True,
                transform=train_transform,
                download=True,
            )
            val_data = datasets.CIFAR100(
                root=configs.dataset_root,
                train=False,
                transform=val_transform,
                download=True,
            )

        case _:
            raise ValueError(f"Dataset {configs.dataset} not supported")

    print(training_data)
    print(val_data)
    NUM_CLASSES = len(training_data.classes)

    train_dataloader = DataLoader(
        training_data,
        batch_size=configs.batch_size,
        shuffle=True,
        num_workers=configs.workers,
    )
    val_dataloader = DataLoader(
        val_data,
        batch_size=configs.batch_size,
        shuffle=False,
        num_workers=configs.workers,
    )

    ##################
    ## SET UP MODEL ##
    ##################
    print(f"Using device: {configs.device}")

    # choose model architecture
    match configs.arch.lower():
        case "resnet18":
            model = models.resnet18(weights=(models.ResNet18_Weights.IMAGENET1K_V1 if configs.pretrained else None))
        case "resnet34":
            model = models.resnet34(weights=(models.ResNet34_Weights.IMAGENET1K_V1 if configs.pretrained else None))
        case "resnet50":
            model = models.resnet50(weights=(models.ResNet50_Weights.IMAGENET1K_V1 if configs.pretrained else None))
        case _:
            raise ValueError(f"Model {configs.arch} not supported")

    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    model = model.to(configs.device)
    print(model)

    # initialize optimizer
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=configs.lr,
        momentum=configs.momentum,
        weight_decay=configs.weight_decay,
        nesterov=True,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: cosine_annealing(
            step,
            configs.epochs * len(train_dataloader),
            1,
            1e-6 / configs.lr,
        ),
    )

    #################
    ## TRAIN MODEL ##
    #################

    for epoch in range(configs.epochs):
        train_stats = train_loop(train_dataloader, model, optimizer)
        val_stats = val_loop(val_dataloader, model)
        print(epoch, train_stats | val_stats)

    print("Done!")

    ################
    ## SAVE MODEL ##
    ################
    torch.save(model.state_dict(), os.path.join(configs.root, "model-weights.pth"))
