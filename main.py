import os
import random

import configargparse
import namegenerator
import numpy as np
import torch
import yaml
from dotenv import load_dotenv
from torch.utils.data.dataloader import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torchvision.transforms.functional import to_pil_image
from tqdm import tqdm
from transformers import AutoProcessor, IdeficsForVisionText2Text

from utils import CustomPipeline, make_batch_of_prompts, f1_score


def run(configs):
    """
    run the model
    """

    # initialize dataset
    transform = transforms.Compose([transforms.ToTensor()])

    train_dataset = CIFAR10(
        root=os.getenv("DOWNLOAD_DIR", default="./data"), download=True, train=True, transform=transform
    )
    val_dataset = CIFAR10(
        root=os.getenv("DOWNLOAD_DIR", default="./data"), download=True, train=False, transform=transform
    )

    # subset breaks the dataset classes, so we need to save them
    dataset_classes = train_dataset.classes

    # shrink dataset for development
    if configs.train_size != -1:
        train_dataset = torch.utils.data.Subset(train_dataset, np.arange(configs.train_size))

    train_loader = DataLoader(
        train_dataset, configs.batch_size, shuffle=True, num_workers=configs.num_workers, pin_memory=True
    )
    val_loader = DataLoader(val_dataset, configs.batch_size, num_workers=configs.num_workers, pin_memory=True)

    # initialize model and processor
    model = IdeficsForVisionText2Text.from_pretrained(configs.checkpoint, torch_dtype=torch.bfloat16).to(configs.device)
    processor = AutoProcessor.from_pretrained(configs.checkpoint)

    # initialize pipeline
    pipeline = CustomPipeline(model, processor, configs)

    # for each batch of images, generate text and compare with targets
    # (think classic pytorch eval loop)
    tbar_loader = tqdm(train_loader, dynamic_ncols=True)
    tbar_loader.set_description("train")

    ps = []
    ls = []
    num_correct = 0
    for images, labels in tbar_loader:
        images = [to_pil_image(image) for image in images]
        labels = [dataset_classes[label] for label in labels]

        # generate prompts around images
        # TODO: improve prompts
        print("labels\n",",".join(labels))
        print("length preds",len(outputs))
        prompts = make_batch_of_prompts(images, labels)

        # get model outputs
        outputs = pipeline(prompts)

        # compare outputs with targets
        for pred, label in zip(outputs, labels):
            print(f"target: {label}\npredicted: {pred.split('Assistant:')[-1]}")
            # TODO: make better accuracy method
            if label in pred.split('Assistant:')[-1].strip().lower():
                num_correct += 1
                
            ps.append(pred.split('Assistant:')[-1].strip().lower())
            ls.append(label)
    
    print(f"accuracy: {num_correct / len(train_loader.dataset)}")
    print(ls,"\n",ps)
    print(f"F1 score: {f1_score(ls, ps)}")
    print("done!")


if __name__ == "__main__":
    # load environment variables
    load_dotenv()

    # parse args/config file
    parser = configargparse.ArgParser(default_config_files=["./config.yml"])
    parser.add_argument("-c", "--config", is_config_file=True, default="./config.yml", help="config file location")
    parser.add_argument("-o", "--output-dir", type=str, default="./out", help="output folder")
    parser.add_argument("-s", "--seed", type=int, help="random seed")
    parser.add_argument("--name", type=str, default="random", help="run name")
    parser.add_argument("--checkpoint", type=str, default="HuggingFaceM4/idefics-9b-instruct", help="model checkpoint")
    parser.add_argument("--train-size", type=int, default=-1, help="train dataset size (for development)")
    parser.add_argument("--batch-size", type=int, default=128, help="training batch size")
    parser.add_argument("--num-workers", type=int, default=4, help="number of workers for dataloader")
    configs, _ = parser.parse_known_args()

    # set device
    configs.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # set run name
    if configs.name == "random":
        configs.name = namegenerator.gen()
    else:
        configs.name = configs.name

    # set seed
    if configs.seed:
        random.seed(configs.seed)
        np.random.seed(configs.seed)
        torch.manual_seed(configs.seed)
        torch.cuda.manual_seed(configs.seed)

    # create output folder
    configs.output_dir = os.path.join(configs.output_dir, configs.name)
    os.makedirs(configs.output_dir, exist_ok=True)

    # save configs object as yaml
    with open(os.path.join(configs.output_dir, "config.yml"), "w", encoding="utf-8") as file:
        yaml.dump(vars(configs), file)

    run(configs)
