import os
import random

import configargparse
import namegenerator
import numpy as np
import torch
import yaml
from transformers import AutoProcessor, IdeficsForVisionText2Text

from utils import CustomPipeline, get_batch_of_images, make_batch_of_prompts


def run(configs):
    """
    run the model
    """

    # initialize model and processor
    model = IdeficsForVisionText2Text.from_pretrained(configs.checkpoint, torch_dtype=torch.bfloat16).to(configs.device)
    processor = AutoProcessor.from_pretrained(configs.checkpoint)

    # initialize pipeline
    pipeline = CustomPipeline(model, processor, configs)

    # initialize dataset
    # TODO:

    # for each batch of images, generate text and compare with targets
    # (think classic pytorch eval loop)
    for batch in range(1):
        images, targets = get_batch_of_images(), ["cat", "dog", "car", "plane", "skier"]

        # generate prompts around images
        prompts = make_batch_of_prompts(images)

        # get model outputs
        outputs = pipeline(prompts)

        # print outputs
        for pred, target in zip(outputs, targets):
            print(f"target: {target}\npredicted: {pred.split('Assistant:')[-1]}")

        # compare outputs with targets
        # TODO:

    print("done!")


if __name__ == "__main__":
    # parse args/config file
    parser = configargparse.ArgParser(default_config_files=["./config.yml"])
    parser.add_argument("-c", "--config", is_config_file=True, default="./config.yml", help="config file location")
    parser.add_argument("-o", "--output-dir", type=str, default="./out", help="output folder")
    parser.add_argument("-s", "--seed", type=int, help="random seed")
    parser.add_argument("--name", type=str, default="random", help="run name")
    parser.add_argument("--checkpoint", type=str, default="HuggingFaceM4/idefics-9b-instruct", help="model checkpoint")
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
