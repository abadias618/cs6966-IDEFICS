import os
import random

import configargparse
import namegenerator
import numpy as np
import torch
import yaml
from transformers import AutoProcessor, IdeficsForVisionText2Text

from utils import CustomPipeline


def run(configs):
    ## from https://huggingface.co/HuggingFaceM4/idefics-9b-instruct

    model = IdeficsForVisionText2Text.from_pretrained(configs.checkpoint, torch_dtype=torch.bfloat16).to(configs.device)
    processor = AutoProcessor.from_pretrained(configs.checkpoint)

    pipeline = CustomPipeline(model, processor, configs)

    # We feed to the model an arbitrary sequence of text strings and images. Images can be either URLs or PIL Images.
    prompts = [
        [
            "User: What is in this image?",
            "https://upload.wikimedia.org/wikipedia/commons/thumb/1/15/Cat_August_2010-4.jpg/1920px-Cat_August_2010-4.jpg",
            "<end_of_utterance>",
            "\nAssistant:",
        ],
        [
            "User: What is in this image?",
            "https://upload.wikimedia.org/wikipedia/commons/7/77/Sarabi-dog.jpg",
            "<end_of_utterance>",
            "\nAssistant:",
        ],
        [
            "User: What is in this image?",
            "https://upload.wikimedia.org/wikipedia/commons/thumb/e/eb/2011_Toyota_Corolla_--_NHTSA.jpg/1920px-2011_Toyota_Corolla_--_NHTSA.jpg",
            "<end_of_utterance>",
            "\nAssistant:",
        ],
        [
            "User: What is in this image?",
            "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3f/P-51_Mustang_edit1.jpg/1920px-P-51_Mustang_edit1.jpg",
            "<end_of_utterance>",
            "\nAssistant:",
        ],
        [
            "User: What is in this image?",
            "https://upload.wikimedia.org/wikipedia/commons/8/84/Ski_Famille_-_Family_Ski_Holidays.jpg",
            "<end_of_utterance>",
            "\nAssistant:",
        ],
    ]

    generated_text = pipeline(prompts)

    for text in enumerate(generated_text):
        print(text[1])


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
