import os
import random

import configargparse
import namegenerator
import numpy as np
import torch
import yaml
from transformers import AutoProcessor, IdeficsForVisionText2Text


def run(configs):
    ## from https://huggingface.co/HuggingFaceM4/idefics-9b-instruct

    model = IdeficsForVisionText2Text.from_pretrained(configs.checkpoint, torch_dtype=torch.bfloat16).to(configs.device)
    processor = AutoProcessor.from_pretrained(configs.checkpoint)

    # We feed to the model an arbitrary sequence of text strings and images. Images can be either URLs or PIL Images.
    prompts = [
        [
            "User: What is in this image?",
            "https://upload.wikimedia.org/wikipedia/commons/8/86/Id%C3%A9fix.JPG",
            "<end_of_utterance>",
            "\nAssistant: This picture depicts Idefix, the dog of Obelix in Asterix and Obelix. Idefix is running on the ground.<end_of_utterance>",
            "\nUser:",
            "https://static.wikia.nocookie.net/asterix/images/2/25/R22b.gif/revision/latest?cb=20110815073052",
            "And who is that?<end_of_utterance>",
            "\nAssistant:",
        ],
    ]

    # --batched mode
    inputs = processor(prompts, add_end_of_utterance_token=False, return_tensors="pt").to(configs.device)
    # --single sample mode
    # inputs = processor(prompts[0], return_tensors="pt").to(device)

    # Generation args
    exit_condition = processor.tokenizer("<end_of_utterance>", add_special_tokens=False).input_ids
    bad_words_ids = processor.tokenizer(["<image>", "<fake_token_around_image>"], add_special_tokens=False).input_ids

    generated_ids = model.generate(
        **inputs,
        eos_token_id=exit_condition,
        bad_words_ids=bad_words_ids,
        max_length=100,
    )
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)
    for i, t in enumerate(generated_text):
        print(f"{i}:\n{t}\n")


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
