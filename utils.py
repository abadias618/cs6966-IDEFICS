from PIL import Image
from Levenshtein import distance
from collections import Counter
import random


example_images = [
    {
        "image": Image.open("sample_images/automobile4.png"),
        "label": "automobile",
        "explanation": "the image is of an automobile because it has four wheels and only carries a few passengers.",
    },
    {
        "image": Image.open("sample_images/ship7.png"),
        "label": "ship",
        "explanation": "the image is of a small boat which is a type of ship and because the object is floating on water.",
    },
    {
        "image": Image.open("sample_images/deer7.png"),
        "label": "deer",
        "explanation": "the image is of a deer because it is a brown animal with four legs and antlers.",
    },
    {
        "image": Image.open("sample_images/horse5.png"),
        "label": "horse",
        "explanation": "the image is of a horse because it is an animal with four legs and a mane and long tail.",
    },
]


class CustomPipeline:
    def __init__(self, model, processor, configs):
        self.model = model
        self.processor = processor
        self.configs = configs

    def __call__(self, prompts):
        ## from https://huggingface.co/HuggingFaceM4/idefics-9b-instruct

        # preprocess inputs
        # --batched mode
        inputs = self.processor(prompts, add_end_of_utterance_token=False, return_tensors="pt").to(self.configs.device)
        # --single sample mode
        # inputs = processor(prompts[0], return_tensors="pt").to(device)

        # generation args
        exit_condition = self.processor.tokenizer("<end_of_utterance>", add_special_tokens=False).input_ids
        bad_words_ids = self.processor.tokenizer(
            ["<image>", "<fake_token_around_image>"], add_special_tokens=False
        ).input_ids

        # get model outout
        generated_ids = self.model.generate(
            **inputs,
            eos_token_id=exit_condition,
            bad_words_ids=bad_words_ids,
            max_length=self.configs.max_length,
        )

        # postprocess outputs
        out = self.processor.batch_decode(generated_ids, skip_special_tokens=True)

        return out


def make_batch_of_prompts(images: list[Image.Image], labels) -> list:
    """
    take in batch of images and make batch of text prompts
    """
    labels_literal = ",".join(labels)

    # load sample images
    random.shuffle(example_images)
    two_sample_images = example_images[:2]

    prompts = []
    for image in images:
        inst_prompt = []

        # load two sample images
        for sample in two_sample_images:
            inst_prompt.extend(
                [
                    f"User: choose one of the items of the following list to classify the image given: {labels_literal}.",
                    sample["image"],
                    "<end_of_utterance>",
                    f"\nAssistant: {sample['label']}, {sample['explanation']} <end_of_utterance>",
                ]
            )

        inst_prompt.extend(
            ## prompt one:
            [
                f"User: choose one of the items of the following list to classify the image given: {labels_literal}. Afterwards, give an explanation in one sentence of why you chose that class.",
                image,
                "<end_of_utterance>",
                "\nAssistant:",
            ]
            # [
            #     f"User: choose one of the items of the following list to classify the image given: {labels_literal}.",
            #     image,
            #     "<end_of_utterance>",
            #     "\nAssistant:",
            # ]
            ## prompt two:
            # [
            #     f"User: Classify this image in one word only, then give an explanation of why you chose to classify the image in that way.",
            #     image,
            #     "<end_of_utterance>",
            #     "\nAssistant:",
            # ]
            # [
            #     f"User: Classify this image in one word only.",
            #     image,
            #     "<end_of_utterance>",
            #     "\nAssistant:",
            # ]
        )

        prompts.append(inst_prompt)

    return prompts


# https://stackoverflow.com/questions/76802665/f1-score-and-accuracy-for-text-similarity
def f1_score(labels: str, preds: str, threshold=0.0) -> float:
    """
    Get f1_score comparing gold standard and prediction.
    """
    tp, fp, fn = 0, 0, 0

    for label, pred in zip(labels, preds):
        similarity_score = 1 - distance(label, pred) / max(len(label), len(pred))
        if similarity_score >= threshold:
            tp += 1
        else:
            fp += 1

    fn = len(labels) - tp

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1_score = 2 * (precision * recall) / (precision + recall)

    return f1_score
