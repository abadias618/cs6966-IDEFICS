from PIL import Image
import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize
from collections import Counter

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
            max_length=100,
        )

        # postprocess outputs
        out = self.processor.batch_decode(generated_ids, skip_special_tokens=True)

        return out


def make_batch_of_prompts(images: list[Image.Image]) -> list:
    """
    take in batch of images and make batch of text prompts
    """

    prompts = []
    for image in images:
        prompts.append(
            [
                "User: Classify this image in 1 to 2 words (preferably 1 word), then, give an explanation in 1 sentence of why you classified that way.",
                image,
                "<end_of_utterance>",
                "\nAssistant:",
            ]
        )

    return prompts

# https://kierszbaumsamuel.medium.com/f1-score-in-nlp-span-based-qa-task-5b115a5e7d41
def f1_score(gold: str, pred: str) -> float:
    """
    Get f1_score comparing gold standard and prediction.
    """
    g_toks = word_tokenize(gold)
    p_toks = word_tokenize(pred)
    common = Counter(g_toks) & Counter(p_toks)
    num_same = sum(common.values())

    if len(g_toks) == 0 or len(p_toks) == 0:
        # if either is empty then F1 is 1 if they agree, 0 otherwise.
        return int(g_toks == p_toks)

    if num_same == 0:
        return 0
    
    precision = 1.0 * num_same / len(p_toks)
    recall = 1.0 * sum_same / len(g_toks)
    f1 = (2 * precision * recall) / (precision + recall)

    return f1