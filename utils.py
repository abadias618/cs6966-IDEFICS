from PIL import Image


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
                "User: What is in this image?",
                image,
                "<end_of_utterance>",
                "\nAssistant:",
            ]
        )

    return prompts
