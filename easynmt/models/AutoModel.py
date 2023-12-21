from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch
from torch.cuda.amp import autocast
from typing import List
import logging


logger = logging.getLogger(__name__)


class AutoModel:
    def __init__(self, model_name: str, tokenizer_name: str = None, easynmt_path: str = None, lang_map=None, tokenizer_args=None):
        """
        Initializes an instance of the AutoModel class.

        Args:
            model_name (str): The name or path of the pre-trained model to be used for translation.
            tokenizer_name (str, optional): The name or path of the tokenizer associated with the pre-trained model. Defaults to None.
            easynmt_path (str, optional): The path to the EasyNMT model if the model_name or tokenizer_name is set to ".". Defaults to None.
            lang_map (dict, optional): A dictionary mapping language codes to specific language codes used by the tokenizer. Defaults to None.
            tokenizer_args (dict, optional): Additional arguments to be passed to the tokenizer. Defaults to None.
        """
        if tokenizer_args is None:
            tokenizer_args = {}

        if lang_map is None:
            lang_map = {}

        if tokenizer_name is None:
            tokenizer_name = model_name

        self.lang_map = lang_map
        self.tokenizer_args = tokenizer_args

        if model_name == ".":
            model_name = easynmt_path

        if tokenizer_name == ".":
            tokenizer_name = easynmt_path

        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, **self.tokenizer_args)
        self.max_length = 512  # Set a smaller value for low memory GPUs


    def translate_sentences(self, sentences: List[str], source_lang: str, target_lang: str, device: str, beam_size: int = 5, with_autocast: bool = False, **kwargs):
        """
        Translates a list of sentences from a source language to a target language.

        Args:
            sentences (List[str]): The list of sentences to be translated.
            source_lang (str): The source language of the sentences.
            target_lang (str): The target language for translation.
            device (str): The device to be used for translation (e.g. "cuda").
            beam_size (int, optional): The beam size for translation. Defaults to 5.
            with_autocast (bool, optional): Whether to use autocast for translation. Defaults to False.
            **kwargs: Additional keyword arguments to be passed to the translation model.

        Returns:
            List[str]: A list of translated sentences.
        """
        self.model.to(device)

        if source_lang in self.lang_map:
            source_lang = self.lang_map[source_lang]

        if target_lang in self.lang_map:
            target_lang = self.lang_map[target_lang]

        self.tokenizer.src_lang = source_lang
        inputs = self.tokenizer(sentences, truncation=True, padding=True, max_length=self.max_length, return_tensors="pt")

        for key in inputs:
            inputs[key] = inputs[key].to(device)

        with torch.no_grad():
            if with_autocast:
                with autocast():
                    if hasattr(self.tokenizer, 'lang_code_to_id'):
                        kwargs['forced_bos_token_id'] = self.tokenizer.lang_code_to_id[target_lang]
                    translated = self.model.generate(**inputs, num_beams=beam_size, **kwargs)
            else:
                if hasattr(self.tokenizer, 'lang_code_to_id'):
                    kwargs['forced_bos_token_id'] = self.tokenizer.lang_code_to_id[target_lang]
                translated = self.model.generate(**inputs, num_beams=beam_size, **kwargs)
            
            output = [self.tokenizer.decode(t, skip_special_tokens=True) for t in translated]

        return output

    def save(self, output_path):
        """
        Saves the model and tokenizer to the specified output path.

        Args:
            output_path (str): The path to save the model and tokenizer.

        Returns:
            dict: A dictionary containing the saved model and tokenizer information.
        """
        self.model.save_pretrained(output_path)
        self.tokenizer.save_pretrained(output_path)
        return {
            "model_name": ".",
            "tokenizer_name": ".",
            "lang_map": self.lang_map,
            "tokenizer_args": self.tokenizer_args
        }
