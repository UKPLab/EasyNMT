from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch
from typing import List
import logging



logger = logging.getLogger(__name__)


class AutoModel:
    def __init__(self, model_name: str, tokenizer_name: str = None, easynmt_path: str = None, lang_map=None, tokenizer_args=None):
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


    def translate_sentences(self, sentences: List[str], source_lang: str, target_lang: str, device: str, beam_size: int = 5, **kwargs):
        self.model.to(device)

        if source_lang in self.lang_map:
            source_lang = self.lang_map[source_lang]

        if target_lang in self.lang_map:
            target_lang = self.lang_map[target_lang]

        self.tokenizer.src_lang = source_lang
        inputs = self.tokenizer(sentences, truncation=True, padding=True, return_tensors="pt")

        for key in inputs:
            inputs[key] = inputs[key].to(device)

        with torch.no_grad():
            if hasattr(self.tokenizer, 'lang_code_to_id'):
                kwargs['forced_bos_token_id'] = self.tokenizer.lang_code_to_id[target_lang]
            translated = self.model.generate(**inputs, num_beams=beam_size, **kwargs)
            output = [self.tokenizer.decode(t, skip_special_tokens=True) for t in translated]

        return output

    def save(self, output_path):
        self.model.save_pretrained(output_path)
        self.tokenizer.save_pretrained(output_path)
        return {
            "model_name": ".",
            "tokenizer_name": ".",
            "lang_map": self.lang_map,
            "tokenizer_args": self.tokenizer_args
        }
