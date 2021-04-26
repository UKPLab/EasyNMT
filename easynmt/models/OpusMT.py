import time
from transformers import MarianMTModel, MarianTokenizer
import torch
from typing import List
import logging

logger = logging.getLogger(__name__)


class OpusMT:
    def __init__(self, easynmt_path: str = None, max_loaded_models: int = 10):
        self.models = {}
        self.max_loaded_models = max_loaded_models
        self.max_length = None

    def load_model(self, model_name):
        if model_name in self.models:
            self.models[model_name]['last_loaded'] = time.time()
            return self.models[model_name]['tokenizer'], self.models[model_name]['model']
        else:
            logger.info("Load model: "+model_name)
            tokenizer = MarianTokenizer.from_pretrained(model_name)
            model = MarianMTModel.from_pretrained(model_name)
            model.eval()

            if len(self.models) >= self.max_loaded_models:
                oldest_time = time.time()
                oldest_model = None
                for loaded_model_name in self.models:
                    if self.models[loaded_model_name]['last_loaded'] <= oldest_time:
                        oldest_model = loaded_model_name
                        oldest_time = self.models[loaded_model_name]['last_loaded']
                del self.models[oldest_model]

            self.models[model_name] = {'tokenizer': tokenizer, 'model': model, 'last_loaded': time.time()}
            return tokenizer, model

    def translate_sentences(self, sentences: List[str], source_lang: str, target_lang: str, device: str, beam_size: int = 5, **kwargs):
        model_name = 'Helsinki-NLP/opus-mt-{}-{}'.format(source_lang, target_lang)
        tokenizer, model = self.load_model(model_name)
        model.to(device)

        inputs = tokenizer(sentences, truncation=True, padding=True, max_length=self.max_length, return_tensors="pt")

        for key in inputs:
            inputs[key] = inputs[key].to(device)

        with torch.no_grad():
            translated = model.generate(**inputs, num_beams=beam_size, **kwargs)
            output = [tokenizer.decode(t, skip_special_tokens=True) for t in translated]

        return output

    def save(self, output_path):
        return {"max_loaded_models": self.max_loaded_models}

