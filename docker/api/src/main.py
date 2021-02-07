from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from easynmt import EasyNMT
from typing import Optional, Union, List
import time
import os
import json
import time
import datetime
import requests
import http3





app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



IS_BACKEND = os.getenv('ROLE', 'FRONT') == 'BACKEND'
BACKEND_URL = os.getenv('BACKEND_URL', 'http://localhost:8080')

print("Booted as backend: {}".format(IS_BACKEND))


model_name = os.getenv('EASYNMT_MODEL', 'opus-mt')
model_args = json.loads(os.getenv('EASYNMT_MODEL_ARGS', '{}') )
print("Load model: "+ model_name)
model = EasyNMT(model_name, load_translator=IS_BACKEND, **model_args)



@app.get("/translate")
async def translate(target_lang: str, text: List[str] = Query([]), source_lang: Optional[str] = '', beam_size: Optional[int] = 5, perform_sentence_splitting: Optional[bool] = True):
    """
    Translation
    :param text: Text that should be translated
    :param target_lang: Target language
    :param source_lang: Language of text (optional)
    :param beam_size: Beam size
    :param perform_sentence_splitting: Split longer documents into individual sentences for translation
    :return:  Returns a json with the entries:
    """

    if not IS_BACKEND:
        async_client = http3.AsyncClient()
        data = {'target_lang': target_lang, 'text': text, 'source_lang': source_lang, 'beam_size': beam_size, 'perform_sentence_splitting': perform_sentence_splitting}
        x = await async_client.post(BACKEND_URL+'/translate', json=data, timeout=3600)
        if x.status_code != 200:
            error_msg = "Error: " + x.text
            try:
                error_msg = x.json()['detail']
            except:
                pass

            raise HTTPException(403, detail=error_msg)
        return x.json()

    else:
        #Check input parameters
        if 'EASYNMT_MAX_TEXT_LEN' in os.environ and len(text) > int(os.getenv('EASYNMT_MAX_TEXT_LEN')):
            raise ValueError("Text was too long. Only texts up to {} characters are allowed".format(os.getenv('EASYNMT_MAX_TEXT_LEN')))

        if beam_size < 1 or ('EASYNMT_MAX_BEAM_SIZE' in os.environ and beam_size > int(os.getenv('EASYNMT_MAX_BEAM_SIZE'))):
            raise ValueError("Illegal beam size")

        if len(source_lang.strip()) == 0:
            source_lang = None

        #Start the translation
        start_time = time.time()
        output = {"target_lang": target_lang, "source_lang": source_lang}


        if source_lang is None:
            detected_langs = model.language_detection(text)
            output['detected_langs'] = detected_langs
            #TODO Grouping and individual translation
            #Exception: add original text


        try:
            output['translated'] = model.translate(text, target_lang=target_lang, source_lang=source_lang, beam_size=beam_size, perform_sentence_splitting=perform_sentence_splitting, batch_size=int(os.getenv('EASYNMT_BATCH_SIZE', 16)))
        except Exception as e:
            raise HTTPException(403, detail="Error: "+str(e))

        output['translation_time'] = time.time()-start_time
        return output


@app.post("/translate")
async def translate_post(request: Request):
    data = await request.json()
    return await translate(**data)


@app.get("/lang_pairs")
async def lang_pairs():
    return model.lang_pairs


@app.get("/get_languages")
async def get_languages(source_lang: Optional[str] = None, target_lang: Optional[str] = None):
    return model.get_languages(source_lang=source_lang, target_lang=target_lang)


@app.get("/language_detection")
async def language_detection(text: str):
    """
    Detects the language for the provided text
    :param text: A single text for which we want to know the language
    :return: The detected language
    """
    return model.language_detection(text)


@app.post("/language_detection")
async def language_detection_post(request: Request):
    """
    Pass a json that has a 'text' key. The 'text' element can either be a string, a list of strings, or
    a dict.
    :return: Languages detected
    """
    data = await request.json()
    if isinstance(data['text'], list):
        return [model.language_detection(t) for t in data['text']]
    elif isinstance(data['text'], dict):
        return {k: model.language_detection(t) for k, t in data['text'].items()}
    return model.language_detection(data['text'])




@app.get("/model_name")
async def lang_pairs():
    """
    Returns the name of the loaded model
    :return: EasyNMT model name
    """
    return model._model_name