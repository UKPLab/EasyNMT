# Docker

We provide a [Docker](https://www.docker.com/) based REST-API for EasyNMT: Send a query with your source text, and the API returns the translated text.

## Setup

To start the EasyNMT REST API on port `24080`run the following docker command:
```
docker run -p 24080:80 easynmt/api:2.0-cpu
```

This uses the CPU image. If you have **GPU (CUDA)**, there are various GPU images available. Have a look at our [Docker Hub Page](https://hub.docker.com/r/easynmt/api/tags?page=1&ordering=last_updated).


## Usage

After you started the Docker image, you can visit: [http://localhost:24080/translate?target_lang=en&text=Hallo%20Welt](http://localhost:24080/translate?target_lang=en&text=Hallo%20Welt)

This should yield the following JSON:
```
{
    "target_lang": "en",
    "source_lang": null,
    "detected_langs": [
        "de"
    ],
    "translated": [
        "Hello world"
    ],
    "translation_time": 0.163145542144775
}
```
If you have started it with a different port, replace `24080` with the port you chose.

Note, for the first translation, the respective models are downloaded. This might take some time. Consecutive calls will be faster.

## Programmatic Usage
- **Python:** [python_query_api.py](examples/python_query_api.py) - Sending requests with Python to the EasyNMT Docker API.
- **Vue.js:** [vue_js_frontend.html](examples/vue_js_frontend.html) Vue.js Code for our [demo](http://easynmt.net/demo/).

## Documentation

To get an overview of all REST API endpoints, with all possible parameters and their description, you open the following url: [http://localhost:24080/docs](http://localhost:24080/docs)

### Endpoints
The following endpoints with the GET method are defined (i.e. you can call them like `http://localhost:24080/name?param1=val1&param2=val2`)

```
/translate
    Translates the text to the given target language.
    :param text: Text that should be translated
    :param target_lang: Target language
    :param source_lang: Language of text. Optional, if empty: Automatic language detection
    :param beam_size: Beam size. Optional
    :param perform_sentence_splitting: Split longer documents into individual sentences for translation. Optional
    :return:  Returns a json with the translated text

/language_detection
    Detects the language for the provided text
    :param text: A single text for which we want to know the language
    :return: The detected language
    
/get_languages
    Returns the languages the model supports
    :param source_lang: Optional. Only return languages with this language as source
    :param target_lang: Optional. Only return languages with this language as target
    :return:
```

You can call the `/translate` and `/language_detection` also with a POST request, giving you the option to pass a list of multiple texts. Then all texts are translated and returned at once.

### Environment Variables
You can control the Docker image using various environment variables:
- *MAX_WORKERS_BACKEND*: Number of worker processes for the translation. Default: 1
- *MAX_WORKERS_FRONTEND*: Number of worker processes for language detection & model info. Default: 2
- *EASYNMT_MODEL*: Which EasyNMT Model to load. Default: opus-mt
- *EASYNMT_MODEL_ARGS*: Json encoded string with parameters when loading EasyNMT: Default: {}
- *EASYNMT_MAX_TEXT_LEN*: Maximal text length for translation. Default: Not set
- *EASYNMT_MAX_BEAM_SIZE*: Maximal beam size for translation. Default: Not set
- *EASYNMT_BATCH_SIZE*: Batch size for translation. Default: 16
- *TIMEOUT*: [Gunicorn timeout](https://docs.gunicorn.org/en/stable/settings.html#timeout). Default: 120

All model files are stored at `/cache/`. You can mount this path to your host machine if you want to re-use previously downloaded models.
