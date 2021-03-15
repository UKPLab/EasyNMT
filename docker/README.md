# Docker

We provide a [Docker](https://www.docker.com/) based REST-API for EasyNMT: Send a query with your source text, and the API returns the translated text.

## Setup

To start the EasyNMT REST API on port `24080`run the following docker command:
```
docker run -p 24080:80 easynmt/api:1.1.0-cpu
```

This uses the CPU image. If you have GPU (CUDA), there are various GPU images available. Have a look at our [Docker Hub Page](https://hub.docker.com/repository/docker/easynmt/api/tags?page=1&ordering=last_updated).


## Usage

After you started the Docker image, you can visit: [http://localhost:8000/translate?target_lang=en&text=Hallo%20Welt](http://localhost:8000/translate?target_lang=en&text=Hallo%20Welt)

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
    "translation_time": 22.163145542144775
}
```

Note, for the first translation, the respective models are downloaded. This might take some time. Consecutive calls will be faster.

The endpoi

## Documentation

To get an overview of all REST API endpoints, with all possible parameters and their description, you open the following url:

```
http://localhost:24080/docs
```

If you have started it with a different port, replace `24080` with the port you chose.