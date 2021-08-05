# EasyNMT - Easy to use, state-of-the-art Neural Machine Translation
This package provides easy to use, state-of-the-art machine translation for more than 100+ languages. The highlights of this package are:

- Easy installation and usage: Use state-of-the-art machine translation with 3 lines of code
- Automatic download of pre-trained machine translation models
- Translation between 150+ languages
- Automatic language detection for 170+ languages
- Sentence and document translation
- Multi-GPU and multi-process translation

At the moment, we provide the following models:
 - [Opus-MT](#Opus-MT) from [Helsinki-NLP](https://github.com/Helsinki-NLP/Opus-MT), supporting 1200+ translation directions for 150+ languages.
  - [mBART50_m2m](#mBART_50) from [Facebook Research](https://arxiv.org/abs/2008.00401), supporting translation between any direction for 50+ languages.
  - [M2M_100](#M2M_100) from [Facebook Research](https://arxiv.org/abs/2010.11125), supporting translation between any direction for 100+ languages.


**Examples:**
 - [EasyNMT Google Colab Example](https://colab.research.google.com/drive/1X47vgSiOphpxS5w_LPtjQgJmiSTNfRNC?usp=sharing) - Step-by-step example how to use EasyNMT with Python.
 - [EasyNMT Opus-MT Online Demo](http://easynmt.net/demo) - Demo to test the translation quality of the Opus-MT model.
- [EasyNMT Google Colab REST API Hosting](https://colab.research.google.com/drive/1kAh_Vt1ipA5-BuoaPX39rCIHFrhpcRpW?usp=sharing) - Example how to host a translation REST API on Google Colab and using the free GPU.


## Docker & REST-API

We provide ready-to-use Docker images, that wrap EasyNMT in a REST API:
```
docker run -p 24080:80 easynmt/api:2.0-cpu
```

Calling the REST API:
```
http://localhost:24080/translate?target_lang=en&text=Hallo%20Welt
```

See [docker/](docker/) for more information on the different Docker images and the REST API endpoints.

Also check our [EasyNMT Google Colab REST API Hosting](https://colab.research.google.com/drive/1kAh_Vt1ipA5-BuoaPX39rCIHFrhpcRpW?usp=sharing) example, on how to use Google Colab and the free GPU to host a translation API.

## Installation for Python
You can install the package via:

```
pip install -U easynmt
```

The models are based on **PyTorch**. If you have a GPU available, see how to install  **[PyTorch with GPU support](https://pytorch.org/get-started/locally/)**. If you use Windows and have issues with the installation, see [this issue](https://github.com/UKPLab/EasyNMT/issues/3) how to solve it.

## Usage
The usage is simple:
```python
from easynmt import EasyNMT
model = EasyNMT('opus-mt')

#Translate a single sentence to German
print(model.translate('This is a sentence we want to translate to German', target_lang='de'))

#Translate several sentences to German
sentences = ['You can define a list with sentences.',
             'All sentences are translated to your target language.',
             'Note, you could also mix the languages of the sentences.']
print(model.translate(sentences, target_lang='de'))
```

### Document Translation
The available models are based on the Transformer architecture, which provide state-of-the-art translation quality. However, the input length is limited to 512 word pieces for the *opus-mt* model and 1024 word pieces for the *M2M* models. 

The `translate()` performs automatic sentence splitting to be able to translate also longer documents:

```python
from easynmt import EasyNMT
model = EasyNMT('opus-mt')

document = """Berlin is the capital and largest city of Germany by both area and population.[6][7] Its 3,769,495 inhabitants as of 31 December 2019[2] make it the most-populous city of the European Union, according to population within city limits.[8] The city is also one of Germany's 16 federal states. It is surrounded by the state of Brandenburg, and contiguous with Potsdam, Brandenburg's capital. The two cities are at the center of the Berlin-Brandenburg capital region, which is, with about six million inhabitants and an area of more than 30,000 km2,[9] Germany's third-largest metropolitan region after the Rhine-Ruhr and Rhine-Main regions. Berlin straddles the banks of the River Spree, which flows into the River Havel (a tributary of the River Elbe) in the western borough of Spandau. Among the city's main topographical features are the many lakes in the western and southeastern boroughs formed by the Spree, Havel, and Dahme rivers (the largest of which is Lake Müggelsee). Due to its location in the European Plain, Berlin is influenced by a temperate seasonal climate. About one-third of the city's area is composed of forests, parks, gardens, rivers, canals and lakes.[10] The city lies in the Central German dialect area, the Berlin dialect being a variant of the Lusatian-New Marchian dialects.

First documented in the 13th century and at the crossing of two important historic trade routes,[11] Berlin became the capital of the Margraviate of Brandenburg (1417–1701), the Kingdom of Prussia (1701–1918), the German Empire (1871–1918), the Weimar Republic (1919–1933), and the Third Reich (1933–1945).[12] Berlin in the 1920s was the third-largest municipality in the world.[13] After World War II and its subsequent occupation by the victorious countries, the city was divided; West Berlin became a de facto West German exclave, surrounded by the Berlin Wall (1961–1989) and East German territory.[14] East Berlin was declared capital of East Germany, while Bonn became the West German capital. Following German reunification in 1990, Berlin once again became the capital of all of Germany.

Berlin is a world city of culture, politics, media and science.[15][16][17][18] Its economy is based on high-tech firms and the service sector, encompassing a diverse range of creative industries, research facilities, media corporations and convention venues.[19][20] Berlin serves as a continental hub for air and rail traffic and has a highly complex public transportation network. The metropolis is a popular tourist destination.[21] Significant industries also include IT, pharmaceuticals, biomedical engineering, clean tech, biotechnology, construction and electronics."""

#Translate the document to German
print(model.translate(document, target_lang='de'))
```

The function breaks down the document into sentences and then translates the sentences individually using the specified model.

### Automatic Language Detection
You can set the `source_lang` for the `translate` method to define the source language. If `source_lang` is not set, [fastText](https://fasttext.cc/blog/2017/10/02/blog-post.html) will be used to automatically determine the source language. This also allows you to provide a list with sentences / documents that have various languages:
 
```python
from easynmt import EasyNMT
model = EasyNMT('opus-mt')

#Translate several sentences to English
sentences = ['Dies ist ein Satz in Deutsch.',   #This is a German sentence
             '这是一个中文句子',    #This is a chinese sentence
             'Esta es una oración en español.'] #This is a spanish sentence
print(model.translate(sentences, target_lang='en'))
```




# Available Models

The following models are currently available. They provide translations between 150+ languages.

| Model | Reference | #Languages | Size | Speed GPU (Sentences/Sec on V100) | Speed CPU (Sentences/Sec) | Comment |
| --- | --- | :---: | :---: | :---: | :---: | --- |
| opus-mt | [Helsinki-NLP](https://github.com/Helsinki-NLP/Opus-MT) | 186 | 300 MB | 50 | 6 | Inidivudal models  (~300 MB) per translation direction
| mbart50_m2m | [Facebook Research](https://github.com/pytorch/fairseq/tree/master/examples/multilingual) | 52 |  2.3 GB | 25  | - | 
| mbart50_m2en | [Facebook Research](https://github.com/pytorch/fairseq/tree/master/examples/multilingual) | 52 |  2.3 GB | 25  | - | Can only translate from the other languages to English. 
| mbart50_en2m | [Facebook Research](https://github.com/pytorch/fairseq/tree/master/examples/multilingual) | 52 |  2.3 GB | 25  | - | Can only translate from English to the other languages. 
| m2m_100_418M | [Facebook Research](https://github.com/pytorch/fairseq/tree/master/examples/m2m_100) | 100 | 1.8 GB | 22 | - | 
| m2m_100_1.2B | [Facebook Research](https://github.com/pytorch/fairseq/tree/master/examples/m2m_100) | 100 | 5.0 GB | 13 | - | 

## Translation Quality

Comparing model translation quality will be added soon here. So far, my personal subjective impression is, that *opus-mt* and *m2m_100_1.2B* yield the best translations.

## Opus-MT
We provide a wrapper for the [pre-trained models](https://github.com/Helsinki-NLP/OPUS-MT-train/tree/master/models) from [Opus-MT](https://github.com/Helsinki-NLP/OPUS-MT-train).

Opus-MT provides 1200+ different translation models, each capable to translate one direction (e.g. from German to English). Each model is about 300 MB of size. 

**Supported languages:** aav, aed, af, alv, am, ar, art, ase, az, bat, bcl, be, bem, ber, bg, bi, bn, bnt, bzs, ca, cau, ccs, ceb, cel, chk, cpf, crs, cs, csg, csn, cus, cy, da, de, dra, ee, efi, el, en, eo, es, et, eu, euq, fi, fj, fr, fse, ga, gaa, gil, gl, grk, guw, gv, ha, he, hi, hil, ho, hr, ht, hu, hy, id, ig, ilo, is, iso, it, ja, jap, ka, kab, kg, kj, kl, ko, kqn, kwn, kwy, lg, ln, loz, lt, lu, lua, lue, lun, luo, lus, lv, map, mfe, mfs, mg, mh, mk, mkh, ml, mos, mr, ms, mt, mul, ng, nic, niu, nl, no, nso, ny, nyk, om, pa, pag, pap, phi, pis, pl, pon, poz, pqe, pqw, prl, pt, rn, rnd, ro, roa, ru, run, rw, sal, sg, sh, sit, sk, sl, sm, sn, sq, srn, ss, ssp, st, sv, sw, swc, taw, tdt, th, ti, tiv, tl, tll, tn, to, toi, tpi, tr, trk, ts, tum, tut, tvl, tw, ty, tzo, uk, umb, ur, ve, vi, vsl, wa, wal, war, wls, xh, yap, yo, yua, zai, zh, zne

**Usage:**
```python
from easynmt import EasyNMT
model = EasyNMT('opus-mt', max_loaded_models=10)
```

The system will automatically detect the suitable Opus-MT model and load it. With the optional parameter `max_loaded_models` you can specify the maximal number of models that are simoultanously loaded. If you then translate with an unseen language direction, the oldest model is unloaded and the new model is loaded.

## mBERT_50

We provide a wrapper for the [mBART50](https://arxiv.org/abs/2008.00401) model from Facebook, that is able to translate between any pair of 50+ languages. There are also models available to translate from English to these languages or vice versa.




**Usage:**
```python
from easynmt import EasyNMT
model = EasyNMT('mbart50_m2m')
```

**Supported languages**: af, ar, az, bn, cs, de, en, es, et, fa, fi, fr, gl, gu, he, hi, hr, id, it, ja, ka, kk, km, ko, lt, lv, mk, ml, mn, mr, my, ne, nl, pl, ps, pt, ro, ru, si, sl, sv, sw, ta, te, th, tl, tr, uk, ur, vi, xh, zh  

## M2M_100
We provide a wrapper for the [M2M 100](https://arxiv.org/abs/2010.11125) model from Facebook, that is able to translate between any pair of 100 languages.



**Supported languages**: af, am, ar, ast, az, ba, be, bg, bn, br, bs, ca, ceb, cs, cy, da, de, el, en, es, et, fa, ff, fi, fr, fy, ga, gd, gl, gu, ha, he, hi, hr, ht, hu, hy, id, ig, ilo, is, it, ja, jv, ka, kk, km, kn, ko, lb, lg, ln, lo, lt, lv, mg, mk, ml, mn, mr, ms, my, ne, nl, no, ns, oc, or, pa, pl, ps, pt, ro, ru, sd, si, sk, sl, so, sq, sr, ss, su, sv, sw, ta, th, tl, tn, tr, uk, ur, uz, vi, wo, xh, yi, yo, zh, zu



As the moment, we provide wrapper for two M2M 100 models:
- **m2m_100_418M**: M2M model with 418 million parameters (1.8 GB)
- **m2m_100_1.2B**: M2M model with 1.2 billion parameters (5.0 GB)

**Usage:**
```python
from easynmt import EasyNMT
model = EasyNMT('m2m_100_418M')   #or: EasyNMT('m2m_100_1.2B') 
```

You can find more information [here](https://github.com/pytorch/fairseq/tree/master/examples/m2m_100). Note: the 12 billion M2M parameters model is currently not supported. 

As soon as you call `EasyNMT('m2m_100_418M')` / `EasyNMT('m2m_100_1.2B')`, the respective model is downloaded and cached locally. 


## Author

Contact person: [Nils Reimers](https://www.nils-reimers.de); [info@nils-reimers.de](mailto:info@nils-reimers.de)

https://www.ukp.tu-darmstadt.de/


Don't hesitate to send us an e-mail or report an issue, if something is broken (and it shouldn't be) or if you have further questions.

> This repository contains experimental software to encourage future research.
