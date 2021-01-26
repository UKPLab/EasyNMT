"""
This example shows how EasyNMT to stream translations.

Streaming translations can be useful when you want to translate a large set of sentences / documents.
The method chunks the data, translates it, and returns the results.

This can be useful if you want to write it e.g. to a file.
"""
from easynmt import EasyNMT

#First, we create a large set of sentences:
sentences = ['This is sentence '+str(i) for i in range(10000)]

target_lang = 'de'      # We want to translate the sentences to German (de)

model = EasyNMT('opus-mt')


#The method model.translate_stream chunks the data into sets of size chunk_size
#It then translate these documents/sentences and yields the result
for translation in model.translate_stream(sentences, show_progress_bar=False, chunk_size=16, target_lang=target_lang):
    print(translation)

