"""


"""
import requests
import pprint
from requests.utils import quote

url = 'http://localhost:24080'


#####################################
# Translation
#####################################

texts = ["This is a list with my documents I want to translate. I can pass complete documents for translation.",
         "But I could also translate individual sentences.",
         "I can pass as many texts to the API as I want.",
         "Incluso puedo mezclar los idiomas que paso a la API de traducción."   #I can even mix the languages I pass to the translation API.
         ]

target_lang = 'de'

#Option 1: Individual GET requests for each document we want to translate
for text in texts:
    r = requests.get(url+"/translate?target_lang="+target_lang+"&text="+quote(text))
    translated_text = r.json()
    print(text)
    print("=>", translated_text)
    print("\n")


#Option 2: A POST requests that sends all texts at once
print("\nPost request that sends all sentences at once")
r = requests.post(url+"/translate", json={'target_lang': target_lang, 'text': texts})
response_data = r.json()
pprint.pprint(response_data, width=280)




#####################################
# Automatic language detection
#####################################

# Option 1: You can do a GET call and pass the individual sentences
print("\nGet requests for automatic language detection")
texts = ["Dies ist ein deutscher Satz", "This is an English sentence", "这是一个中文句子"]
for text in texts:
    r = requests.get(url+"/language_detection?text="+quote(text))
    print(text, "==>", r.json())


# Option 2: You do a POST call and pass the sentences as json:
r = requests.post(url+"/language_detection", json={'text': texts})
print("\nAll sentences sent via post endpoint:", r.json())


