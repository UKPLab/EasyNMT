"""
This example tests the translation with all available models.
"""
from easynmt import EasyNMT

available_models = ['opus-mt', 'mbart50_m2m', 'm2m_100_418M', 'm2m_100_1.2B']

for model_name in available_models:
    print("\n\nLoad model:", model_name)
    model = EasyNMT(model_name)

    sentences = ['In dieser Liste definieren wir mehrere Sätze.',
                 'Jeder dieser Sätze wird dann in die Zielsprache übersetzt.',
                 'Puede especificar en esta lista la oración en varios idiomas.',
                 'El sistema detectará automáticamente el idioma y utilizará el modelo correcto.']
    translations = model.translate(sentences, target_lang='en')

    print("Translations:")
    for sent, trans in zip(sentences, translations):
        print(sent)
        print("=>", trans, "\n")
