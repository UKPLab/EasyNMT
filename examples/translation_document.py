documents = ["""Berlin is the capital and largest city of Germany by both area and population. Its 3,769,495 inhabitants as of 31 December 2019[2] make it the most-populous city of the European Union, according to population within city limits.[8] The city is also one of Germany's 16 federal states. It is surrounded by the state of Brandenburg, and contiguous with Potsdam, Brandenburg's capital. The two cities are at the center of the Berlin-Brandenburg capital region, which is, with about six million inhabitants and an area of more than 30,000 km2,[9] Germany's third-largest metropolitan region after the Rhine-Ruhr and Rhine-Main regions. Berlin straddles the banks of the River Spree, which flows into the River Havel (a tributary of the River Elbe) in the western borough of Spandau. Among the city's main topographical features are the many lakes in the western and southeastern boroughs formed by the Spree, Havel, and Dahme rivers (the largest of which is Lake Müggelsee). Due to its location in the European Plain, Berlin is influenced by a temperate seasonal climate. About one-third of the city's area is composed of forests, parks, gardens, rivers, canals and lakes.[10] The city lies in the Central German dialect area, the Berlin dialect being a variant of the Lusatian-New Marchian dialects.

First documented in the 13th century and at the crossing of two important historic trade routes,[11] Berlin became the capital of the Margraviate of Brandenburg (1417–1701), the Kingdom of Prussia (1701–1918), the German Empire (1871–1918), the Weimar Republic (1919–1933), and the Third Reich (1933–1945).[12] Berlin in the 1920s was the third-largest municipality in the world.[13] After World War II and its subsequent occupation by the victorious countries, the city was divided; West Berlin became a de facto West German exclave, surrounded by the Berlin Wall (1961–1989) and East German territory.[14] East Berlin was declared capital of East Germany, while Bonn became the West German capital. Following German reunification in 1990, Berlin once again became the capital of all of Germany.

Berlin is a world city of culture, politics, media and science.[15][16][17][18] Its economy is based on high-tech firms and the service sector, encompassing a diverse range of creative industries, research facilities, media corporations and convention venues.[19][20] Berlin serves as a continental hub for air and rail traffic and has a highly complex public transportation network. The metropolis is a popular tourist destination.[21] Significant industries also include IT, pharmaceuticals, biomedical engineering, clean tech, biotechnology, construction and electronics.""",
 """This is the second document. With some test sentences. 

This document contains also a list, which we want to translate.

This is a list:
    - Berlin is a city in Germany
    - Paris is the capital of France
    - London is a large city in England

This is my final sentence.""",
"""Mon document final est rédigé en français. Nous voulons donc vérifier si l'étape de détection de la langue fonctionne."""
             ]

from easynmt import EasyNMT


target_lang = 'de'      # We want to translate the sentences to German (de)

model = EasyNMT('opus-mt')

translations = model.translate(documents, target_lang=target_lang, batch_size=8, beam_size=3)

for doc in translations:
    print(doc)
    print("\n======\n")