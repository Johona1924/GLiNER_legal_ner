from typing import Dict, Union
from gliner import GLiNER
import gradio as gr

model = GLiNER.from_pretrained("urchade/gliner_multi_pii-v1")


examples = [
    [
        "Libretto von Marius Petipa, basierend auf der Novelle von 1822 ``Trilby, ou Le Lutin d'Argail`` von Charles Nodier, erstmals präsentiert vom Ballett des Moskauer Kaiserlichen Bolschoi-Theaters am 25. Januar/6. Februar (Julianisch/Gregorianisch), 1870, in Moskau mit Polina Karpakova als Trilby und Ludiia Geiten als Miranda und neu inszeniert von Petipa für das Kaiserliche Ballett im Kaiserlichen Bolschoi-Kamenny-Theater am 17.–29. Januar 1871 in St. Petersburg mit Adèle Grantzow als Trilby und Lev Ivanov als Graf Leopold.",
        "person, book, location, date, actor, character",
        0.5,
        True,
    ],
    [
        """
* Data Scientist, Data Analyst, or Data Engineer with 1+ years of experience.
* Experience with technologies such as Docker, Kubernetes, or Kubeflow
* Machine Learning experience preferred
* Experience with programming languages such as Python, C++, or SQL preferred
* Experience with technologies such as Databricks, Qlik, TensorFlow, PyTorch, Python, Dash, Pandas, or NumPy preferred
* BA or BS degree
* Active Secret OR Active Top Secret or Active TS/SCI clearance
""",
        "software package, programming language, software tool, degree, job title",
        0.5,
        False,
    ],
    [
        "Cependant, les deux modèles manquent d'autres symptômes fréquents de la DM, notamment l'atrophie dépendante du type de fibre, la myotonie, la cataracte et l'infertilité masculine.",
        "disease, symptom",
        0.5,
        False,
    ],
    [
        "Synergie entre les voies de transduction du signal est obligatoire pour l'expression de c-fos dans les lignées cellulaires B et T : implication pour le contrôle de c-fos via l'immunoglobuline de surface et les récepteurs d'antigènes des cellules T.",
        "DNA, RNA, cell line, cell type, protein",
        0.5,
        False,
    ],
    [
        "La scelta dei moduli di codifica e decodifica di dnpg può essere piuttosto flessibile, ad esempio reti di memoria a lungo termine (lstm) o reti neurali convoluzionali (cnn).",
        "short acronym, long acronym",
        0.5,
        False,
    ],
    [
        "Amelia Earhart flew her single-engine Lockheed Vega 5B across the Atlantic to Paris.",
        "person, company, location, airplane",
        0.5,
        True,
    ],
    [
        "Feldman contribue aux programmes ``State of the Revs`` et ``Revolution Postgame Live`` de NBC Sports Boston, ainsi qu'à 98.5 the SportsHub, la couverture MLS de SiriusXM FC et à d'autres radios et podcasts de la Nouvelle-Angleterre et à l'échelle nationale.",
        "person, company, location",
        0.5,
        False,
    ],
    [
        "Il 25 luglio 1948, nel 39° anniversario della traversata della Manica di Blériot, il Type 618 Nene-Viking volò da Heathrow a Parigi (Villacoublay) al mattino trasportando lettere alla vedova e al figlio di Blériot (segretario della FAI), che lo accolsero all'aeroporto.",
        "date, location, person, organization",
        0.5,
        False,
    ],
    [
        "Leo & Ian won the 1962 Bathurst Six Hour Classic at Mount Panorama driving a Daimler SP250 sports car, (that year the 500-mile race for touring cars was held at Phillip Island).",
        "person, date, location, organization, competition",
        0.5,
        False,
    ],
    [
        "Die Shore Line-Route der CNS&M diente bis 1955 von Süden nach Norden den Gemeinden Illinois: Chicago, Evanston, Wilmette, Kenilworth, Winnetka, Glencoe, Highland Park, Highwood, Fort Sheridan, Lake Forest, Lake Bluff, North Chicago, Waukegan, Zion und Winthrop Harbor sowie Kenosha, Racine und Milwaukee (das ``KRM'') in Wisconsin.",
        "location, organization, date",
        0.5,
        False,
    ],
    [
        "La comète C/2006 M4 (SWAN) est une comète non périodique découverte fin juin 2006 par Robert D. Matson d'Irvine, en Californie, et Michael Mattiazzo d'Adélaïde, en Australie-Méridionale, dans des images publiquement disponibles de l'Observatoire solaire et héliosphérique (SOHO).",
        "person, organization, date, location",
        0.5,
        False,
    ],
    [
        "Du 29 novembre 2011 au 31 mars 2012, Karimloo est retourné dans ``Les Misérables`` pour jouer le rôle principal de Jean Valjean au Queen's Theatre, à Londres, pour lequel il a remporté le Theatregoers' Choice Award 2013 du meilleur remplacement dans un rôle.",
        "person, actor, award, date, location",
        0.5,
        False,
    ],
    [
        "Una clinica di salute a Mexicali, sostenuta dall'ex candidato governatore della Baja California Enrique Acosta Fregoso (PRI), è stata chiusa il 15 giugno dopo aver venduto una presunta ``cura'' per COVID-19 per tra 10.000 e 50.000 pesos messicani.",
        "location, organization, person, date, currency",
        0.5,
        False,
    ],
    [
        "Erbaut 1793, war es das Zuhause von Mary Young Pickersgill, als sie 1806 nach Baltimore zog, und der Ort, an dem sie später das ``Star Spangled Banner'' nähte, 1813, die riesige überdimensionierte Garnisonsflagge, die im Sommer 1814 über Fort McHenry in Whetstone Point im Baltimore Harbor während des Angriffs der britischen Royal Navy in der Schlacht von Baltimore im Krieg von 1812 wehte.",
        "date, person, location, organization, event, flag",
        0.5,
        False,
    ],
]


def ner(
    text, labels: str, threshold: float, nested_ner: bool
) -> Dict[str, Union[str, int, float]]:
    labels = labels.split(",")
    return {
        "text": text,
        "entities": [
            {
                "entity": entity["label"],
                "word": entity["text"],
                "start": entity["start"],
                "end": entity["end"],
                "score": 0,
            }
            for entity in model.predict_entities(
                text, labels, flat_ner=not nested_ner, threshold=threshold
            )
        ],
    }


with gr.Blocks(title="GLiNER-M-v2.1") as demo:
    gr.Markdown(
        """
        # GLiNER-base
        GLiNER is a Named Entity Recognition (NER) model capable of identifying any entity type using a bidirectional transformer encoder (BERT-like). It provides a practical alternative to traditional NER models, which are limited to predefined entities, and Large Language Models (LLMs) that, despite their flexibility, are costly and large for resource-constrained scenarios.
        ## Links
        * Model: https://huggingface.co/urchade/gliner_multi-v2.1
        * All GLiNER models: https://huggingface.co/models?library=gliner
        * Paper: https://arxiv.org/abs/2311.08526
        * Repository: https://github.com/urchade/GLiNER
        """
    )
    with gr.Accordion("How to run this model locally", open=False):
        gr.Markdown(
            """
            ## Installation
            To use this model, you must install the GLiNER Python library:
            ```
            !pip install gliner
            ```
         
            ## Usage
            Once you've downloaded the GLiNER library, you can import the GLiNER class. You can then load this model using `GLiNER.from_pretrained` and predict entities with `predict_entities`.
            """
        )
        gr.Code(
            '''
from gliner import GLiNER
model = GLiNER.from_pretrained("urchade/gliner_mediumv2.1")
text = """
Cristiano Ronaldo dos Santos Aveiro (Portuguese pronunciation: [kɾiʃˈtjɐnu ʁɔˈnaldu]; born 5 February 1985) is a Portuguese professional footballer who plays as a forward for and captains both Saudi Pro League club Al Nassr and the Portugal national team. Widely regarded as one of the greatest players of all time, Ronaldo has won five Ballon d'Or awards,[note 3] a record three UEFA Men's Player of the Year Awards, and four European Golden Shoes, the most by a European player. He has won 33 trophies in his career, including seven league titles, five UEFA Champions Leagues, the UEFA European Championship and the UEFA Nations League. Ronaldo holds the records for most appearances (183), goals (140) and assists (42) in the Champions League, goals in the European Championship (14), international goals (128) and international appearances (205). He is one of the few players to have made over 1,200 professional career appearances, the most by an outfield player, and has scored over 850 official senior career goals for club and country, making him the top goalscorer of all time.
"""
labels = ["person", "award", "date", "competitions", "teams"]
entities = model.predict_entities(text, labels)
for entity in entities:
    print(entity["text"], "=>", entity["label"])
            ''',
            language="python",
        )
        gr.Code(
            """
Cristiano Ronaldo dos Santos Aveiro => person
5 February 1985 => date
Al Nassr => teams
Portugal national team => teams
Ballon d'Or => award
UEFA Men's Player of the Year Awards => award
European Golden Shoes => award
UEFA Champions Leagues => competitions
UEFA European Championship => competitions
UEFA Nations League => competitions
Champions League => competitions
European Championship => competitions
            """
        )

    input_text = gr.Textbox(
        value=examples[0][0], label="Text input", placeholder="Enter your text here"
    )
    with gr.Row() as row:
        labels = gr.Textbox(
            value=examples[0][1],
            label="Labels",
            placeholder="Enter your labels here (comma separated)",
            scale=2,
        )
        threshold = gr.Slider(
            0,
            1,
            value=0.3,
            step=0.01,
            label="Threshold",
            info="Lower the threshold to increase how many entities get predicted.",
            scale=1,
        )
        nested_ner = gr.Checkbox(
            value=examples[0][2],
            label="Nested NER",
            info="Allow for nested NER?",
            scale=0,
        )
    output = gr.HighlightedText(label="Predicted Entities")
    submit_btn = gr.Button("Submit")
    examples = gr.Examples(
        examples,
        fn=ner,
        inputs=[input_text, labels, threshold, nested_ner],
        outputs=output,
        cache_examples=True,
    )

    # Submitting
    input_text.submit(
        fn=ner, inputs=[input_text, labels, threshold, nested_ner], outputs=output
    )
    labels.submit(
        fn=ner, inputs=[input_text, labels, threshold, nested_ner], outputs=output
    )
    threshold.release(
        fn=ner, inputs=[input_text, labels, threshold, nested_ner], outputs=output
    )
    submit_btn.click(
        fn=ner, inputs=[input_text, labels, threshold, nested_ner], outputs=output
    )
    nested_ner.change(
        fn=ner, inputs=[input_text, labels, threshold, nested_ner], outputs=output
    )

demo.queue()
demo.launch()
