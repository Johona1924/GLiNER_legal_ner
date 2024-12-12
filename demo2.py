from typing import Dict, Union
from gliner import GLiNER
import gradio as gr

# Define the list of accepted models
models = {
    "Model A": "jruepp/swiss_legal_GLiNER",
    "Model B": "urchade/gliner_multi_pii-v1",
}

def load_model(model_name):
    return GLiNER.from_pretrained(models[model_name])

# Load default models
model_a = load_model("Model A")
model_b = load_model("Model B")

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
    text, labels: str, threshold: float, nested_ner: bool, model_name: str
) -> Dict[str, Union[str, int, float]]:
    labels = labels.split(",")
    model = load_model(model_name)
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
    gr.Markdown("### Evaluate GLiNER Models")
    
    # Evaluation mode: single or comparison
    eval_mode = gr.Radio(
        choices=["Single Model", "Compare Two Models"],
        value="Single Model",
        label="Evaluation Mode"
    )
    
    # Inputs shared by both modes
    input_text = gr.Textbox(
        value=examples[0][0],
        label="Text Input",
        placeholder="Enter your text here"
    )
    labels = gr.Textbox(
        value=examples[0][1],
        label="Labels",
        placeholder="Enter your labels here (comma separated)"
    )
    threshold = gr.Slider(
        0,
        1,
        value=0.5,
        step=0.01,
        label="Threshold"
    )
    nested_ner = gr.Checkbox(
        value=examples[0][2],
        label="Nested NER",
    )
    
    # Single model selection
    with gr.Row(visible=True) as single_model_row:
        model_select_single = gr.Dropdown(
            list(models.keys()), label="Select Model", value="Model A"
        )
        output_single = gr.HighlightedText(label="Model Predictions")
    
    # Two-model comparison
    with gr.Row(visible=False) as comparison_row:
        model_select_a = gr.Dropdown(
            list(models.keys()), label="Select Model A", value="Model A"
        )
        model_select_b = gr.Dropdown(
            list(models.keys()), label="Select Model B", value="Model B"
        )
        output_a = gr.HighlightedText(label="Model A Predictions")
        output_b = gr.HighlightedText(label="Model B Predictions")
    
    # Toggle visibility based on evaluation mode
    def toggle_mode(eval_mode):
        return (
            eval_mode == "Single Model",  # Show single model row
            eval_mode == "Compare Two Models"  # Show comparison row
        )
    
    eval_mode.change(
        fn=toggle_mode,
        inputs=[eval_mode],
        outputs=[single_model_row, comparison_row]
    )
    
    # Single model evaluation
    def evaluate_single(text, labels, threshold, nested_ner, model_name):
        result = ner(text, labels, threshold, nested_ner, model_name)
        return result["entities"]
    
    eval_single_btn = gr.Button("Evaluate Single Model")
    eval_single_btn.click(
        fn=evaluate_single,
        inputs=[input_text, labels, threshold, nested_ner, model_select_single],
        outputs=output_single
    )
    
    # Two-model comparison
    def compare_models(text, labels, threshold, nested_ner, model_a, model_b):
        pred_a = ner(text, labels, threshold, nested_ner, model_a)
        pred_b = ner(text, labels, threshold, nested_ner, model_b)
        return pred_a["entities"], pred_b["entities"]
    
    compare_btn = gr.Button("Compare Models")
    compare_btn.click(
        fn=compare_models,
        inputs=[input_text, labels, threshold, nested_ner, model_select_a, model_select_b],
        outputs=[output_a, output_b]
    )
    
    # Add examples and pre-fill input fields
    examples_widget = gr.Examples(
        examples=examples,
        inputs=[input_text, labels, threshold, nested_ner],
    )

demo.launch()