from typing import Dict, Union
from gliner import GLiNER
import gradio as gr

# Define available models
available_models = {
    "Model A": "jruepp/swiss_legal_GLiNER",
    "Model B": "urchade/gliner_multi_pii-v1",
    "Model C": "urchade/gliner_mediumv2.1",
}

# Invert the `available_models` dictionary for display-to-key mapping
display_to_key = {v: k for k, v in available_models.items()}

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
        """Participants à la procédure Leandro Choffat, représenté par Me André Gossin, avocat, recourant, contre Ministère public de la République et canton du Jura, Le Château, 2900 Porrentruy, intimé. Objet Ordonnance de classement; indemnité, recours contre la décision de la Chambre pénale des recours du Tribunal cantonal de la République et canton du Jura du 14 juillet 2022 ( CPR 60 + 61 / 2022 ). Faits: A. Le 9 octobre 2019, une instruction a été ouverte contre Leandro Choffat notamment pour contrainte, menaces, utilisation abusive d ' une installation de télécommunication et dommages à la propriété, au motif qu ' il aurait, entre le 31 août et le 8 octobre 2019, suivi quotidiennement Jeannette Vieira ( ci-après: la plaignante ) en voiture; il l ' aurait également suivie jusque devant chez un ami, devant chez elle et sur son lieu de travail, lui aurait envoyé de nombreux messages, aurait tenté de lui téléphoner à plusieurs reprises, lui aurait dit " si tu n ' arrives pas à payer tes factures, tu peux demander une avance à celui qui t ' a baisée tout le week-end " et aurait rayé son véhicule. Le 18 février 2021, la procédure pénale a été suspendue, la plaignante ayant donné son accord le 12 février 2021 à une suspension de la procédure pour une durée de six mois s ' agissant des """,
        "person,organization,location",
        0.5,
        False,
    ],
    [
        """Verfahrensbeteiligte 6B_313 / 2015 Ines Zingg, vertreten durch Rechtsanwalt Michael Felder, Beschwerdeführer, gegen Oberstaatsanwaltschaft des Kantons Zürich, Florhofgasse 2, 8090 Zürich, Beschwerdegegnerin, und 6B_271 / 2015 Oberstaatsanwaltschaft des Kantons Zürich, Florhofgasse 2, 8090 Zürich, Beschwerdeführerin, gegen Ines Zingg, vertreten durch Rechtsanwalt Michael Felder, Beschwerdegegner. Gegenstand 6B_313 / 2015 Totschlag ( Art. 113 StGB ), Mord ( Art. 112 StGB ), 6B_271 / 2015 Verminderung der Schuldfähigkeit; Willkür, Beschwerden gegen das Urteil des Obergerichts des Kantons Zürich, I. Strafkammer, vom 12. November 2014. Sachverhalt: A. Ines Zingg tötete Maliya Barone am 10. März 2012 mit mehreren Messerstichen. Zuvor hatte Ruth Schaffner, die damalige Partnerin von Ines Zingg, diesen mehrfach dazu gedrängt, an einem " Dreier " mit ihr und ihrem Bekannten Maliya Barone teilzunehmen. Dabei zeigte sie jedoch zunehmend weniger ( sexuelles ) Interesse an Ines Zingg, was seinerseits zu Frustrationen und Kränkungen führte. Die Beziehung zwischen Ines Zingg und Ruth Schaffner verschlechterte sich zusehends. Ruth Schaffner wollte diese schliesslich beenden. Über die Endphase der Beziehung bestehen unterschiedliche Auffassungen. Jedenfalls forderte Ruth Schaffner Ines Zingg auf, aus ihrer Wohnung auszuziehen. Zu diesem Zweck ersuchte sie die lokale Polizei sowie eine Anwältin um Rat. Ines Zingg bat Ruth Schaffner, länger bleiben zu dürfen. Diese willigte ein, was zu einem Hin und Her führte.""",
        "person,organization,location",
        0.5,
        False,
    ],
    [
        """Partecipanti al procedimento Miriam Giani, ricorrente, contro Dipartimento delle istituzioni del Cantone Ticino, Sezione della popolazione, 6500 Bellinzona, Sostituto del Giudice delle misure coercitive del Cantone Ticino, 6901 Lugano. Oggetto Carcerazione in vista di rinvio coatto, ricorso contro la sentenza emanata il 12 agosto 2015 dal Tribunale amministrativo del Cantone Ticino. Fatti: A. Miriam Giani ( 1976 ), cittadino del Mali, ha presentato una prima domanda d ' asilo in Svizzera nel 2012, la quale è stata respinta il 5 aprile 2013 dall ' allora Ufficio federale della migrazione ( diventato poi la Segreteria di Stato della migrazione SEM ), con una decisione di non entrata in materia. Una seconda domanda, depositata dopo che Miriam Giani ha cercato invano di ottenere l ' asilo in Germania nel 2013, è stata ugualmente evasa con una decisione di non entrata in materia il 24 gennaio 2014. Del suo allontanamento è stato incaricato il Cantone Ticino. Durante il suo soggiorno in Svizzera Miriam Giani è stato condannato a più riprese per infrazione e contravvenzione alla LStup ( RS 812. 121 ) e per infrazione alla LStr ( RS 142. 20 ). Arrestato, è stato detenuto per 221 giorni nel Canton Ginevra, fino al 1 ° luglio 2015. Interrogato dalla Polizia cantonale ticinese il 2 luglio seguente, egli ha dichiarato di non volere rientrare in Mali e di non disporre dei necessari documenti""",
        "person,organization,location",
        0.5,
        False
    ],
    [
        """Considérant en droit: 1. 1. 1. Si le dispositif d ' un arrêt du Tribunal fédéral est peu clair, incomplet ou équivoque, ou si ses éléments sont contradictoires entre eux ou avec les motifs, ou s ' il contient des erreurs de rédaction ou de calcul, le Tribunal fédéral, à la demande écrite d ' une partie ou d ' office, interprète ou rectifie l ' arrêt ( art. 129 al. 1 LTF ). L ' objet de la rectification est de permettre la correction des erreurs de rédaction ou de pures fautes de calcul dans le dispositif ( PIERRE FERRARI, Commentaire de la LTF, 2e éd, n. 6 ad art. 129 ). Outre des conclusions, la partie doit rendre plausible la nécessité de l ' interprétation ou de la rectification, à défaut de quoi la requête sera déclarée irrecevable ( op. cit., n. 7 ). 1. 2. En l ' espèce, la requérante n ' expose pas en quoi la rectification de l ' arrêt du Tribunal fédéral du 18 décembre 2017 qu ' elle sollicite serait nécessaire. De plus, comme la juridiction cantonale et le Tribunal fédéral ne sont pas entrés en matière sur les recours interjetés par A. _, il paraît douteux que l ' assureur requérant dispose d ' un intérêt digne de protection à obtenir une rectification de la désignation des parties figurant dans l ' arrêt précité ( par analogie, voir l ' art. 89 al. 1 let. b et c LTF ). """,
        "citation,law",
        0.5,
        False
    ],
    [
        """Erwägungen: 1. A. _ erstattete am 18. Dezember 2018 beim Untersuchungsamt Uznach Strafanzeige gegen die Regionalstellenleiterin des Konkursamtes Rapperswil - Jona, B. _, wegen " Hausfriedensbruchs, Missachtung der Freiheitsrechte insbesondere auf Eigentum ( minimales Inventar als Existenzminimum ) " etc. Diese Straftaten soll sie nach der Anzeige am 28. November 2018 bei Wohnungsräumungen im Konkursverfahen gegen ihn begangen haben. Am 1. März 2019 erteilte die Anklagekammer des Kantons St. Gallen die Ermächtigung zur Eröffnung eines Strafverfahrens gegen B. _ nicht. Mit Beschwerde vom 11. April 2019 beantragt A. _, B. _ angemessen zu bestrafen. Zudem sei ihm sämtliches Inventar zurückzugeben oder ein Schadenersatz von Fr. 62 ' 000. - - zu leisten. Vernehmlassungen wurden keine eingeholt. 2. """,
        "citation,law",
        0.5,
        False
    ],
    [
        """Diritto: 1. Il ricorso in materia di diritto pubblico può essere presentato per violazione del diritto, conformemente a quanto stabilito dagli art. 95 e 96 LTF. L ' accertamento dei fatti può venir censurato solo se è stato svolto in modo manifestamente inesatto o in violazione del diritto ai sensi dell ' art. 95 LTF e se l ' eliminazione del vizio può essere determinante per l ' esito del procedimento ( art. 97 cpv. 1 e 105 cpv. 1 e 2 LTF ). Se, tuttavia, il ricorso è presentato contro una decisione d ' assegnazione o rifiuto di prestazioni pecuniarie dell ' assicurazione militare o dell ' assicurazione contro gli infortuni - come nel caso concreto - può essere censurato qualsiasi accertamento inesatto o incompleto dei fatti giuridicamente rilevanti ( art. 97 cpv. 2 LTF ); il Tribunale federale in tal caso non è vincolato dall ' accertamento dei fatti operato dall ' autorità inferiore ( art. 105 cpv. 3 LTF ). 2. Oggetto del contendere è la questione se la Corte cantonale abbia a ragione confermato l ' entità della rendita di invalidità di cui beneficia l ' assicurato. La IMI già in sede cantonale non era più contestata. 3.""",
        "citation,law",
        0.5,
        False
    ],
    [
        """A blog about weather, climate, and science. Occasionally, we ' ll post on other topics of interest. Wednesday, February 29, 2012 Destructive Tornado and Severe Weather Event Possible This is the probabilistic tornado and severe thunderstorm ( latter defined as hail ≥ 1 " in diameter and / or winds ≥ 58 mph ) outlook valid from 6am Friday until 6am Saturday as forecast by the National Weather Service ' s Storm Prediction Center. Two things ratchet up my concern when tornadoes and severe storms are contemplated: Overnight Out of Season While very early March is technically " in season, " a number of these areas haven ' t had any severe weather in 2012. Plus, the dynamics ( jet stream strength, etc. ) of this event may keep it going during the overnight hours with fast-moving ( potentially less " lead time " ) storms. Breaking it down: the hatched area ( in this case ) means tornadoes ≥ F-2 intensity and / or thunderstorm-generated winds ≥ 75 mph. This far ( more than 24 hours ) out, those are very high probabilities ( 45 % ), as well. So, if you live in these areas, I urge you to make sure you to conduct the following reviews: 6 comments: Looks like we ' re right in the bullseye for tomorrow in Louisville Metro LMK - ( SDF ). """,
        "meteorological agency, abbreviation, date, location",
        0.5,
        False
    ],
    [
        """SINGAPORE - A police report has been made regarding videos taken of Prime Minister Lee Hsien Loong ' s son, Mr Li Yipeng, who was offered a ride in a private car, said the police in a statement on Sunday ( March 17 ). The car was driven by a 31-year-old Singaporean man, added the police. It is believed the vehicle was not a private-hire car. In the videos, the man was heard asking Mr Li repeatedly to confirm his identity, his home address and also his security arrangements. The videos were taken without Mr Li ' s knowledge or permission, said the statement. The police said that they are " looking into the matter with the assistance of the driver because the nature of the questions raises serious security concerns, given Mr Li ' s background ". The police said that the 31-year-old man was previously convicted of taking a vehicle without the owner ' s consent under the Road Traffic Act. An offence of driving without valid insurance under the Act was taken into consideration during sentencing in 2014. Separately, it is understood that the man was previously involved in a theft-in-dwelling case and was given a warning by the police. There was also a police report lodged against him for criminal intimidation. Mr Li, 36, who has Asperger ' s syndrome, is the second of PM Lee ' s four children.""",
        "date, organization, politician, location, person, document"
    ],
    [
        """Herr Abdulrahman Abdullahi aus Allada in Benin hat heute Abend im Casino ' Goldene Krone ' in Cotonou 50. 000 CFA-Francs gewonnen. Er hat sich für das Jackpotspiel ' Mega Fortune ' angemeldet und hat mit den Zahlen 3, 11, 13, 23 und 47 gewonnen. Seine persönlichen Daten für den Auszahlungsbescheid sind: Vorname Abdulrahman, Nachname Abdullahi, Adresse 34 Rue de la Liberté 123, Postleitzahl 01 BP 3456, Telefonnummer + 229 66 77 88 99 und E-Mail-Adresse abdullahi. abdulrahman @ gmail. com. Seine persönliche Identifikationsnummer für den Casino-Kontoauszug ist: 12345678901234567890. """,
        "postal code, casino, person, casino account number, phone number, email, address",
        0.5,
        False
    ],
    [
        """Mamadou Diop, né le 25 / 11 / 1990 à Dakar, est patiente du service de gynécologie-obstétrique de l ' hôpital Président-Kennedy. Son numéro d ' assurance maladie est A123456789. Sa dernière consultation a eu lieu le 12 / 02 / 2023. Ses antécédents médicaux comprennent une allergie aux antibiotiques du groupe des pénicillines. Son dernier examen clinique a révélé une grossesse de 28 semaines. Sa prochaine rendez-vous est prévu le 15 / 03 / 2023.""",
        "date,person,health insurance id number,pregnancy status, weeks",
        0.5,
        False
    ]

]

# Load selected models
def load_models(model_a_name, model_b_name):
    model_a = GLiNER.from_pretrained(available_models[model_a_name])
    model_b = GLiNER.from_pretrained(available_models[model_b_name])
    return {"Model A": model_a, "Model B": model_b}

# Define the NER function
def ner(
    text: str, labels: str, threshold: float, nested_ner: bool, model_name: str, loaded_models: Dict
) -> Dict[str, Union[str, int, float]]:
    labels = labels.split(",")
    model = loaded_models[model_name]
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

# Build the Gradio Interface
with gr.Blocks(title="Model Comparison Demo") as demo:
    gr.Markdown("## Compare Two Preloaded Models")

    # Text input
    input_text = gr.Textbox(
        value=examples[0][0],
        label="Text Input",
        placeholder="Enter your text here",
    )
    labels = gr.Textbox(
        value=examples[0][1],
        label="Labels",
        placeholder="Enter your labels here (comma separated)",
    )
    threshold = gr.Slider(
        0, 1, value=0.5, step=0.01, label="Threshold"
    )
    nested_ner = gr.Checkbox(value=False, label="Nested NER")

    # Dropdowns for model selection
    model_select_a = gr.Dropdown(
        choices=list(display_to_key.keys()),  # Display actual model names
        value=list(display_to_key.keys())[0],  # Default to the first model name
        label="Select Model A",
    )
    model_select_b = gr.Dropdown(
        choices=list(display_to_key.keys()),  # Display actual model names
        value=list(display_to_key.keys())[1],  # Default to the second model name
        label="Select Model B",
    )

    # Load models when selected
    load_btn = gr.Button("Load Selected Models")

    # Outputs for each model
    output_a = gr.HighlightedText(label="Model A Predictions")
    output_b = gr.HighlightedText(label="Model B Predictions")

    # Variables to hold loaded models
    loaded_models = gr.State(value={})

    def set_loaded_models(model_a_name, model_b_name):
        model_a_key = display_to_key[model_a_name]
        model_b_key = display_to_key[model_b_name]
        return load_models(model_a_key, model_b_key)

    load_btn.click(
        fn=set_loaded_models,
        inputs=[model_select_a, model_select_b],
        outputs=loaded_models,
    )

    # Model A evaluation
    eval_btn_a = gr.Button("Evaluate Model A")
    eval_btn_a.click(
        fn=ner,
        inputs=[input_text, labels, threshold, nested_ner, gr.State(value="Model A"), loaded_models],
        outputs=output_a,
    )

    # Model B evaluation
    eval_btn_b = gr.Button("Evaluate Model B")
    eval_btn_b.click(
        fn=ner,
        inputs=[input_text, labels, threshold, nested_ner,gr.State(value="Model B"), loaded_models],
        outputs=output_b,
    )

    # Examples widget
    gr.Examples(
        examples,
        inputs=[input_text, labels],
    )

demo.launch()
