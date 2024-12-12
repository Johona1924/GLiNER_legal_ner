# legal_ner

Quick File Descriptions:

- check_anon_data.ipynb: prepare the anonymization dataset by filtering some data points, and chunking the sequences.
- check_citation_data.ipynb: prepare the citation dataset by filtering some data points
- create_citation_extract.ipynb: create the citation dataset from rcds/swiss_citation_extraction
- finetune_anon.py: fine tune the anonymization model
- finetune_citation.py: fine tune the citation NER model
- test_anon.ipynb: test the anonymization model
- test_citation_extract.ipynb: test the citation NER model
- utils.py: utility functions
- explore_datasets.ipynb: explore length, labels, texts, languages, and statistics of datasets used for training/fine-tuning

TODO:
- as one can see above, there are a lot of duplicate files, should be deduplicated
- a lot of the utils code is duplicated in utils.py and in the notebooks, it should be always imported from utils.py
