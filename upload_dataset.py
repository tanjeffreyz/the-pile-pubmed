import pandas as pd
from datasets import Dataset


DATA_PATH = '/data/pubmed/PUBMED_title_abstracts_2020_baseline.jsonl'
CHUNK_SIZE = 1024


def generator():
    with open(DATA_PATH, 'r') as file:
        with pd.read_json(file, lines=True, chunksize=CHUNK_SIZE) as reader:
            for chunk in reader:
                for record in chunk.to_dict('records'):
                    yield record


dataset = Dataset.from_generator(generator)
dataset.push_to_hub('LLM-PBE/the-pile-pubmed', private=True)
