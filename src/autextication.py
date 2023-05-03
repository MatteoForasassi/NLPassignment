import pandas as pd
from sklearn.model_selection import train_test_split
import os
from typing import Literal, List, Dict
from corpus import Corpus


class Autextication(Corpus):

    identifier: str = 'autextication'

    def __init__(self, split: Literal['train', 'validation'],
                 corpora_path: str):

        super().__init__(split, corpora_path)
        self.corpora_path = corpora_path
        self.corpus_path = os.path.join(self.corpora_path,'AUTEXTIFICATION/AUTEXTIFICATION/subtask_1/en/train.tsv')

        self.split = split
        self.data = List[Dict]

    def _split_data(self, data) -> List[Dict]:

        # split the src into train, test and validation sets
        train_data, val_data = train_test_split(data, test_size=0.20, random_state=42)
        train_data, test_data = train_test_split(train_data, test_size=0.20, random_state=42)
        if self.split == 'train': return train_data
        if self.split == 'validation': return val_data
        if self.split == 'test': return test_data

    def load_data(self) -> List[Dict]:

        text_list = []
        # Read the TSV file into a Pandas dataframe
        df = pd.read_csv(
            self.corpus_path,
            delimiter='\t')

        for index, row in df.iterrows():
            text = row['text']
            label = row['label']
            # do something with the text and label variables

            text_dict = {
                'text': text,
                'label': label
            }
            text_list.append(text_dict)

        return self._split_data(text_list)

    def get_identifier(self) -> str:
        return self.identifier






