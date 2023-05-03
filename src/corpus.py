from typing import Literal
from abc import ABC, abstractmethod


class Corpus(ABC):
    def __init__(
            self,
            split: Literal['train', 'validation', 'test'],
            corpora_path: str

    ):
        self.corpora_path = corpora_path
        self.split = split

    @property
    @abstractmethod
    def get_identifier(self) -> str:
        pass

    @abstractmethod
    def load_data(self) -> list:
        pass