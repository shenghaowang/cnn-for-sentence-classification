import importlib
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from loguru import logger
from omegaconf import DictConfig

# from loguru import logger
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

from preprocess.utils import Cols


class TextVectorizer:
    def __init__(self, model: str = "en_core_web_md"):
        """Create word vectors from given comments"""
        self.model = importlib.import_module(model).load()

    def tokenize(self, text: str) -> List[str]:
        """Tokenize a given sentence into a list of tokens

        Parameters
        ----------
        text : str
            sentence to tokenize

        Returns
        -------
        List[str]
            list of tokens generated from the sentence
        """
        doc = self.model.make_doc(text)
        tokens = [token for token in doc]

        return tokens

    def vectorize(self, text: str) -> List[np.ndarray]:
        """Given a sentence, tokenize it and returns a list of
        pre-trained word vector for each token.

        Parameters
        ----------
        text : str
            sentence to tokenize

        Returns
        -------
        List[np.ndarray]
            list of pretrained word vectors
        """
        doc = self.model.make_doc(text)
        word_vecs = [token.vector for token in doc]

        return word_vecs


class TextDataset(Dataset):
    """Creates an pytorch dataset to consume our pre-loaded text data
    Reference: https://pytorch.org/tutorials/beginner/basics/data_tutorial.html
    """

    def __init__(self, data: List[Tuple[str, int]], vectorizer: TextVectorizer):
        self.dataset = data
        self.vectorizer = vectorizer

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        text, label = self.dataset[idx]
        word_vecs = self.vectorizer.vectorize(text)

        return {
            "vectors": word_vecs,
            "label": label,
            "text": text,  # for debugging only
        }


class TextDataModule(pl.LightningDataModule):
    """LightningDataModule: Wrapper class for the dataset to be used in training"""

    def __init__(
        self,
        batch_size: int,
        max_seq_len: int,
        processed_data_dir: DictConfig,
        cols: Cols,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.processed_data_dir = processed_data_dir
        self.cols = cols

    def load_data(self, data_file: Path) -> List[Tuple[str, int]]:
        """Load texts and labels

        Parameters
        ----------
        data_file : Path
            directory of the data file

        Returns
        -------
        List[Tuple[str, int]]
            list of loaded texts and labels
        """
        df = pd.read_csv(data_file)
        X = df[self.cols.processed_text].to_numpy()
        y = df[self.cols.label].to_numpy()

        data = []
        for idx, text in enumerate(X):
            data.append((text, y[idx]))

        return data

    def setup(self, vectorizer: TextVectorizer):
        train_data = self.load_data(self.processed_data_dir.train)
        val_data = self.load_data(self.processed_data_dir.val)
        test_data = self.load_data(self.processed_data_dir.test)

        logger.debug(f"Volume of training data: {len(train_data)}")
        logger.debug(f"Volume of validation data: {len(val_data)}")
        logger.debug(f"Volume of test data: {len(test_data)}")

        self.train_ds = TextDataset(train_data, vectorizer)
        self.val_ds = TextDataset(val_data, vectorizer)
        self.test_ds = TextDataset(test_data, vectorizer)

    def collate_fn(self, batch):
        """Convert the input raw data from the dataset into model input"""
        # Sort batch according to sequence length
        batch.sort(key=lambda x: len(x["vectors"]), reverse=True)

        # Convert word vectors to tensors
        word_vector = [torch.Tensor(item["vectors"]) for item in batch]

        # Trim sequences to ensure consistent length
        word_vector = [
            torch.nn.ZeroPad2d((0, 0, 0, self.max_seq_len - len(vec)))(vec)
            if self.max_seq_len > len(vec)
            else vec[: self.max_seq_len, :]
            for vec in word_vector
        ]

        labels = torch.LongTensor(np.array([item["label"] for item in batch]))

        # Pad each vector sequence to the same size
        # [batch_size, word_vec_dim, sequence_length]
        padded_word_vector = pad_sequence(word_vector, batch_first=True).transpose(1, 2)

        return {
            "vectors": padded_word_vector,
            "label": labels,
            "texts": [item["text"] for item in batch],
        }

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            collate_fn=self.collate_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            collate_fn=self.collate_fn,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            collate_fn=self.collate_fn,
        )
