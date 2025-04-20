from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from gensim.models import KeyedVectors
from loguru import logger
from omegaconf import DictConfig

# from loguru import logger
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

from preprocess.utils import Cols, load_vocab_with_freq


class TextDataset(Dataset):
    """Creates an pytorch dataset to consume our pre-loaded text data
    Reference: https://pytorch.org/tutorials/beginner/basics/data_tutorial.html
    """

    def __init__(
        self,
        data: List[Tuple[str, int]],
        word2vec: Dict[str, np.ndarray],
        embedding_dim: int = 300,
    ):
        self.dataset = data
        self.word2vec = word2vec
        self.embedding_dim = embedding_dim

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        text, label = self.dataset[idx]
        # word_vecs = self.vectorizer.vectorize(text)

        tokens = text.split()
        vectors = []
        for word in tokens:
            if word in self.word2vec:
                vec = torch.tensor(self.word2vec[word])

            else:
                vec = torch.zeros(self.embedding_dim)

            vectors.append(vec)

        return {
            "vectors": torch.stack(vectors),
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

    def setup(self, word2vec_path: Path, vocab_path: Path):
        train_data = self.load_data(self.processed_data_dir.train)
        val_data = self.load_data(self.processed_data_dir.val)
        test_data = self.load_data(self.processed_data_dir.test)

        logger.debug(f"Volume of training data: {len(train_data)}")
        logger.debug(f"Volume of validation data: {len(val_data)}")
        logger.debug(f"Volume of test data: {len(test_data)}")

        # Load word vectors
        word2vec = KeyedVectors.load_word2vec_format(word2vec_path, binary=True)
        # word2vec.init_sims(replace=True)

        # Add random vectors for unseen words
        embedding_dim = word2vec.vector_size
        vocab = load_vocab_with_freq(vocab_path)
        word2vec = build_extended_vectors(word2vec, vocab, embedding_dim)

        # Init datasets
        self.train_ds = TextDataset(train_data, word2vec, embedding_dim)
        self.val_ds = TextDataset(val_data, word2vec, embedding_dim)
        self.test_ds = TextDataset(test_data, word2vec, embedding_dim)

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


def add_unknown_words(
    word_vecs: KeyedVectors, vocab, min_df=1, k=300, seed=42
) -> KeyedVectors:
    """
    Add random vectors for unseen words (i.e., words in `vocab` but not in `word_vecs`).

    Parameters
    ----------
    word_vecs : dict-like
        A dictionary (e.g., gensim KeyedVectors or Python dict) of known word -> vector (np.array).
    vocab : dict
        A dict mapping words to their document frequency or counts.
    min_df : int
        Minimum number of times a word must appear to be added to the vocab.
    k : int
        Dimensionality of the embedding vectors.
    seed : int
        Random seed for reproducibility.
    """
    np.random.seed(seed)
    unseen = 0

    for word in vocab:
        if word not in word_vecs and vocab[word] >= min_df:
            word_vecs[word] = np.random.uniform(-0.25, 0.25, k).astype(np.float32)
            unseen += 1

    logger.debug(f"Size of vocabulary: {len(word_vecs)}")
    logger.debug(f"Added {unseen} unseen words to the vocab")

    return word_vecs


def build_extended_vectors(
    word2vec, vocab, k=300, min_df=2, seed=42
) -> Dict[str, np.ndarray]:
    np.random.seed(seed)
    extended = {}
    unseen = 0

    logger.debug(f"Size of vocabulary: {len(vocab)}")

    for word in vocab:
        if word in word2vec:
            extended[word] = word2vec[word]

        elif vocab[word] >= min_df:
            extended[word] = np.random.uniform(-0.25, 0.25, k).astype(np.float32)
            unseen += 1

    logger.debug(f"Number of unknown words: {unseen}")

    return extended
