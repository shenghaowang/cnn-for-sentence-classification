import re
from collections import Counter
from dataclasses import dataclass
from enum import Enum
from itertools import chain
from pathlib import Path
from typing import Dict, List

# import cleantext
import pandas as pd
from loguru import logger
from omegaconf import DictConfig


@dataclass
class Cols:
    raw_text: str
    processed_text: str
    seq_len: str
    label: str


class DataType(Enum):
    TEXT_LABEL = "text_label"
    POS_NEG = "pos_neg"


def read_text_label_data(raw_datadir: DictConfig) -> pd.DataFrame:
    """Read the text and label data from a CSV file.

    Parameters
    ----------
    raw_datadir : DictConfig
        Path to the CSV files containing the text and label data.

    Returns
    -------
    pd.DataFrame
        Merged dataframe containing the text and label data.
    """
    return pd.concat([pd.read_csv(data) for data in raw_datadir.values()])


def read_pos_neg_data(raw_datadir: DictConfig, cols: Cols) -> pd.DataFrame:
    """Read the positive and negative data from CSV files.

    Parameters
    ----------
    raw_datadir : DictConfig
        Path to the CSV files containing the positive and negative data,
        with the positive data in `pos` and negative data in `neg`.
    cols : Cols
        Column names for the text and label data.

    Returns
    -------
    pd.DataFrame
        Merged dataframe containing the positive and negative data.
    """
    with open(raw_datadir.pos, encoding="ISO-8859-1") as f:
        lines = [line.strip() for line in f]

    pos_df = pd.DataFrame(lines, columns=[cols.raw_text])
    pos_df[cols.label] = 1

    with open(raw_datadir.neg, encoding="ISO-8859-1") as f:
        lines = [line.strip() for line in f]

    neg_df = pd.DataFrame(lines, columns=[cols.raw_text])
    neg_df[cols.label] = 0

    return pd.concat([pos_df, neg_df], ignore_index=True)


def clean_text(text: str, lowercase: bool = True) -> str:
    """Clean the text message

    Parameters
    ----------
    text : str
        original text
    lowercase : bool, optional
        whether to convert text to lowercase, by default True

    Returns
    -------
    str
        processed text
    """

    text = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", text)
    text = re.sub(r"\'s", " 's", text)
    text = re.sub(r"\'ve", " 've", text)
    text = re.sub(r"n\'t", " n't", text)
    text = re.sub(r"\'re", " 're", text)
    text = re.sub(r"\'d", " 'd", text)
    text = re.sub(r"\'ll", " 'll", text)
    text = re.sub(r",", " , ", text)
    text = re.sub(r"!", " ! ", text)
    text = re.sub(r"\(", " ( ", text)
    text = re.sub(r"\)", " ) ", text)
    text = re.sub(r"\?", " ? ", text)
    text = re.sub(r"\s{2,}", " ", text)

    return text.strip().lower() if lowercase else text.strip()


def write_processed_data(df: pd.DataFrame, data_dir: str):
    """Write processed data to local disk

    Parameters
    ----------
    df : pd.DataFrame
        processed comment data

        Required fields:
            - processed_text
            - label

    data_dir : str
        local path for exporting the data
    """
    df.to_csv(data_dir, index=False)


def build_vocab(texts: List[str], vocab_path: Path) -> None:
    all_tokens = []

    for text in texts:
        tokens = text.split()
        all_tokens.append(tokens)

    logger.info(f"Size of vocabulary: {len(all_tokens)}")

    # Flatten and count
    vocab = Counter(chain.from_iterable(all_tokens))

    with open(vocab_path, "w", encoding="utf-8") as f:
        for word, freq in vocab.items():
            f.write(f"{word}\t{freq}\n")


def load_vocab_with_freq(vocab_path: Path) -> Dict[str, int]:
    """Load vocabulary with frequencies from a file."""

    vocab = {}
    with open(vocab_path, "r", encoding="utf-8") as f:
        for line in f:
            word, freq = line.strip().split("\t")
            vocab[word] = int(freq)
    return vocab
