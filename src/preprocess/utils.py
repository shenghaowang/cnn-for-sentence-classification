import re
from dataclasses import dataclass
from enum import Enum

import cleantext
import pandas as pd
from omegaconf import DictConfig
from stopwordsiso import stopwords


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


def clean_en(text: str, rm_stops: bool = False) -> str:
    """Clean the text message

    Parameters
    ----------
    text : str
        original text
    rm_stops : bool, optional
        if stopwords need to be removed, by default False

    Returns
    -------
    str
        processed text with only Chinese characters
    """

    return cleantext.clean(text, stopwords=rm_stops)


def clean_zh(text: str, rm_stops: bool = False) -> str:
    """Clean the text message

    Parameters
    ----------
    text : str
        original text
    rm_stops : bool, optional
        if stopwords need to be removed, by default False

    Returns
    -------
    str
        processed text with only Chinese characters
    """
    if rm_stops:
        text = rm_stopwords(text, "cn")

    frags = [frag for frag in re.findall(r"[\u4e00-\u9fff]+", text)]

    return "".join(frags) if len(frags) > 0 else text


def rm_stopwords(text: str, stp_lang: str) -> str:
    """Remove stopwords from the text

    Parameters
    ----------
    text : str
        original text
    stp_lang : str
        language code for stopword list,
        e.g. "en" for English

    Returns
    -------
    str
        processed text with no stopword
    """
    return "".join([char for char in text if char not in stopwords(stp_lang)])


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
