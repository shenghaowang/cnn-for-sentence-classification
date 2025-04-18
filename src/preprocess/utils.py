import re
from dataclasses import dataclass

import cleantext
import pandas as pd
from stopwordsiso import stopwords


@dataclass
class Cols:
    raw_text: str
    processed_text: str
    seq_len: str
    label: str


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
