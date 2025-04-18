import importlib
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from sklearn.model_selection import train_test_split

from preprocess.utils import Cols, write_processed_data


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig) -> None:
    logger.info(OmegaConf.to_yaml(cfg, resolve=True))

    raw_data = cfg.dataset.raw
    processed_data = cfg.dataset.processed
    rm_stops = cfg.dataset.rm_stopwords
    cols = Cols(**cfg.dataset.fields)

    df = pd.read_csv(raw_data.train)
    logger.info(f"Training data: {df.shape}")
    logger.debug(f"\n{df.head()}")

    # Check the class distribution
    logger.info(
        f"Class distribution of the training data:\n{df[cols.label].value_counts()}"
    )

    # Check the length of texts in number of words
    df[cols.seq_len] = df[cols.raw_text].apply(lambda text: len(text.split()))
    logger.info(f"Length of texts:\n{df['len'].describe()}")

    for percentile in [50, 75, 95]:
        logger.info(
            f"{percentile}th percentile of the text length: "
            + f"{np.percentile(df[cols.seq_len].values, percentile)}"
        )

    clean_text_func = getattr(
        importlib.import_module("preprocess.utils"), cfg.clean_text_func
    )

    # Preview text processing
    logger.debug("Preview the processed text:")
    for text in df[cols.raw_text][:5]:
        logger.debug(f"Original text: {text}")
        logger.debug(f"Processed text: {clean_text_func(text, rm_stops)}")
        logger.debug("\n")

    # Create directory for storing the processed data
    output_dir = Path(processed_data.train).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # To reduce the chance of data drift, combine the raw datasets
    # and resplit for training, validation, and test
    combined_df = pd.concat([pd.read_csv(data) for data in cfg.dataset.raw.values()])
    combined_df[cols.processed_text] = combined_df[cols.raw_text].apply(clean_text_func)
    train_df, rest_df = train_test_split(combined_df, test_size=0.3, random_state=42)
    val_df, test_df = train_test_split(rest_df, test_size=0.5, random_state=42)

    logger.info(f"Volume of training data: {len(train_df)}")
    logger.info(f"Volume of dev data: {len(val_df)}")
    logger.info(f"Volume of test data: {len(test_df)}")

    # Export data
    required_cols = [cols.processed_text, cols.label]
    write_processed_data(train_df[required_cols], processed_data["train"])
    write_processed_data(val_df[required_cols], processed_data["val"])
    write_processed_data(test_df[required_cols], processed_data["test"])


if __name__ == "__main__":
    main()
