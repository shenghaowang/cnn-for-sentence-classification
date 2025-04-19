import hydra
import torch
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from data.text_data import load_data
from model.cnn import ConvNet
from preprocess.utils import Cols
from train_and_infer.trainer import trainer


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig):
    logger.info(OmegaConf.to_yaml(cfg, resolve=True))
    processed_data = cfg.dataset.processed
    cols = Cols(**cfg.dataset.fields)

    torch.manual_seed(seed=42)

    # Load training, validation, and test data
    train_data = load_data(processed_data.train, cols.processed_text, cols.label)
    val_data = load_data(processed_data.val, cols.processed_text, cols.label)
    test_data = load_data(processed_data.test, cols.processed_text, cols.label)

    logger.info(f"Volume of training data: {len(train_data)}")
    logger.info(f"Volume of validation data: {len(val_data)}")
    logger.info(f"Volume of test data: {len(test_data)}")

    # Initialise text classification model
    train_cfg = cfg.dataset.train
    trainer(
        model=ConvNet(
            hyparams=cfg.model.hyperparams,
            in_channels=train_cfg.word_vec_dim,
            seq_len=train_cfg.max_seq_len,
            output_dim=train_cfg.num_classes if train_cfg.num_classes > 2 else 1,
        ),
        train_params=train_cfg,
        hyperparams=cfg.model.hyperparams,
        train_data=train_data,
        valid_data=val_data,
        test_data=test_data,
        model_file=cfg.model_file,
    )


if __name__ == "__main__":
    main()
