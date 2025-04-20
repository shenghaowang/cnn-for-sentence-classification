from pathlib import Path

import hydra
import pytorch_lightning as pl
import torch
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import EarlyStopping

from data.text_data import TextDataModule
from model.cnn import ConvNet
from model.lstm import LSTM
from model.model_type import ModelType
from model.text_classifier import TextClassifier
from preprocess.utils import Cols
from train.metrics_logger import MetricsLogger


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig):
    logger.info(OmegaConf.to_yaml(cfg, resolve=True))
    train_cfg = cfg.dataset.train

    torch.manual_seed(seed=42)

    # Init data module
    dm = TextDataModule(
        batch_size=train_cfg.batch_size,
        max_seq_len=train_cfg.max_seq_len,
        processed_data_dir=cfg.dataset.processed,
        cols=Cols(**cfg.dataset.fields),
    )
    dm.setup(word2vec_path=cfg.word2vec_path, vocab_path=cfg.dataset.processed.vocab)

    # Init neural network
    hyperparams = cfg.model.hyperparams
    if cfg.model.name == ModelType.CNN.value:
        model = ConvNet(
            num_filters=hyperparams.num_filters,
            kernel_sizes=hyperparams.kernel_sizes,
            embedding_dim=hyperparams.embedding_dim,
            output_dim=train_cfg.num_classes if train_cfg.num_classes > 2 else 1,
        )

    elif cfg.model.name == ModelType.LSTM.value:
        model = LSTM(
            input_dim=hyperparams.input_dim,
            hidden_dim=hyperparams.hidden_dim,
            num_layers=hyperparams.num_layers,
            output_dim=train_cfg.num_classes if train_cfg.num_classes > 2 else 1,
        )

    else:
        raise ValueError(f"Unknown model type: {cfg.model.name}")

    # Create trainer
    metrics_logger = MetricsLogger()
    early_stop_callback = EarlyStopping(
        monitor="val_loss", min_delta=0.001, patience=3, verbose=True, mode="min"
    )
    trainer = pl.Trainer(
        max_epochs=cfg.model.hyperparams.max_epochs,
        callbacks=[metrics_logger, early_stop_callback],
    )

    # Init text classifier
    classifier = TextClassifier(
        model,
        num_class=train_cfg.num_classes,
        learning_rate=cfg.model.hyperparams.learning_rate,
        batch_size=train_cfg.batch_size,
    )

    # Train and validate the model
    trainer.fit(
        classifier,
        train_dataloaders=dm.train_dataloader(),
        val_dataloaders=dm.val_dataloader(),
    )

    # Test the model
    trainer.test(classifier, dm.test_dataloader())

    # Predict on the same test set to show some output
    output = trainer.predict(classifier, dm.test_dataloader())

    for i in range(2):
        logger.info("====================")
        logger.info(f"Text: {output[1]['texts'][i]}")
        logger.info(f"Prediction: {output[1]['predictions'][i].numpy()}")
        logger.info(f"Actual Label: {output[1]['label'][i].numpy()}")

    # Export fitted model
    model_dir = Path(cfg.model_file).parent
    model_dir.mkdir(parents=True, exist_ok=True)

    torch.save(classifier.state_dict(), cfg.model_file)
    logger.info(f"Fitted model is exported to {cfg.model_file}.")


if __name__ == "__main__":
    main()
