from pathlib import Path

import hydra
import pytorch_lightning as pl
import torch
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from data.text_data import TextDataModule, TextVectorizer
from model.cnn import ConvNet
from model.text_classifier import TextClassifier
from preprocess.utils import Cols
from train_and_infer.metrics_logger import MetricsLogger


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig):
    logger.info(OmegaConf.to_yaml(cfg, resolve=True))
    train_cfg = cfg.dataset.train

    torch.manual_seed(seed=42)

    # Initialize our data loader with the passed vectorizer
    dm = TextDataModule(
        batch_size=train_cfg.batch_size,
        max_seq_len=train_cfg.max_seq_len,
        processed_data_dir=cfg.dataset.processed,
        cols=Cols(**cfg.dataset.fields),
    )
    dm.setup(vectorizer=TextVectorizer())

    # Initialise text classification model
    model = ConvNet(
        hyparams=cfg.model.hyperparams,
        in_channels=train_cfg.word_vec_dim,
        seq_len=train_cfg.max_seq_len,
        output_dim=train_cfg.num_classes if train_cfg.num_classes > 2 else 1,
    )

    # Create a pytorch trainer
    metrics_logger = MetricsLogger()
    trainer = pl.Trainer(
        max_epochs=cfg.model.hyperparams.max_epochs,
        callbacks=[metrics_logger]
        # check_val_every_n_epoch=1
    )

    # Instantiate a new model
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
