from pathlib import Path
from typing import List, Tuple, Union

import pytorch_lightning as pl
import torch
from loguru import logger
from omegaconf import DictConfig
from pytorch_lightning.callbacks import Callback

from data.text_data import TextDataModule, TextVectorizer
from model.cnn import ConvNet
from model.text_classifier import TextClassifier


def trainer(
    model: Union[ConvNet, str],
    train_params: DictConfig,
    hyperparams: DictConfig,
    train_data: List[Tuple],
    valid_data: List[Tuple],
    test_data: List[Tuple],
    model_file: Path,
) -> None:

    # Create a pytorch trainer
    metrics_logger = MetricsLogger()
    trainer = pl.Trainer(
        max_epochs=hyperparams.max_epochs,
        callbacks=[metrics_logger]
        # check_val_every_n_epoch=1
    )

    # Initialize our data loader with the passed vectorizer
    dm = TextDataModule(
        vectorizer=TextVectorizer(),
        batch_size=train_params.batch_size,
        max_seq_len=train_params.max_seq_len,
        train_data=train_data,
        valid_data=valid_data,
        test_data=test_data,
    )

    # Instantiate a new model
    classifier = TextClassifier(
        model,
        num_class=train_params.num_classes,
        learning_rate=hyperparams.learning_rate,
        batch_size=train_params.batch_size,
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
    model_dir = Path(model_file).parent
    model_dir.mkdir(parents=True, exist_ok=True)

    torch.save(classifier.state_dict(), model_file)
    logger.info(f"Fitted model is exported to {model_file}.")


class MetricsLogger(Callback):
    def __init__(self):
        super().__init__()
        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []

    def on_validation_epoch_end(self, trainer, pl_module):
        # Extract metrics from the trainer's logger after the validation epoch ends
        train_loss = trainer.callback_metrics.get("train_loss_epoch", None)
        val_loss = trainer.callback_metrics.get("val_loss", None)
        train_acc = trainer.callback_metrics.get("train_acc_epoch", None)
        val_acc = trainer.callback_metrics.get("val_acc", None)

        # Only append metrics if they exist
        if train_loss is not None:
            self.train_losses.append(train_loss.item())

        if val_loss is not None:
            self.val_losses.append(val_loss.item())

        if train_acc is not None:
            self.train_accs.append(train_acc.item())

        if val_acc is not None:
            self.val_accs.append(val_acc.item())

        # Optionally, print metrics for each epoch
        if train_loss is not None and val_loss is not None:
            logger.debug(
                f"Epoch {trainer.current_epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}"
            )

        if train_acc is not None and val_acc is not None:
            logger.debug(
                f"Epoch {trainer.current_epoch}: train_acc={train_acc:.4f}, val_acc={val_acc:.4f}"
            )
