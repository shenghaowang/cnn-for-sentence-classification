from loguru import logger
from pytorch_lightning.callbacks import Callback


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
