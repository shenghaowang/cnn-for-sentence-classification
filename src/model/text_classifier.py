import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchmetrics


class TextClassifier(pl.LightningModule):
    def __init__(self, model, num_class: int, learning_rate: float, batch_size: int):
        super().__init__()
        self.save_hyperparameters()

        self.model = model
        self.learning_rate = learning_rate
        self.is_binary = num_class == 2
        self.batch_size = batch_size

        if num_class > 2:
            self.loss_fn = nn.CrossEntropyLoss()
            self.f1 = torchmetrics.F1Score(task="multiclass", num_classes=num_class)

        else:
            self.loss_fn = nn.BCEWithLogitsLoss()
            self.f1 = torchmetrics.classification.BinaryF1Score()

    def forward(self, x):
        return self.model(x)

    def _calculate_loss(self, batch, stage="train"):
        # Fetch data and transform labels
        x = batch["vectors"]
        y = batch["label"].to(torch.float32)

        # Perform prediction and calculate loss and F1 score
        logits = self.forward(x)

        if self.is_binary:
            y = y.float()
            logits = logits.squeeze(-1)
            predictions = torch.round(torch.sigmoid(logits))

        else:
            y = y.long()
            predictions = torch.argmax(logits, dim=1)

        loss = self.loss_fn(logits, y)

        # Apply L2 regularization
        l2_lambda = 0.01
        fc_params = torch.cat([x.view(-1) for x in self.model.fc.parameters()])
        total_loss = loss + l2_lambda * torch.norm(fc_params, 2)

        # Compute accuracy
        acc = (predictions == y).float().mean()

        return total_loss, acc

    def training_step(self, batch, batch_idx):
        loss, acc = self._calculate_loss(batch, "train")

        # Logging
        self.log(
            "train_loss_epoch",
            loss,
            on_epoch=True,
            prog_bar=True,
            batch_size=self.batch_size,
        )
        self.log(
            "train_acc_epoch",
            acc,
            on_epoch=True,
            prog_bar=True,
            batch_size=self.batch_size,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        loss, acc = self._calculate_loss(batch, "val")

        # Logging
        self.log(
            "val_loss", loss, on_epoch=True, prog_bar=True, batch_size=self.batch_size
        )
        self.log(
            "val_acc", acc, on_epoch=True, prog_bar=True, batch_size=self.batch_size
        )
        return loss

    def test_step(self, batch, batch_idx):
        loss, acc = self._calculate_loss(batch, "test")

        # Logging
        self.log("test_acc", acc, prog_bar=True)
        return loss

    def predict_step(self, batch, batch_idx):
        x = batch["vectors"]
        y_hat = self.model(x)
        predictions = torch.argmax(y_hat, dim=1)

        return {
            "logits": y_hat,
            "predictions": predictions,
            "label": batch["label"],
            "texts": batch["texts"],
        }

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",  # because we want to minimize val_loss
            factor=0.5,  # reduce LR by half
            patience=2,  # wait 2 epochs with no improvement
            verbose=True,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",  # name must match what you log
                "interval": "epoch",
                "frequency": 1,
            },
        }

    def on_epoch_start(self):
        """Create a new progress bar for each epoch"""
        print("\n")
