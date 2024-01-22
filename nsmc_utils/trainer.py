import os
import torch
import logging
from transformers import PreTrainedModel
from transformers.optimization import AdamW
from pytorch_lightning import LightningModule
from torch.optim.lr_scheduler import ExponentialLR
from .arguments import TrainArguments

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint


logger = logging.getLogger(__name__)

def accuracy(preds, labels, ignore_index=None):
    with torch.no_grad():
        assert preds.shape[0] == len(labels)
        correct = torch.sum(preds == labels)
        total = torch.sum(torch.ones_like(labels))
        if ignore_index is not None:
            # 모델이 맞춘 것 가운데 ignore index에 해당하는 것 제외
            correct -= torch.sum(torch.logical_and(preds == ignore_index, preds == labels))
            # accuracy의 분모 가운데 ignore index에 해당하는 것 제외
            total -= torch.sum(labels == ignore_index)
    return correct.to(dtype=torch.float) / total.to(dtype=torch.float)


class ClassificationTask(LightningModule):
    def __init__(self, model: PreTrainedModel, args: TrainArguments):
        super().__init__()
        self.model = model
        self.args = args

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.args.learning_rate)
        scheduler = ExponentialLR(optimizer, gamma=0.9)
        return {
            'optimizer': optimizer,
            'scheduler': scheduler
        }

    def training_step(self, inputs, batch_idx):
        # outputs: SequenceClassiferOutput
        outputs = self.model(**inputs)
        preds = outputs.logits.argmax(dim=-1)  # 1일 확률
        labels = inputs['labels']
        acc = accuracy(preds, labels)
        self.log("loss", outputs.loss, prog_bar=False, logger=True, on_step=True, on_epoch=False)
        self.log("acc", acc, prog_bar=True, logger=True, on_step=True, on_epoch=False)
        return outputs.loss

    def validation_step(self, inputs, batch_idx):
        # outputs: SequenceClassiferOutput
        outputs = self.model(**inputs)
        preds = outputs.logits.argmax(dim=-1)
        labels = inputs['labels']
        acc = accuracy(preds, labels)
        self.log("val_loss", outputs.loss, prog_bar=True, logger=True, on_step=False, on_epoch=True)
        self.log("val_acc", acc, prog_bar=True, logger=True, on_step=False, on_epoch=True)
        return outputs.loss


def get_trainer(args, return_trainer_only=True):
    ckpt_path = os.path.abspath(args.model_dir)
    os.makedirs(ckpt_path, exist_ok=True)
    checkpoint_callback = ModelCheckpoint(
        dirpath=ckpt_path,
        save_top_k=args.save_top_k,
        monitor=args.monitor.split()[1],
        mode=args.monitor.split()[0],
        filename='{epoch}-{val_loss:.2f}',
    )
    trainer = Trainer(
        max_epochs=args.epochs,
        fast_dev_run=args.test_mode,
        num_sanity_val_steps=None if args.test_mode else 0,
        callbacks=[checkpoint_callback],
        default_root_dir=ckpt_path,
        # For GPU Setup
        deterministic=torch.cuda.is_available() and args.seed is not None,
        accelerator='cuda' if torch.cuda.is_available() else 'cpu',
        precision=16 if args.fp16 else 32,
    )
    logger.info("======== GPU: {}".format(torch.cuda.device_count() if torch.cuda.is_available() else None))
    if return_trainer_only:
        return trainer
    else:
        return checkpoint_callback, trainer