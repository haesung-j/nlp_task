import re
import os
from glob import glob
from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class TrainArguments:
    pretrained_model_name: str = field(
        default="beomi/kcbert-base",
        metadata={"help": "pretrained model name"}
    )
    model_dir: str = field(
        default=None,
        metadata={"help": "The output model dir."}
    )
    max_seq_length: int = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        }
    )
    data_root_path: str = field(
        default='./data/nsmc',
        metadata={"help": "Data root directory."}
    )
    monitor: str = field(
        default="min val_loss",
        metadata={"help": "monitor condition (save top k)"}
    )
    save_top_k: int = field(
        default=1,
        metadata={"help": "save top k model checkpoints."}
    )
    seed: int = field(
        default=None,
        metadata={"help": "random seed."}
    )
    learning_rate: float = field(
        default=5e-5,
        metadata={"help": "learning rate"}
    )
    epochs: int = field(
        default=3,
        metadata={"help": "max epochs"}
    )
    batch_size: int = field(
        default=64,
        metadata={"help": "batch size. if 0, Let PyTorch Lightening find the best batch size"}
    )
    cpu_workers: int = field(
        default=os.cpu_count(),
        metadata={"help": "number of CPU workers"}
    )
    fp16: bool = field(
        default=False,
        metadata={"help": "Enable train on FP16"}
    )
    test_mode: bool = field(
        default=False,
        metadata={"help": "Test Mode enables `fast_dev_run`"}
    )


@dataclass
class DeployArguments:
    def __init__(self,
                 pretrained_model_name: Optional[str] = None,
                 model_dir: Optional[str] = None,
                 max_seq_length: int = 64
                 ):
        self.pretrained_model_name = pretrained_model_name
        self.model_dir = model_dir
        self.max_seq_legnth = max_seq_length

        ckpt_file_names = glob(os.path.join(model_dir, "*.ckpt"))
        ckpt_file_names = [name for name in ckpt_file_names if "temp" not in name and "tmp" not in name]

        if len(ckpt_file_names) == 0:
            raise Exception("No *.ckpt file in {}".format(model_dir))

        # Select minimum val_loss model
        min_loss = 999999999
        idx = -1
        for i, name in enumerate(ckpt_file_names):
            pattern = re.compile(r"[0-9\.]+.ckpt")
            val_loss = re.findall(pattern, ckpt_file_names[0])[-1].replace(".ckpt", "")
            val_loss = float(val_loss)
            if val_loss < min_loss:
                min_loss = val_loss
                idx = i

        self.selected_model = ckpt_file_names[idx]


