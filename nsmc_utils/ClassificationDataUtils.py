import os
import csv
import logging
import time
import torch

from dataclasses import dataclass, field
from typing import List, Optional
from transformers import PreTrainedTokenizer, BertConfig, BertForSequenceClassification
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertTokenizer

logger = logging.getLogger(__name__)

# args = {
#     'max_seq_length': 128,
#     'data_root_path': './data/nsmc'
# }

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
class ClassificationTexts:
    text_a: str
    text_b: Optional[str] = None
    label: Optional[str] = None


@dataclass
class ClassificationFeatures:
    input_ids: List[int]
    attention_mask: Optional[List[int]] = None
    token_type_ids: Optional[list[int]] = None
    label: Optional[int] = None


class NsmcCorpus:
    """
    nsmc 관련 클래스
    """
    def __init__(self):
        pass

    def get_texts(self, data_root_path, mode):
        """
        data_root_path 내 nsmc 데이터를 불러와 list[ClassificationTexts]로 반환
        :param data_root_path:
        :param mode:
        :return: list
        """
        data_path = os.path.join(data_root_path, f"ratings_{mode}.txt")
        logger.info(f"{mode} 데이터를 불러오는 중..")
        lines = list(csv.reader(open(data_path, 'r', encoding='utf-8'), delimiter='\t', quotechar='"'))
        texts = []
        for i, line in enumerate(lines):
            if i == 0:
                continue
            _, text_a, label = line
            texts.append(ClassificationTexts(text_a=text_a, text_b=None, label=label))

        logger.info(f"{mode} 데이터 불러오기 완료")
        return texts

    def get_labels(self):
        return ["0", "1"]

    @property
    def num_labels(self):
        return len(self.get_labels())


def _convert_texts_to_classification_features(
        texts: List[ClassificationTexts],
        tokenizer: PreTrainedTokenizer,
        args: TrainArguments,
        label_list: List[str]
) -> List[ClassificationFeatures]:
    """
    주어진 texts, label을 tokenizer를 통해 BERT 학습용 데이터로 변환하는 함수
    :param texts: ClassificationText(class 변수로 text_a, text_b, label을 가짐)
    :param tokenizer: huggingface transformer로부터 불러온 사전 학습된 토크나이저
    :param args: TrainArguments로 정의한 아규먼트
    :param label_list: 리스트 형태의 label 목록
    :return: ["input_ids", "attention_mask", "token_type_ids", "label"]로 이루어진 ClassificationFeatures를 원소로 갖는 리스트
    """
    label_map = {label: i for i, label in enumerate(label_list)}
    labels = [label_map[text.label] for text in texts]

    logger.info("Tokenize sentences, it could take a lot of time...")

    start = time.time()
    # batch_encoding - dictionary type(keys: ['input_ids', 'token_type_ids', 'attention_mask'])
    batch_encoding = tokenizer(
        [(text.text_a, text.text_b) for text in texts],
        max_length=args.max_seq_length,
        padding='max_length',
        truncation=True
    )
    logger.info("tokenize sentence [took %.3f s]", time.time()-start)

    logger.info("Convert tokenized sentences to ClassificationFeatures")
    features = []
    for i in range(len(texts)):
        inputs = {key: batch_encoding[key][i] for key in batch_encoding.keys()} # 하나의 샘플 추출
        feature = ClassificationFeatures(**inputs, label=labels[i])  # label까지 포함한 feature 생성
        features.append(feature)

    logger.info("Done!")

    logger.debug("=== first 3 texts examples ===")
    for i, text in enumerate(texts[:3]):
        if text.text_b is None:
            logger.debug("sentence: {}".format(text.text_a))
        else:
            sentence = text.text_a + " + " + text.text_b
            logger.debug("sentence A, B: {}".format(sentence))
        logger.debug("tokens: {}".format(" ".join(tokenizer.convert_ids_to_tokens(features[i].input_ids))))
        logger.debug("label: {}".format(text.label))
        logger.debug("features: {}".format(features[i]))

    return features


class ClassificationDataset(Dataset):
    def __init__(self,
                 args: TrainArguments,
                 tokenizer: PreTrainedTokenizer,
                 corpus,
                 mode: Optional[str] = 'train',
                 convert_texts_to_classification_features_fn=_convert_texts_to_classification_features
                 ):
        if corpus is not None:
            self.corpus = corpus
        else:
            raise KeyError("corpus is not valid!")

        if not mode in ['train', 'val', 'test']:
            raise KeyError(f"mode({mode}) is not a valid split name!")

        # Load data features
        data_root_path = args.data_root_path
        logger.info(f"Creating features from dataset file at {data_root_path}")
        texts = self.corpus.get_texts(data_root_path, mode)
        self.features = convert_texts_to_classification_features_fn(
            texts,
            tokenizer,
            args,
            self.corpus.get_labels()
        )

    def __len__(self):
        return len(self.features)

    def __getitem__(self, i):
        return self.features[i]

    def get_labels(self):
        return self.corpus.get_labels()


def data_collator(features):
    """
    DataLoader 내부에서 뽑은 인스턴스들을 배치로 만드는 역할 수행 함수
    즉, input_ids, attention_mask, token_type_ids, label 각각 행으로 쌓거나, 늘어놓아(label의 경우),
    각 key별로 배치 생성해줌
    """
    if not isinstance(features[0], dict):
        features = [vars(f) for f in features]

    first = features[0]
    batch = {}

    # handling for labels.
    if "label" in first and first["label"] is not None:
        label = first["label"].item() if isinstance(first["label"], torch.Tensor) else first["label"]
        dtype = torch.long if isinstance(label, int) else torch.float
        batch["labels"] = torch.tensor([f["label"] for f in features], dtype=dtype)
    elif "label_ids" in first and first["label_ids"] is not None:
        if isinstance(first["label_ids"], torch.Tensor):
            batch["labels"] = torch.stack([f["label_ids"] for f in features])
        else:
            dtype = torch.long if type(first["label_ids"][0]) is int else torch.float
            batch["labels"] = torch.tensor([f["label_ids"] for f in features], dtype=dtype)

    # Handling of all other possible keys.
    for k, v in first.items():
        if k not in ("label", "label_ids") and v is not None and not isinstance(v, str):
            if isinstance(v, torch.Tensor):
                batch[k] = torch.stack([f[k] for f in features])
            else:
                batch[k] = torch.tensor([f[k] for f in features], dtype=torch.long)

    return batch


if __name__=="__main__":
    logger.setLevel(logging.INFO)
    stream_hander = logging.StreamHandler()
    logger.addHandler(stream_hander)

    args = TrainArguments(
        max_seq_length=128,
        data_root_path='./data/nsmc',
        batch_size=64,
        model_dir='./model'
    )

    corpus = NsmcCorpus()

    tokenizer = BertTokenizer.from_pretrained(
        args.pretrained_model_name,
        do_lower_case=False
    )

    train_dataset = ClassificationDataset(
        args=args,
        corpus=corpus,
        tokenizer=tokenizer,
        mode='train'
    )

    val_dataset = ClassificationDataset(
        args=args,
        corpus=corpus,
        tokenizer=tokenizer,
        mode='test'
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=RandomSampler(train_dataset, replacement=False),
        drop_last=False,
        collate_fn=data_collator
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        sampler=SequentialSampler(val_dataset),
        collate_fn=data_collator,
        drop_last=False
    )

    # 모델 불러오기
    pretrained_model_config = BertConfig.from_pretrained(
        args.pretrained_model_name,
        num_labels=corpus.num_labels
    )

    model = BertForSequenceClassification.from_pretrained(
        args.pretrained_model_name,
        config=pretrained_model_config
    )

