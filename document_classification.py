import logging
import warnings
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import BertTokenizer, BertConfig, BertForSequenceClassification
from nsmc_utils.ClassificationDataUtils import data_collator, NsmcCorpus, ClassificationDataset, TrainArguments
from nsmc_utils.ClassificationTrainer import ClassificationTask, get_trainer


warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s:%(message)s")

def run():
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

    task = ClassificationTask(model, args)
    trainer = get_trainer(args)
    trainer.fit(task,
                train_dataloaders=train_dataloader,
                val_dataloaders=val_dataloader
                )


if __name__ == '__main__':

    run()