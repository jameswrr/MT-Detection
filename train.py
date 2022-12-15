from pathlib import Path
import argparse
import json
import evaluate
import numpy as np
from typing import Any, Dict, List, Union

from datasets import DatasetDict, load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer
)


def main():
    print(f"Loading tokenizer: {args.model_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    print(f"Loading dataset from: {args.data_dir}")
    dataset = load_dataset_dict(args.data_dir, splits=["train", "val", "test"])

    print("Tokenizing dataset")
    tokenized_dataset = tokinize(dataset, tokenizer)
    
    id2label = {0: "EN", 1: "MT"}
    label2id = {"EN": 0, "MT": 1}

    print(f"Setting up training params using: {args.data_dir}")
    training_params = json.load(open((args.config_path)))
    training_args = TrainingArguments(**training_params)
    print(f"Training params: {training_params}")

    print(f"Loading model: {args.model_name_or_path}")
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name_or_path, 
        num_labels=2, 
        id2label=id2label, 
        label2id=label2id,
    )

    print("Training")
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["val"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    trainer.train()


def load_dataset_dict(data_dir: Path, splits: List[str] = ["train", "val", "test"]) -> DatasetDict:
    data_files = {split: str(data_dir / f"{split}.csv") for split in splits}
    dataset = load_dataset("csv", data_files=data_files)
    return dataset

def tokinize(dataset, tokenizer):
    def preprocess_function(examples):
        return tokenizer(examples["text"], truncation=True)
    return dataset.map(preprocess_function, batched=True)

def compute_metrics(eval_pred):
    accuracy = evaluate.load("accuracy")
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    DISTILBERT_UNCASED = "distilbert-base-uncased"
    parser.add_argument("-m", "--model_name_or_path", type=str, default=DISTILBERT_UNCASED)
    parser.add_argument("-c", "--config", dest="config_path", default="config/train.config.distil.FR.json", type=Path)
    parser.add_argument("-d", "--data_dir", type=Path, default="data/splits/FR")

    args = parser.parse_args()

    main()