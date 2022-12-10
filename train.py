from pathlib import Path
import argparse
import json

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)

def main():
    print(f"Loading model: {args.model_name_or_path}")
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name_or_path)

    print(f"Loading tokenizer: {args.model_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    print(f"Setting up training params using: {args.data_dir}")
    training_params = json.load(open((args.config_path)))
    training_args = TrainingArguments(**training_params)
    print(f"Training params: {training_params}")

    print(f"Loading dataset from: {args.data_dir}")
    dataset = load_dataset_dict(args.data_dir, splits=["train", "val"])

    print("Tokenizing dataset")
    tokenized_dataset = tokenize(dataset, tokenizer)





if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    DISTILBERT_UNCASED = "distilbert-base-uncased"
    parser.add_argument("-m", "--model_name_or_path", type=str, default=DISTILBERT_UNCASED)
    parser.add_argument("-c", "--config", dest="config_path", default="config/train.config.mc.json", type=Path)
    parser.add_argument("-d", "--data_dir", type=Path, default="data/splits/multitarget-conan")
    parser.add_argument("-f", "--freeze_first_n", type=int, default=0)

    args = parser.parse_args()

    main()