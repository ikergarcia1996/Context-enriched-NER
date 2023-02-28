import os
from model_utils.token_classification_hf import run_token_classification_model
from model_utils.eval_utils import evaluate_file
from model_utils.eval_utils import create_table_avg
import argparse
from datetime import datetime
from dataset_format.tsv2json import tsv2json
import shutil
from typing import List
from dataset_format.tag_encoding import rewrite_tags


def clean(directory: str):
    for filename in os.listdir(directory):
        if os.path.isfile(os.path.join(directory, filename)):
            if (
                "txt" not in filename
                and "tsv" not in filename
                and "json" not in filename
            ):
                os.remove(os.path.join(directory, filename))
        if os.path.isdir(os.path.join(directory, filename)):
            shutil.rmtree(os.path.join(directory, filename))


def vote_predictions(
    output_dir: str,
    num_iters: int,
):
    labels: List[List[List[str]]] = []
    for i in range(num_iters):
        labels_file: List[List[str]] = []
        with open(os.path.join(output_dir, str(i), "predictions.txt"), "r") as f:
            for line in f:
                line_labels = line.strip().split()
                labels_file.append(line_labels)

        if len(labels) == 0:
            for _ in range(len(labels_file)):
                labels.append([])

        for j in range(len(labels_file)):
            labels[j].append(labels_file[j])

    with open(os.path.join(output_dir, "predictions.txt"), "w") as f:
        for sentence_labels in labels:
            voted_labels = []
            assert len(sentence_labels) == num_iters
            assert all(
                len(sentence_labels[0]) == len(sentence_labels[i])
                for i in range(num_iters)
            )

            for i in range(len(sentence_labels[0])):
                possible_labels = [x[i] for x in sentence_labels]
                voted_labels.append(
                    max(set(possible_labels), key=possible_labels.count)
                )

            print(" ".join(rewrite_tags(voted_labels, encoding="iob2")), file=f)


def run_seql(
    train_file: str,
    dev_file: str,
    test_file: str,
    model_name: str,
    cache_dir: str,
    output_dir: str,
    num_train_epochs: int = 4,
    batch_size: int = 32,
    gradient_accumulation_steps: int = 1,
    learning_rate: float = 2e-5,
    fp16: bool = True,
    encoding: str = "iob2",
    merge_tags: bool = False,
    label_all_tokens: bool = False,
    lr_scheduler_type: str = None,
    warmup_ratio: str = None,
    warmup_steps: str = None,
    max_seq_length: int = None,
    experiment_name: str = str(datetime.now()),
    number_of_experiments: int = 5,
    run_ner_file: str = "third_party/run_ner.py",
    deepspeed: bool = False,
    deepspeed_gpu_id: int = 0,
    save_models: bool = False,
):

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for exp_no in range(number_of_experiments):
        exp_output_dir = os.path.join(output_dir, str(exp_no))
        if not os.path.exists(exp_output_dir):
            os.makedirs(exp_output_dir)

        tsv2json(
            input_path=train_file,
            output_path=os.path.join(exp_output_dir, "train.json"),
            encoding=encoding,
        )

        tsv2json(
            input_path=dev_file,
            output_path=os.path.join(exp_output_dir, "dev.json"),
            encoding=encoding,
        )

        tsv2json(
            input_path=test_file,
            output_path=os.path.join(exp_output_dir, "test.json"),
            encoding=encoding,
        )

        run_token_classification_model(
            run_ner_file=run_ner_file,
            train_file=os.path.join(exp_output_dir, "train.json"),
            validation_file=os.path.join(exp_output_dir, "dev.json"),
            test_file=os.path.join(exp_output_dir, "test.json"),
            model_name=model_name,
            cache_dir=cache_dir,
            output_dir=exp_output_dir,
            num_train_epochs=num_train_epochs,
            batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            learning_rate=learning_rate,
            fp16=fp16,
            label_all_tokens=label_all_tokens,
            lr_scheduler_type=lr_scheduler_type,
            warmup_ratio=warmup_ratio,
            warmup_steps=warmup_steps,
            experiment_name=experiment_name,
            max_seq_length=max_seq_length,
            deepspeed=deepspeed,
            deepspeed_gpu_id=deepspeed_gpu_id,
        )

        evaluate_file(
            original_dataset_path=test_file,
            json_path=os.path.join(exp_output_dir, "test.json"),
            tags_path=os.path.join(exp_output_dir, "predictions.txt"),
            output_dir=exp_output_dir,
            output_name="test",
            encoding="iob2",
            merge_tags=merge_tags,
        )

        if not save_models:
            clean(exp_output_dir)

    create_table_avg(
        output_dir=output_dir,
        num_iters=number_of_experiments,
        output_path=os.path.join(output_dir, "result_table.csv"),
    )

    vote_predictions(output_dir=output_dir, num_iters=number_of_experiments)

    evaluate_file(
        original_dataset_path=test_file,
        json_path=os.path.join(os.path.join(output_dir, str(0)), "test.json"),
        tags_path=os.path.join(output_dir, "predictions.txt"),
        output_dir=output_dir,
        output_name="test",
        encoding="iob2",
        merge_tags=merge_tags,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run NER experiments")

    parser.add_argument(
        "--train_file",
        type=str,
        required=True,
        help="Train file in tsv format",
    )

    parser.add_argument(
        "--dev_file",
        type=str,
        required=True,
        help="Dev file in tsv format",
    )

    parser.add_argument(
        "--test_file",
        type=str,
        required=True,
        help="Path to the ner test file",
    )

    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="Hugging face model name or path",
    )

    parser.add_argument(
        "--cache_dir",
        type=str,
        default="../../transformer_models/",
        help="Cache dir for models",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory",
    )

    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=4,
        help="Number of training epochs",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Training/eval batch size",
    )

    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Training gradient accumulation steps",
    )

    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-5,
        help="Learning rate",
    )

    parser.add_argument(
        "--number_of_experiments",
        type=int,
        default=5,
        help="Number of experiments (repeat experiments with different random seed and average results)",
    )

    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Use fp16 for training",
    )

    parser.add_argument(
        "--label_all_tokens",
        action="store_true",
        help="Label all tokens",
    )

    parser.add_argument(
        "--run_ner_file",
        type=str,
        default="third_party/run_ner.py",
        help="Output directory",
    )

    parser.add_argument(
        "--lr_scheduler_type",
        type=str,
        default=None,
        help="The scheduler type to use.",
    )

    parser.add_argument(
        "--warmup_ratio",
        type=float,
        default=None,
        help="Linear warmup over warmup_ratio fraction of total steps.",
    )

    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=None,
        help="Linear warmup over warmup_steps.",
    )

    parser.add_argument(
        "--experiment_name",
        type=str,
        default=str(datetime.now()),
        help="Experiment name for logdir",
    )

    parser.add_argument(
        "--encoding",
        type=str,
        default="iob2",
        choices=["iob", "iob2", "bilou"],
        help="Tag format",
    )

    parser.add_argument(
        "--merge_tags",
        action="store_true",
        help="Rewrite B I I O I as B I I I I",
    )

    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=None,
        help="The maximum total input sequence length after tokenization. If set, sequences longer "
        "than this will be truncated, sequences shorter will be padded.",
    )

    parser.add_argument(
        "--deepspeed",
        action="store_true",
        help="use deepspeed stage 2",
    )

    parser.add_argument(
        "--deepspeed_gpu_id",
        type=int,
        default=0,
        help="Which GPU should deepspeed use",
    )

    parser.add_argument(
        "--keep_models",
        action="store_true",
        help="Do not remove the trained models",
    )

    args = parser.parse_args()

    run_seql(
        train_file=args.train_file,
        dev_file=args.dev_file,
        test_file=args.test_file,
        model_name=args.model_name,
        cache_dir=args.cache_dir,
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        fp16=args.fp16,
        encoding=args.encoding,
        merge_tags=args.merge_tags,
        label_all_tokens=args.label_all_tokens,
        number_of_experiments=args.number_of_experiments,
        run_ner_file=args.run_ner_file,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_ratio=args.warmup_ratio,
        warmup_steps=args.warmup_steps,
        experiment_name=args.experiment_name,
        max_seq_length=args.max_seq_length,
        deepspeed=args.deepspeed,
        deepspeed_gpu_id=args.deepspeed_gpu_id,
        save_models=args.keep_models,
    )
