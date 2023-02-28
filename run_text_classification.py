from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_scheduler,
)
from dataset import get_dataloader
from questions import (
    id2fine,
    id2general_category,
    fine2id,
    general_category2id,
    fine2general,
)
import os
import torch
from seqeval.metrics import f1_score, classification_report
import json
from typing import List
import numpy as np
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
from accelerate import Accelerator
from torch.optim import AdamW
from tqdm.auto import tqdm
import math
import argparse
from utils import read_all_sentences_tsv
from tag_encoding import rewrite_labels


def compute_metrics(
    json_path, pred_labels, positions, output_path, category: str = "fine"
):
    json_dict = json.load(open(json_path, "r", encoding="utf8"))

    l = len(id2fine) if category == "fine" else len(id2general_category)
    cat_matrix = np.zeros((l, l), dtype=np.int32)
    total: int = 0
    correct: int = 0
    # Save predictions
    for (sentence_no, entity_no), pred_label in zip(positions, pred_labels):
        sentence_no = str(sentence_no)
        entity_no = str(entity_no)
        json_dict[sentence_no]["entities"][entity_no]["pred_label"] = (
            id2fine[pred_label]
            if category == "fine"
            else id2general_category[pred_label]
        )
        pred_idx = pred_label
        true_idx = (
            fine2id[json_dict[sentence_no]["entities"][entity_no]["fine_cat"]]
            if category == "fine"
            else general_category2id[
                json_dict[sentence_no]["entities"][entity_no]["general_cat"]
            ]
        )
        cat_matrix[true_idx, pred_idx] += 1
        if pred_idx == true_idx:
            correct += 1
        total += 1

    # Write predicted labels in iob format

    gold_labels: List[List[str]] = []
    pred_labels: List[List[str]] = []

    for sentence_no, sentence_dict in json_dict.items():
        sentence_dict["pred_labels"] = ["O"] * len(sentence_dict["words"])
        for entity_no, entity_dict in sentence_dict["entities"].items():
            start = entity_dict["start"]
            end = entity_dict["end"]
            sentence_dict["pred_labels"][start] = "B-" + entity_dict["pred_label"]
            for i in range(start + 1, end):
                sentence_dict["pred_labels"][i] = "I-" + entity_dict["pred_label"]
        if category == "fine":
            gold_labels.append(sentence_dict["labels"])
        else:
            labels = []
            for label in sentence_dict["labels"]:
                if label == "O":
                    labels.append(label)
                else:
                    p, l = label.split("-")
                    labels.append(f"{p}-{fine2general[l]}")
            gold_labels.append(labels)
        pred_labels.append(sentence_dict["pred_labels"])

    with open(output_path + ".json", "w", encoding="utf8") as f:
        json.dump(json_dict, f, ensure_ascii=False, indent=4)

    with open(output_path + ".tsv", "w") as f:
        for sentence_no, sentence_dict in json_dict.items():
            for word, label in zip(
                sentence_dict["words"], sentence_dict["pred_labels"]
            ):
                print(f"{word}\t{label}", file=f)
            print(file=f)

    print(f"Predicted labels saved in {output_path}")

    f1 = f1_score(y_true=gold_labels, y_pred=pred_labels, zero_division="1")
    print(f"F1 score: {round(f1*100,2)}")
    print(
        f"Accuracy: {round((correct / total)*100,2)}. Correct: {correct}. Total: {total}"
    )
    with open(output_path + ".txt", "w", encoding="utf8") as f:
        print(
            classification_report(
                y_true=gold_labels, y_pred=pred_labels, zero_division="1"
            ),
            file=f,
        )
        print(f"F1 score: {f1} ", file=f)
        print(
            f"Accuracy: {correct / total}. Correct: {correct}. Total: {total}", file=f
        )
        print(f"Classification report saved in {output_path}")

    json.dump(json_dict, open(output_path, "w", encoding="utf8"), indent=4)

    names = id2fine if category == "fine" else id2general_category
    names = [names[i] for i in names.keys()]

    df_cm = pd.DataFrame(cat_matrix, index=names, columns=names)
    plt.figure(figsize=(20, 25))
    sn.heatmap(df_cm, annot=True, fmt="g")
    plt.savefig(output_path + ".png")
    print(f"Confusion matrix saved in {output_path}")

    return f1


def text_classification(
    model_name_or_path: str,
    output_dir: str,
    train_json_path: str = None,
    dev_json_path: str = None,
    test_json_path: str = None,
    label_category: str = "fine",
    train_batch_size: int = 32,
    gradient_accumulation_steps: int = 1,
    eval_batch_size: int = 32,
    learning_rate: float = 5e-5,
    num_train_epochs: int = 3,
    max_len: int = 512,
    num_workers: int = min(os.cpu_count(), 16),
    do_not_save_model: bool = False,
    include_wikidata_description: bool = False,
    include_wikidata_arguments: bool = False,
    include_wikipedia_summary: bool = True,
):
    assert label_category in ["fine", "general"]
    assert (
        train_json_path is not None
        or dev_json_path is not None
        or test_json_path is not None
    ), "At least one of train_json_path, dev_json_path, test_json_path must be provided"
    assert (
        train_json_path is not None or dev_json_path is not None
    ), "train_json_path and dev_json_path must be provided together"

    accelerator = Accelerator()

    print(f"Loading tokenizer and model from {model_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    num_labels = len(id2fine) if label_category == "fine" else len(id2general_category)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name_or_path, num_labels=num_labels
    )

    print("Adding [START_ENT], [END_ENT] and [TAB] tokens")
    print(f"Tokenizer len before: {len(tokenizer)}")
    tokenizer.add_tokens(["[START_ENT]", "[END_ENT]", "[TAB]"])
    model.resize_token_embeddings(len(tokenizer))
    print(f"Tokenizer len after: {len(tokenizer)}")

    if train_json_path is not None:
        print(f"Loading train data from {train_json_path}")
        train_dataloader = get_dataloader(
            json_path=train_json_path,
            tokenizer=tokenizer,
            max_len=max_len,
            label_category=label_category,
            batch_size=train_batch_size,
            num_workers=num_workers,
            shuffle=True,
            include_wikidata_description=include_wikidata_description,
            include_wikidata_arguments=include_wikidata_arguments,
            include_wikipedia_summary=include_wikipedia_summary,
        )
        print(f"Loading dev data from {dev_json_path}")
        dev_dataloader = get_dataloader(
            json_path=dev_json_path,
            tokenizer=tokenizer,
            max_len=max_len,
            label_category=label_category,
            batch_size=train_batch_size,
            num_workers=num_workers,
            shuffle=False,
            include_wikidata_description=include_wikidata_description,
            include_wikidata_arguments=include_wikidata_arguments,
            include_wikipedia_summary=include_wikipedia_summary,
        )

        # TRAIN

        num_update_steps_per_epoch = math.ceil(
            math.ceil(len(train_dataloader) / gradient_accumulation_steps)
            / accelerator.num_processes
        )

        max_train_steps = num_train_epochs * num_update_steps_per_epoch

        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]

        optimizer = AdamW(
            params=optimizer_grouped_parameters, lr=learning_rate, eps=1e-7
        )

        model, optimizer, train_dataloader, dev_dataloader = accelerator.prepare(
            model, optimizer, train_dataloader, dev_dataloader
        )

        lr_scheduler = get_scheduler(
            name="cosine",
            optimizer=optimizer,
            num_warmup_steps=100,
            num_training_steps=max_train_steps,
        )

        completed_steps = 0

        best_epoch_metric: float = -1
        validation_dir: str = os.path.join(output_dir, "val_logs")
        os.makedirs(validation_dir, exist_ok=True)

        running_loss = 0
        num_batches = 0

        print(
            f"==============================\n"
            f"TRAINING MODEL: {model_name_or_path}\n"
            f"output_dir: {output_dir}\n"
            f"train_json_path: {train_json_path}\n"
            f"dev_json_path: {dev_json_path}\n"
            f"num_labels: {num_labels}\n"
            f"max_train_steps: {max_train_steps}\n"
            f"train_batch_size: {train_batch_size}\n"
            f"gradient_accumulation_steps: {gradient_accumulation_steps}\n"
            f"learning_rate: {learning_rate}\n"
            f"num_epochs: {num_train_epochs}\n"
            f"max_len: {max_len}\n"
            f"include_wikidata_description: {include_wikidata_description}\n"
            f"include_wikidata_arguments: {include_wikidata_arguments}\n"
            f"include_wikipedia_summary: {include_wikipedia_summary}\n"
            f"==============================\n"
        )

        progress_bar = tqdm(
            range(max_train_steps),
            disable=not accelerator.is_local_main_process,
            ascii=True,
            desc="Training, running loss: 0.0",
        )

        for epoch in range(num_train_epochs):
            model.train()
            for step, model_inputs in enumerate(train_dataloader):
                outputs = model(
                    input_ids=model_inputs["input_ids"],
                    labels=model_inputs["labels"],
                    attention_mask=model_inputs["attention_mask"],
                )
                loss = outputs.loss
                loss = loss / gradient_accumulation_steps
                accelerator.backward(loss)
                running_loss += loss.item()
                num_batches += 1

                if (
                    step % gradient_accumulation_steps == 0
                    or step == len(train_dataloader) - 1
                ):
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
                    progress_bar.update(1)
                    completed_steps += 1
                    progress_bar.set_description(
                        f"Training, running loss: {round(running_loss / num_batches, 4)}"
                    )

            # EVAL
            model.eval()
            samples_seen = 0
            with torch.no_grad():
                predictions: List[int] = []
                positions: List[List[int]] = []
                for step, model_inputs in enumerate(
                    tqdm(
                        dev_dataloader,
                        disable=not accelerator.is_local_main_process,
                        ascii=True,
                        desc="Evaluating",
                    )
                ):
                    outputs = model(
                        input_ids=model_inputs["input_ids"],
                        attention_mask=model_inputs["attention_mask"],
                    )

                    logits = outputs.logits
                    prediction = torch.argmax(logits, dim=-1)

                    prediction = accelerator.gather(prediction).cpu().tolist()
                    position = (
                        accelerator.gather(model_inputs["position"]).cpu().tolist()
                    )

                    if accelerator.is_main_process:
                        if accelerator.num_processes > 1:
                            if step == len(dev_dataloader) - 1:
                                prediction = prediction[
                                    : (len(dev_dataloader.dataset) - samples_seen)
                                ]
                                position = position[
                                    : (len(dev_dataloader.dataset) - samples_seen)
                                ]
                        else:
                            samples_seen += len(prediction)

                        predictions.extend(prediction)
                        positions.extend(position)

                if accelerator.is_main_process:
                    f1 = compute_metrics(
                        json_path=dev_json_path,
                        pred_labels=predictions,
                        positions=positions,
                        category=label_category,
                        output_path=os.path.join(validation_dir, f"epoch_{epoch}"),
                    )
                    print(f"Epoch {epoch} - F1: {f1}")
                    if f1 > best_epoch_metric:
                        print(
                            f"New best epoch - {model_name_or_path}. "
                            f"lr:{learning_rate}. "
                            f"bs:{train_batch_size*gradient_accumulation_steps*accelerator.num_processes}. "
                            f"max_len:{max_len}. "
                            f"incl_wikidata_desc:{include_wikidata_description}. "
                            f"incl_wikidata_args:{include_wikidata_arguments}. "
                            f"incl_wikipedia_sum:{include_wikipedia_summary}. "
                            f"!!! F1: {f1}"
                        )
                        best_epoch_metric = f1
                        if not do_not_save_model:
                            print(f"Saving model to {output_dir}")
                            unwrapped_model = accelerator.unwrap_model(model)
                            unwrapped_model.save_pretrained(
                                output_dir, save_function=accelerator.save
                            )
                            tokenizer.save_pretrained(output_dir)

    if test_json_path is not None:
        print(f"Loading test data from {test_json_path}")
        test_dataloader = get_dataloader(
            json_path=test_json_path,
            tokenizer=tokenizer,
            max_len=max_len,
            label_category=label_category,
            batch_size=eval_batch_size,
            num_workers=num_workers,
            shuffle=False,
            include_wikidata_description=include_wikidata_description,
            include_wikidata_arguments=include_wikidata_arguments,
            include_wikipedia_summary=include_wikipedia_summary,
        )

        model.eval()
        model, test_dataloader = accelerator.prepare(model, test_dataloader)

        samples_seen = 0
        with torch.no_grad():
            predictions: List[int] = []
            positions: List[List[int]] = []
            for step, model_inputs in enumerate(
                tqdm(
                    test_dataloader,
                    disable=not accelerator.is_local_main_process,
                    ascii=True,
                    desc="Testing",
                )
            ):
                outputs = model(
                    input_ids=model_inputs["input_ids"],
                    attention_mask=model_inputs["attention_mask"],
                )

                logits = outputs.logits
                prediction = torch.argmax(logits, dim=-1)

                prediction = accelerator.gather(prediction).cpu().tolist()
                position = accelerator.gather(model_inputs["position"]).cpu().tolist()

                if accelerator.is_main_process:
                    if accelerator.num_processes > 1:
                        if step == len(test_dataloader) - 1:
                            prediction = prediction[
                                : (len(test_dataloader.dataset) - samples_seen)
                            ]
                            position = position[
                                : (len(test_dataloader.dataset) - samples_seen)
                            ]
                    else:
                        samples_seen += len(prediction)

                    predictions.extend(prediction)
                    positions.extend(position)

            if accelerator.is_main_process:
                json_dict = json.load(open(test_json_path, "r", encoding="utf8"))
                for (sentence_no, entity_no), pred_label in zip(positions, predictions):
                    sentence_no = str(sentence_no)
                    entity_no = str(entity_no)
                    json_dict[sentence_no]["entities"][entity_no]["pred_label"] = (
                        id2fine[pred_label]
                        if label_category == "fine"
                        else id2general_category[pred_label]
                    )
                with open(os.path.join(output_dir, "test_predictions.tsv"), "w") as f:
                    for sentence_no, sentence_dict in json_dict.items():
                        sentence_dict["pred_labels"] = ["O"] * len(
                            sentence_dict["words"]
                        )
                        for entity_no, entity_dict in sentence_dict["entities"].items():
                            start = entity_dict["start"]
                            end = entity_dict["end"]
                            sentence_dict["pred_labels"][start] = (
                                "B-" + entity_dict["pred_label"]
                            )
                            for i in range(start + 1, end):
                                sentence_dict["pred_labels"][i] = (
                                    "I-" + entity_dict["pred_label"]
                                )

                        for word, label in zip(
                            sentence_dict["words"], sentence_dict["pred_labels"]
                        ):
                            print(f"{word}\t{label}", file=f)
                        print(file=f)
                print(
                    f"Predictions saved to {os.path.join(output_dir, 'test_predictions.tsv')}"
                )

                with open(os.path.join(output_dir, "test_predictions.json"), "w") as f:
                    json.dump(json_dict, f, indent=4)
                print(
                    f"Predictions dictionary saved to {os.path.join(output_dir, 'test_predictions.json')}"
                )


def text_classification_main(
    model_name_or_path: str,
    output_dir: str,
    train_json_path: str = None,
    dev_json_path: str = None,
    test_json_path: str = None,
    label_category: str = "fine",
    train_batch_size: int = 32,
    gradient_accumulation_steps: int = 1,
    eval_batch_size: int = 32,
    learning_rate: float = 5e-5,
    num_train_epochs: int = 3,
    max_len: int = 512,
    num_workers: int = min(os.cpu_count(), 16),
    do_not_save_model: bool = False,
    include_wikidata_description: bool = False,
    include_wikidata_arguments: bool = False,
    include_wikipedia_summary: bool = True,
    number_of_experiments: int = 1,
):

    assert number_of_experiments > 0, "number_of_experiments must be > 0"

    for i in range(number_of_experiments):
        print(f"Experiment {i+1}/{number_of_experiments}")

        text_classification(
            model_name_or_path=model_name_or_path,
            output_dir=os.path.join(output_dir, str(i)),
            train_json_path=train_json_path,
            dev_json_path=dev_json_path,
            test_json_path=test_json_path,
            label_category=label_category,
            train_batch_size=train_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            eval_batch_size=eval_batch_size,
            learning_rate=learning_rate,
            num_train_epochs=num_train_epochs,
            max_len=max_len,
            num_workers=num_workers,
            do_not_save_model=do_not_save_model,
            include_wikidata_description=include_wikidata_description,
            include_wikidata_arguments=include_wikidata_arguments,
            include_wikipedia_summary=include_wikipedia_summary,
        )

    # Combine experiments
    labels: List[List[List[str]]] = []
    words: List[List[List[str]]] = []
    for i in range(number_of_experiments):
        (
            sentences_words,
            sentences_labels,
            sentences_labelled_entities,
            sentences_labelled_entities_labels,
        ) = read_all_sentences_tsv(
            os.path.join(output_dir, str(i), "test_predictions.tsv")
        )

        if len(labels) == 0:
            for _ in range(len(sentences_labels)):
                labels.append([])
                words.append([])

        for j in range(len(sentences_labels)):
            labels[j].append(sentences_labels[j])
            words[j].append(sentences_words[j])

    with open(os.path.join(output_dir, "predictions.tsv"), "w") as tsv_file, open(
        os.path.join(output_dir, "predictions.conll"), "w"
    ) as labels_file:
        for sentence_labels, sentence_words in zip(labels, words):
            voted_labels = []
            assert len(sentence_labels) == number_of_experiments
            assert all(
                len(sentence_labels[0]) == len(sentence_labels[i])
                for i in range(number_of_experiments)
            )
            assert all(
                len(sentence_words[0]) == len(sentence_words[i])
                for i in range(number_of_experiments)
            )

            for i in range(len(sentence_labels[0])):
                possible_labels = [x[i] for x in sentence_labels]
                voted_labels.append(
                    max(set(possible_labels), key=possible_labels.count)
                )

            voted_labels = rewrite_labels(voted_labels, encoding="iob2")
            for word, label in zip(sentence_words[0], voted_labels):
                print(f"{word} {label}", file=tsv_file)
                print(f"{label}", file=labels_file)
            print(file=tsv_file)
            print(file=labels_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train_json_path",
        type=str,
        default=None,
        help="Path to the training data in json file",
    )
    parser.add_argument(
        "--dev_json_path",
        type=str,
        default=None,
        help="Path to the development data in json file",
    )
    parser.add_argument(
        "--test_json_path",
        type=str,
        default=None,
        help="Path to the test data in json file",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="xlmr-roberta-large",
        help="Path to the pre-trained model or model identifier from huggingface.co/models",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--label_category",
        type=str,
        default="fine",
        choices=["fine", "general"],
        help="Whether to use fine-grained or general labels",
    )
    parser.add_argument(
        "--max_len",
        type=int,
        default=512,
        help="The maximum total input sequence length after tokenization. Sequences longer than this will be "
        "truncated, sequences shorter will be padded.",
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=32,
        help="Batch size per GPU/CPU for training.",
    )
    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=32,
        help="Batch size per GPU/CPU for evaluation.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="The initial learning rate for AdamW.",
    )
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=4,
        help="Total number of training epochs to perform.",
    )
    parser.add_argument(
        "--do_not_save_model",
        action="store_true",
        help="Disable model checkpoint saving",
    )
    parser.add_argument(
        "--include_wikidata_description",
        action="store_true",
        help="Whether to include Wikidata description in the input",
    )
    parser.add_argument(
        "--include_wikidata_arguments",
        action="store_true",
        help="Whether to include Wikidata arguments in the input",
    )
    parser.add_argument(
        "--include_wikipedia_summary",
        action="store_true",
        help="Whether to include Wikipedia summary in the input",
    )

    parser.add_argument(
        "--number_of_experiments",
        type=int,
        default=1,
        help="Number of experiments to run",
    )

    args = parser.parse_args()
    text_classification_main(
        model_name_or_path=args.model_name_or_path,
        train_json_path=args.train_json_path,
        dev_json_path=args.dev_json_path,
        test_json_path=args.test_json_path,
        output_dir=args.output_dir,
        label_category=args.label_category,
        max_len=args.max_len,
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        do_not_save_model=args.do_not_save_model,
        include_wikidata_description=args.include_wikidata_description,
        include_wikidata_arguments=args.include_wikidata_arguments,
        include_wikipedia_summary=args.include_wikipedia_summary,
        number_of_experiments=args.number_of_experiments,
    )
