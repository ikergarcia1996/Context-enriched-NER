import argparse
import os
from typing import List, TextIO
from tqdm import tqdm
from tag_encoding import rewrite_labels
from multiprocessing import Pool
from fine2ent import dataset2general
from fine2general import dataset2general


def get_sentence_multiconer(
    file: TextIO, set_unique_label: bool = False
) -> (List[str], List[str], List[str], List[str]):
    words: List[str] = []
    labels: List[str] = []

    line: str = file.readline().rstrip().strip()
    i = 0
    while len(line) == 0 and i < 10:
        line: str = file.readline().rstrip().strip()
        i += 1

    while line:
        if line.startswith("# id"):
            line = file.readline().rstrip().strip()
            continue

        word: str
        label: str
        try:
            word, label = line.split()
        except ValueError:
            try:
                word, label, _ = line.split()
            except ValueError:
                try:
                    word, _, _, label = line.split()
                except ValueError:
                    try:
                        word, _, _ = line.split()
                        label = "O"
                    except ValueError:
                        raise ValueError(f"Error splitting line: {line}")
        if label == "_":
            label = "O"
        words.append(word)
        labels.append(label)

        line = file.readline().rstrip().strip()

    # _ = file.readline().rstrip().strip()
    labels = rewrite_labels(labels, encoding="iob2")

    labelled_entities: List[str] = []
    labelled_entities_labels: List[str] = []
    current_label: List[str] = []

    if set_unique_label:
        new_labels = []
        for label in labels:
            if label != "O":
                new_labels.append(f"{label[:1]}-TARGET")
            else:
                new_labels.append(label)
        labels = new_labels

    for word, label in zip(words, labels):
        if label.startswith("B-") or label.startswith("U-"):
            if current_label:
                labelled_entities.append(" ".join(current_label))

            current_label = [word]
            labelled_entities_labels.append(f"{label[2:]}")

        elif label.startswith("I-") or label.startswith("L-"):
            current_label.append(word)
        else:
            if current_label:
                labelled_entities.append(" ".join(current_label))
                current_label = []

    if current_label:
        labelled_entities.append(" ".join(current_label))

    assert len(words) == len(labels), (
        f"Error redding sentence. "
        f"len(words)={len(words)}, "
        f"len(labels)={len(labels)}. "
        f"words: {words}, "
        f"labels: {labels}"
    )
    assert len(labelled_entities) == len(labelled_entities_labels), (
        f"Error redding sentence. "
        f"len(labelled_entities)={len(labelled_entities)}, "
        f"len(labelled_entities_labels)={len(labelled_entities_labels)}.\n"
        f"words: {words}\n"
        f"labels: {labels}\n"
        f"labelled_entities: {labelled_entities}\n"
        f"labelled_entities_labels: {labelled_entities_labels}.\n"
        f"file: {file.name}"
    )

    return words, labels, labelled_entities, labelled_entities_labels


def read_all_sentences_tsv_multiconer(
    dataset_path: str, set_unique_label: bool = False
) -> (List[List[str]], List[List[str]], List[List[str]], List[List[str]]):
    # print(f"Reading dataset from {dataset_path}.")
    sentences_words: List[List[str]] = []
    sentences_labels: List[List[str]] = []
    sentences_labelled_entities: List[List[str]] = []
    sentences_labelled_entities_labels: List[List[str]] = []

    with open(dataset_path, "r", encoding="utf-8") as dataset_file:
        (
            words,
            labels,
            labelled_entities,
            labelled_entities_labels,
        ) = get_sentence_multiconer(
            file=dataset_file,
            set_unique_label=set_unique_label,
        )
        while words:
            sentences_words.append(words)
            sentences_labels.append(labels)
            sentences_labelled_entities.append(labelled_entities)
            sentences_labelled_entities_labels.append(labelled_entities_labels)

            (
                words,
                labels,
                labelled_entities,
                labelled_entities_labels,
            ) = get_sentence_multiconer(
                file=dataset_file, set_unique_label=set_unique_label
            )

    # print(f"Read {len(sentences_words)} sentences from {dataset_path}.")

    return (
        sentences_words,
        sentences_labels,
        sentences_labelled_entities,
        sentences_labelled_entities_labels,
    )


def preprocess(args):
    data_path, directory_path = args
    lang_id = directory_path.split("-")[0].lower()
    for split in ["train", "dev", "test"]:
        input_path = os.path.join(data_path, directory_path, f"{lang_id}_{split}.conll")

        output_path = os.path.join(data_path, f"finegrained/{lang_id}_{split}.conll")

        sentences_words, sentences_labels, _, _ = read_all_sentences_tsv_multiconer(
            input_path
        )

        with open(output_path, "w", encoding="utf8") as output_file:
            for words, labels in zip(sentences_words, sentences_labels):
                for word, label in zip(words, labels):
                    print(f"{word} {label}", file=output_file)
                print(file=output_file)

        dataset2general(output_path, output_path.replace("finegrained", "entity"))
        dataset2general(output_path, output_path.replace("finegrained", "general"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--dataset_path",
        type=str,
        default="multiconer2023",
        help="Dataset to preprocess",
    )

    args = parser.parse_args()

    os.makedirs(os.path.join(args.dataset_path, "finegrained"), exist_ok=True)
    os.makedirs(os.path.join(args.dataset_path, "entity"), exist_ok=True)
    os.makedirs(os.path.join(args.dataset_path, "general"), exist_ok=True)

    print(f"Preprocessing dataset from {args.dataset_path}.")
    # Iterate for every folder in dataset_path
    directories = [
        x
        for x in os.listdir(args.dataset_path)
        if os.path.isdir(os.path.join(args.dataset_path, x))
    ]

    # Process directories in parallel. Set the number of processes to 4 to avoid memory issues. Print progress bar with tqdm.

    with Pool(processes=4) as p, tqdm(
        total=len(directories), desc="Preprocessing data"
    ) as pbar:
        for _ in tqdm(
            p.imap_unordered(
                preprocess,
                [
                    (args.dataset_path, directory_path)
                    for directory_path in directories
                    if directory_path not in ["finegrained", "entity", "general"]
                ],
            )
        ):
            pbar.update()
            print()

    print(
        f"We have finished preprocessing the dataset from {args.dataset_path}. "
        f"The preprocessed dataset is in {args.dataset_path}/finegrained, "
        f"{args.dataset_path}/entity and {args.dataset_path}/general."
    )
