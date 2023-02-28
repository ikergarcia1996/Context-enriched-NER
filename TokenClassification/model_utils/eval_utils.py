from typing import List, TextIO, Set
import logging
from numpy import ndarray
from model_utils.utils import run_bash_command
from shlex import quote
import numpy as np
from model_utils.token_classification_hf import inference_token_classification_model
import os
from shutil import rmtree
from tabulate import tabulate
from dataset_format.tags2tsv import tags2tsv
from dataset_format.tsv2json import tsv2json
import glob
import statistics
from dataset_format.tag_encoding import rewrite_tags
from seqeval.metrics import f1_score, classification_report


def print_sentence(
    output: TextIO,
    words: List[str],
    golds: List[str],
    precs: List[str],
    encoding: str,
) -> None:
    assert len(words) == len(golds) == len(precs), (
        f"Error, we have a different number of lens for the words,"
        f" tags, golds and predictions lists. words:"
        f" {words}. golds: {golds}. precs: {precs}"
    )

    golds = rewrite_tags(tags=golds, encoding="IOB2")
    precs = rewrite_tags(tags=precs, encoding="IOB2")

    for word, gold, prec in zip(words, golds, precs):
        print(f"{word} {gold} {prec}", file=output)

    print(file=output)


def eval_conlleval(
    input_path: str, output_path: str, conlleval_script: str = "third_party/conlleval"
) -> None:
    command: str = (
        f"{quote(conlleval_script)} < {quote(input_path)} > {quote(output_path)}"
    )
    run_bash_command(command)


def get_float(s: str) -> float:
    # "98,1%;" -> 98.1
    return float(s.replace("%", "").replace(";", ""))


def get_f1(result_file_path: str) -> float:
    with open(result_file_path) as f:
        f.readline()
        results_line: str = f.readline().rstrip()
        _, acc, _, p, _, r, _, f1 = results_line.split()
    return get_float(f1)


def create_eval_file(
    original_dataset_path: str,
    model_predictions_path: str,
    output_path: str,
    encoding: str,
) -> None:

    with open(original_dataset_path, "r", encoding="utf-8") as original_data, open(
        model_predictions_path, "r", encoding="utf-8"
    ) as model_precs, open(output_path, "w+", encoding="utf-8") as output:
        line_precs: str = model_precs.readline()
        golds: List[str] = []
        precs: List[str] = []
        words: List[str] = []
        line_no = 0
        while line_precs:
            line_no += 1
            gold_line: str = original_data.readline()

            if line_precs == "\n":

                while gold_line != "\n" and gold_line:
                    try:
                        word_gold, _ = gold_line.rstrip().split()
                    except ValueError:
                        try:
                            word_gold, _, _ = gold_line.rstrip().split()
                        except ValueError:
                            raise ValueError(
                                f"Error in line {line_no}. Unable to split line in 2 fields: [{gold_line}]"
                            )
                    try:
                        logging.warning(f"No prediction for the word {word_gold}")
                    except UnicodeError:
                        logging.warning(
                            f"No prediction for the word UKN (unicode error)"
                        )

                    words.append(word_gold)
                    precs.append("O")
                    golds.append(tag_gold)

                    gold_line = original_data.readline()

                print_sentence(output, words, golds, precs, encoding)

                golds = []
                precs = []
                words = []

            else:

                try:
                    word_prec, prec = line_precs.rstrip().split()
                except ValueError:
                    try:
                        word_prec, prec, _ = line_precs.rstrip().split()
                    except ValueError:
                        raise ValueError(
                            f"Error in line {line_no}. Unable to split line in 2 fields: [{gold_line}]"
                        )

                try:
                    word_gold, tag_gold = gold_line.rstrip().split()
                except ValueError:
                    try:
                        word_gold, tag_gold, _ = gold_line.rstrip().split()
                    except ValueError:
                        raise ValueError(
                            f"Error in line {line_no}. Unable to split line in 2 fields: [{gold_line}]"
                        )
                if word_prec == word_gold:
                    words.append(word_prec)
                    precs.append(prec)
                    golds.append(tag_gold)

                else:
                    raise ValueError(
                        f"Error in line {line_no}. Error reading the files: "
                        f"Line gold data: {gold_line}. Line predictions data: {line_precs}"
                    )

            line_precs = model_precs.readline()

        if words and golds and precs:
            print_sentence(output, words, golds, precs, encoding)


def generate_dictionary(
    eval_file_path: str, output_path: str, output_path_incorrect_sentences: str
):
    dictionary = {}
    with open(eval_file_path, "r", encoding="utf8") as eval_file, open(
        output_path, "w+", encoding="utf8"
    ) as output, open(
        output_path_incorrect_sentences, "w+", encoding="utf8"
    ) as output_sentences:

        sentence = []
        correct_sentence = True

        for line in eval_file:
            line = line.rstrip().strip()
            try:
                if line:
                    word, gold, pred = line.split()
                    if word in dictionary:
                        dictionary[word]["Occurrences"] += 1
                    else:
                        dictionary[word] = {
                            "TP": 0,
                            "TN": 0,
                            "FP": 0,
                            "FN": 0,
                            "Occurrences": 1,
                        }
                    if gold != pred:

                        correct_sentence = False
                        if gold == "O":
                            sentence.append(f"\033[94m [{word} -- O as T] \033[0m")
                            dictionary[word]["FP"] += 1
                        else:
                            if pred == "O":
                                dictionary[word]["FN"] += 1
                                sentence.append(f"\033[92m [{word} -- T as O \033[0m]")
                            else:
                                sentence.append(word)
                                if gold == "O":
                                    dictionary[word]["TN"] += 1
                                else:
                                    dictionary[word]["TP"] += 1
                    else:
                        sentence.append(word)
                        if gold == "O":
                            dictionary[word]["TN"] += 1
                        else:
                            dictionary[word]["TP"] += 1
                    if pred != "O":
                        sentence[-1] = f"\033[4m{sentence[-1]}\033[0m"
                else:
                    if not correct_sentence:
                        print(" ".join(sentence), file=output_sentences)

                    sentence = []
                    correct_sentence = True

            except ValueError:
                raise ValueError(f"Error splitting line: {line}")
        table_entries = []
        for item in sorted(
            dictionary.items(),
            key=lambda x: x[1]["FN"] + x[1]["FP"],
            reverse=True,
        ):
            table_entries.append(
                [
                    item[0],
                    item[1]["TP"],
                    item[1]["TN"],
                    item[1]["FP"],
                    item[1]["FN"],
                    item[1]["Occurrences"],
                ]
            )
        print(
            tabulate(
                table_entries,
                headers=[
                    "Word",
                    "TP",
                    "TN",
                    "FP",
                    "FN",
                    "Occurrences",
                ],
            ),
            file=output,
        )


def run_eval_file(
    run_ner_file: str,
    train_json: str,
    eval_file_path: str,
    model_dir: str,
    output_name: str,
    batch_size: int,
    fp16: bool,
    encoding: str = "iob2",
):

    path_tmp: str = os.path.join(model_dir, "tmp")
    if not os.path.exists(path_tmp):
        os.makedirs(path_tmp)

    tsv2json(
        input_path=eval_file_path,
        output_path=os.path.join(path_tmp, "test.json"),
        encoding=encoding,
    )

    inference_token_classification_model(
        run_ner_file=run_ner_file,
        train_file=train_json,
        test_file=os.path.join(path_tmp, "test.json"),
        model_name=model_dir,
        output_dir=model_dir,
        fp16=fp16,
        batch_size=batch_size,
    )

    tags2tsv(
        input_json=os.path.join(path_tmp, "test.json"),
        input_txt=os.path.join(model_dir, "test_predictions.txt"),
        output_path=os.path.join(model_dir, f"{output_name}.model_predictions.tsv"),
        encoding=encoding,
    )

    rmtree(path_tmp)

    create_eval_file(
        original_dataset_path=eval_file_path,
        model_predictions_path=os.path.join(
            model_dir, f"{output_name}.model_predictions.tsv"
        ),
        output_path=os.path.join(model_dir, f"{output_name}.eval_file.tsv"),
        encoding=encoding,
    )

    eval_conlleval(
        input_path=os.path.join(model_dir, f"{output_name}.eval_file.tsv"),
        output_path=os.path.join(model_dir, f"{output_name}.eval_result.txt"),
    )

    generate_dictionary(
        eval_file_path=os.path.join(model_dir, f"{output_name}.eval_file.tsv"),
        output_path=os.path.join(model_dir, f"{output_name}.test_summary.txt"),
        output_path_incorrect_sentences=os.path.join(
            model_dir, f"{output_name}.test_incorrect_sentences.txt"
        ),
    )


def eval_seqeval(
    input_path: str,
    output_path: str,
    set_unique_label: bool = False,
) -> None:
    gold_labels = []
    pred_labels = []
    with open(input_path, "r", encoding="utf8") as f:
        current_gold = []
        current_pred = []
        for line in f:
            line = line.strip()
            if line == "":
                gold_labels.append(current_gold)
                pred_labels.append(current_pred)
                current_gold = []
                current_pred = []
            else:
                word, gold, pred = line.split()
                current_gold.append(gold)
                current_pred.append(pred)

    for i in range(len(gold_labels)):
        gold_labels[i] = rewrite_tags(tags=gold_labels[i], encoding="IOB2")
        pred_labels[i] = rewrite_tags(tags=pred_labels[i], encoding="IOB2")

    if set_unique_label:
        for i in range(len(gold_labels)):
            for j in range(len(gold_labels[i])):
                if gold_labels[i][j] != "O":
                    gold_labels[i][j] = gold_labels[i][j].split("-")[0] + "-ENTITY"
                if pred_labels[i][j] != "O":
                    pred_labels[i][j] = pred_labels[i][j].split("-")[0] + "-ENTITY"

    with open(output_path, "w", encoding="utf8") as f:
        report = classification_report(
            y_true=gold_labels, y_pred=pred_labels, zero_division="1"
        )
        print(
            report,
            file=f,
        )
        print(report)
        f1 = f1_score(y_true=gold_labels, y_pred=pred_labels, zero_division="1")
        print(f"F1 score: {f1} ", file=f)
        print(f"F1 score: {f1} ")


def evaluate_file(
    original_dataset_path: str,
    json_path: str,
    tags_path: str,
    output_dir: str,
    output_name: str,
    encoding: str = "iob2",
    merge_tags: bool = False,
    remove_misc: bool = True,
):

    tags2tsv(
        input_json=json_path,
        input_txt=tags_path,
        output_path=os.path.join(output_dir, f"{output_name}.model_predictions.tsv"),
        encoding=encoding,
        merge_tags=merge_tags,
        remove_misc=remove_misc,
    )

    create_eval_file(
        original_dataset_path=original_dataset_path,
        model_predictions_path=os.path.join(
            output_dir, f"{output_name}.model_predictions.tsv"
        ),
        output_path=os.path.join(output_dir, f"{output_name}.eval_file.tsv"),
        encoding=encoding,
    )

    eval_conlleval(
        input_path=os.path.join(output_dir, f"{output_name}.eval_file.tsv"),
        output_path=os.path.join(output_dir, f"{output_name}.eval_result.txt"),
    )

    eval_seqeval(
        input_path=os.path.join(output_dir, f"{output_name}.eval_file.tsv"),
        output_path=os.path.join(output_dir, f"{output_name}.eval_result_seqeval.txt"),
        set_unique_label=True,
    )

    generate_dictionary(
        eval_file_path=os.path.join(output_dir, f"{output_name}.eval_file.tsv"),
        output_path=os.path.join(output_dir, f"{output_name}.test_summary.txt"),
        output_path_incorrect_sentences=os.path.join(
            output_dir, f"{output_name}.test_incorrect_sentences.txt"
        ),
    )

    print(f"Evaluation:")
    print(f" Original dataset: {original_dataset_path}")
    print("Output path: ", end="")
    print(os.path.join(output_dir, f"{output_name}"))


def evaluate_tsv(
    original_dataset_path: str,
    preds_path: str,
    output_dir: str,
    output_name: str,
    encoding: str,
):

    create_eval_file(
        original_dataset_path=original_dataset_path,
        model_predictions_path=preds_path,
        output_path=os.path.join(output_dir, f"{output_name}.eval_file.tsv"),
        encoding=encoding,
    )

    eval_conlleval(
        input_path=os.path.join(output_dir, f"{output_name}.eval_file.tsv"),
        output_path=os.path.join(output_dir, f"{output_name}.eval_result.txt"),
    )

    generate_dictionary(
        eval_file_path=os.path.join(output_dir, f"{output_name}.eval_file.tsv"),
        output_path=os.path.join(output_dir, f"{output_name}.test_summary.txt"),
        output_path_incorrect_sentences=os.path.join(
            output_dir, f"{output_name}.test_incorrect_sentences.txt"
        ),
    )

    print(f"Evaluation:")
    print(f" Original dataset: {original_dataset_path}")
    print("Output path: ", end="")
    print(os.path.join(output_dir, f"{output_name}"))



def create_table_avg(
    output_dir: str,
    num_iters: int,
    output_path: str,
    name: str = "",
    replace: bool = False,
) -> None:

    files = glob.glob(os.path.join(os.path.join(output_dir, "0"), "*.eval_result.txt"))
    files = [f for f in files if "2" not in os.path.basename(f)]
    files.sort()
    results: List[List[float]] = [
        [-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0] for _ in files
    ]

    for file_n, eval_file in enumerate(files):
        iters_acc: List[float] = []
        iters_prec: List[float] = []
        iters_rec: List[float] = []
        iters_f1: List[float] = []
        for i in range(0, num_iters):
            it_dir: str = os.path.join(output_dir, f"{i}")
            with open(
                os.path.join(it_dir, os.path.basename(eval_file)), encoding="utf-8"
            ) as f:
                f.readline()
                results_line: str = f.readline().rstrip()
                _, acc, _, p, _, r, _, f1 = results_line.split()
                acc = get_float(acc)
                p = get_float(p)
                r = get_float(r)
                f1 = get_float(f1)
                if f1 < 5:
                    print(
                        f"Found f1 < 5: {f1}. File: {eval_file}. We will skip this result"
                    )
                else:
                    iters_acc.append(acc)
                    iters_prec.append(p)
                    iters_rec.append(r)
                    iters_f1.append(f1)

                f.close()

        if len(iters_acc) > 0:
            avg_acc = sum(iters_acc) / len(iters_acc)
            avg_prec = sum(iters_prec) / len(iters_prec)
            avg_rec = sum(iters_rec) / len(iters_rec)
            avg_f1 = sum(iters_f1) / len(iters_f1)
            stdev_acc = statistics.stdev(iters_acc)
            stdev_prec = statistics.stdev(iters_prec)
            stdev_rec = statistics.stdev(iters_rec)
            stdev_f1 = statistics.stdev(iters_f1)
        else:
            avg_acc = -1.0
            avg_prec = -1.0
            avg_rec = -1.0
            avg_f1 = -1.0
            stdev_acc = -1.0
            stdev_prec = -1.0
            stdev_rec = -1.0
            stdev_f1 = -1.0

        results[file_n] = [
            avg_acc,
            stdev_acc,
            avg_prec,
            stdev_prec,
            avg_rec,
            stdev_rec,
            avg_f1,
            stdev_f1,
        ]

    if not os.path.exists(output_path) or replace:
        with open(output_path, "w+", encoding="utf-8") as file:
            print(
                "ModelName,"
                + ",".join(
                    [
                        ".".join(os.path.basename(filename).split(".")[:2]) + metric
                        for filename in files
                        for metric in [
                            "-Acc",
                            "-Acc_stdev" "-P",
                            "-P_stdev",
                            "-R",
                            "R-stdev",
                            "-F1",
                            "-F1_stdev",
                        ]
                    ]
                ),
                file=file,
            )

    with open(output_path, "a+", encoding="utf-8") as file:
        print(
            name
            + ","
            + ",".join([str(result) for metrics in results for result in metrics]),
            file=file,
        )
    [print(result) for result in results]
    output_path_f1 = os.path.splitext(output_path)[0] + "_f1.csv"
    if not os.path.exists(output_path_f1) or replace:
        with open(output_path_f1, "w+", encoding="utf-8") as file:
            print(
                "ModelName,"
                + ",".join(
                    [
                        ".".join(os.path.basename(filename).split(".")[:2]) + metric
                        for filename in files
                        for metric in [
                            "-F1",
                            "-F1_stdev",
                        ]
                    ]
                ),
                file=file,
            )

    with open(output_path_f1, "a+", encoding="utf-8") as file:
        f1r = [l[-2:] for l in results]

        print(
            name
            + ","
            + ",".join([str(result) for metrics in f1r for result in metrics]),
            file=file,
        )


def create_table_relax(
    output_dir: str,
    num_iters: int,
    output_path: str,
    name: str = "",
) -> None:

    files = glob.glob(
        os.path.join(os.path.join(output_dir, "0"), "*.RELAX_eval_result.txt")
    )
    files.sort()
    results: ndarray = np.array([[0.0, 0.0, 0.0, 0.0] for _ in files])

    for i in range(0, num_iters):
        it_dir: str = os.path.join(output_dir, f"{i}")
        for file_n, eval_file in enumerate(files):
            with open(
                os.path.join(it_dir, os.path.basename(eval_file)), encoding="utf-8"
            ) as f:
                f.readline()
                results_line: str = f.readline().rstrip()
                _, acc, _, p, _, r, _, f1, _, wf1 = results_line.split()
                results[file_n] += np.array(
                    [get_float(acc), get_float(p), get_float(r), get_float(f1)]
                )
                f.close()

    results = results / num_iters

    print(results)

    if not os.path.exists(output_path):
        with open(output_path, "w+", encoding="utf-8") as file:
            print(
                "ModelName,"
                + ",".join(
                    [
                        ".".join(os.path.basename(filename).split(".")[:2]) + metric
                        for filename in files
                        for metric in ["-Acc", "-P", "-R", "-F1"]
                    ]
                ),
                file=file,
            )

            print(
                name
                + ","
                + ",".join([str(result) for metrics in results for result in metrics]),
                file=file,
            )
    else:

        with open(output_path, "a+", encoding="utf-8") as file:
            print(
                name
                + ","
                + ",".join([str(result) for metrics in results for result in metrics]),
                file=file,
            )


def get_dictionary_tags(file_path: str) -> Set[str]:

    tags_list: List[str] = []

    with open(file_path, "r", encoding="utf8") as dataset_file:
        current_tag: List[str] = []
        for line in dataset_file:
            line: str = line.rstrip().strip()
            word: str
            tag: str
            if line:
                try:
                    word, tag = line.split()
                except ValueError:
                    raise ValueError(f"Error splitting line: {line}")

                if tag.startswith("B"):
                    if current_tag:
                        tags_list.append(" ".join(current_tag).lower())

                    current_tag = [tag]

                if tag.startswith("I"):
                    current_tag.append(tag)

                if tag.startswith("O"):
                    if current_tag:
                        tags_list.append(" ".join(current_tag).lower())
                        current_tag = []
            else:
                if current_tag:
                    tags_list.append(" ".join(current_tag).lower())
                    current_tag = []

    return set(tags_list)
