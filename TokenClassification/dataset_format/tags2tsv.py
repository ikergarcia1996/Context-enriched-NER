from typing import List, Set, TextIO
import json
import argparse
from dataset_format.tag_encoding import rewrite_tags


def get_sentence_json(line: str) -> (List[str], List[str]):
    set: Set[str : List[str]] = json.loads(line.rstrip().strip())
    return set["words"], set["ner_tags"]


def get_tags(line: str) -> (List[str], List[str]):
    return line.rstrip().strip().split()


def to_tsv(words: List[str], gold_tags: List[str], pred_tags: List[str]) -> str:
    if not (len(words) == len(gold_tags) == len(pred_tags)):
        try:
            print(
                f" WARNING (tags2tsv)! len(words): {len(words)}. len(gold_tags): {len(gold_tags)}. len(pred_tags): {len(pred_tags)}.\n"
                f"words:{words}. gold_tags: {gold_tags}. pred_tags:{pred_tags}."
            )
        except UnicodeError:
            print("WARNING (tags2tsv)! UnicodeError")

    if len(pred_tags) < len(gold_tags):
        pred_tags = pred_tags + ["O"] * (len(gold_tags) - len(pred_tags))
    if len(pred_tags) > len(gold_tags):
        raise ValueError("pred_tags longer than gold_tags, something is wrong")

    assert len(words) == len(gold_tags) == len(pred_tags)

    return "\n".join([f"{word} {ptag}" for word, ptag in zip(words, pred_tags)]) + "\n"


def fix_tags(
    tags: List[str],
    encoding: str = "iob2",
    merge_tags: bool = False,
    remove_misc: bool = False,
) -> List[str]:

    # Merge I tags
    # B I I O I -> B I I I I
    if merge_tags:
        for i in range(1, len(tags) - 1):
            if (
                tags[i] == "O"
                and (tags[i - 1].startswith("B") or tags[i - 1].startswith("I"))
                and tags[i + 1].startswith("I")
                and tags[i - 1].split("-")[-1] == tags[i + 1].split("-")[-1]
            ):
                tags[i] = tags[i + 1]

    tags = rewrite_tags(tags=tags, encoding=encoding, remove_misc=remove_misc)

    return tags


def tags2tsv(
    input_json: str,
    input_txt: str,
    output_path: str,
    encoding: str = "iob2",
    merge_tags: bool = False,
    remove_misc: bool = False,
):
    with open(input_json, "r", encoding="utf8") as input_json:
        with open(input_txt, "r", encoding="utf8") as input_txt:
            with open(output_path, "w+", encoding="utf8") as output_file:

                for line_json, line_txt in zip(input_json, input_txt):
                    words, gold_tags = get_sentence_json(line_json)
                    pred_tags = get_tags(line_txt)
                    pred_tags = fix_tags(
                        tags=pred_tags, encoding=encoding, merge_tags=merge_tags
                    )
                    print(
                        to_tsv(words=words, gold_tags=gold_tags, pred_tags=pred_tags),
                        file=output_file,
                    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="tags 2 .tsv")
    parser.add_argument(
        "--input_json",
        type=str,
        required=True,
        help=".json dataset path",
    )

    parser.add_argument(
        "--input_txt",
        type=str,
        required=True,
        help=".txt tags path",
    )

    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help=".tsv output dataset path",
    )

    args = parser.parse_args()

    tags2tsv(
        input_json=args.input_json,
        input_txt=args.input_txt,
        output_path=args.output_path,
    )
