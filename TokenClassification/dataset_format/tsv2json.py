from typing import List, TextIO, Set
import json
import argparse
from dataset_format.tag_encoding import rewrite_tags


def get_sentence_tsv(file: TextIO) -> (List[str], List[str]):
    words: List[str] = []
    tags: List[str] = []

    line: str = file.readline().rstrip().strip()
    while line:
        word: str
        tag: str
        if len(line) > 2:
            try:
                word, tag = line.split()
            except ValueError:
                try:
                    word, tag, _ = line.split()
                except ValueError:
                    try:
                        tag = line.split()[0]
                        if not (tag.startswith("B-") or tag.startswith("I-")):
                            raise ValueError
                        word = None
                    except ValueError:
                        raise ValueError(f"Error splitting line: {line}")

            if word is not None:
                words.append(word)
                tags.append(tag)
            else:
                print(f"Warning: skipping line: {line}")
        else:
            print(f"WARNING, SKIPPING LINE: {line}")

        line = file.readline().rstrip().strip()

    return words, tags


def to_json(words: List[str], tags: List[str]) -> str:
    json_format: Set[str : List[str]] = {"words": words, "ner_tags": tags}
    return json.dumps(json_format)


def tsv2json(
    input_path: str,
    output_path: str,
    encoding: str = "iob2",
) -> int:
    number_of_sentences: int = 0
    with open(input_path, "r", encoding="utf8") as input_file:
        with open(output_path, "w+", encoding="utf8") as output_file:

            words, tags = get_sentence_tsv(input_file)

            while words and tags:
                assert len(words) == len(tags)
                tags = rewrite_tags(tags=tags, encoding=encoding)
                print(to_json(words=words, tags=tags), file=output_file)
                number_of_sentences += 1
                words, tags = get_sentence_tsv(input_file)

    return number_of_sentences


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=".tsv 2 .json")
    parser.add_argument(
        "--input_path",
        type=str,
        required=True,
        help=".tsv dataset path",
    )

    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help=".json output dataset path",
    )

    args = parser.parse_args()

    tsv2json(input_path=args.input_path, output_path=args.output_path)
