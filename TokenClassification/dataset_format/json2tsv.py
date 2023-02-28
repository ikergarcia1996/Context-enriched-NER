from typing import List, Set
import json
import argparse
from dataset_format.tag_encoding import rewrite_tags
from dataset_format.tags2tsv import fix_tags


def get_sentence(
    line: str,
    encoding: str = "iob2",
    merge_tags: bool = False,
) -> (List[str], List[str]):
    set: Set[str : List[str]] = json.loads(line.rstrip().strip())
    return set["words"], fix_tags(
        set["ner_tags"],
        encoding=encoding,
        merge_tags=merge_tags,
    )


def to_tsv(words: List[str], tags: List[str]) -> str:
    return "\n".join([f"{word} {tag}" for word, tag in zip(words, tags)])


def json2tsv(
    input_path: str,
    output_path: str,
    encoding: str = "iob2",
    merge_tags: bool = False,
    block_size=65536,
):
    with open(input_path, "r", encoding="utf8") as input_file:
        with open(output_path, "w+", encoding="utf8") as output_file:

            lines: List[str] = input_file.readlines(block_size)

            while lines:
                print(
                    "\n\n".join(
                        [
                            to_tsv(
                                *get_sentence(
                                    line,
                                    encoding=encoding,
                                    merge_tags=merge_tags,
                                )
                            )
                            for line in lines
                        ]
                    ),
                    file=output_file,
                )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=".json 2 .tsv")
    parser.add_argument(
        "--input_path",
        type=str,
        required=True,
        help=".json dataset path",
    )

    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help=".tsv output dataset path",
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

    args = parser.parse_args()

    json2tsv(input_path=args.input_path, output_path=args.output_path)
