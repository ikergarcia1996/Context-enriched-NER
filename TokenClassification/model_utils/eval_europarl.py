import os
from dataset_format.tsv2json import tsv2json
from model_utils.utils import count_lines
from model_utils.eval_utils import evaluate_file
from typing import Dict, List
import json
from model_utils.relax_seqeval import relax_eval_file

lang_codes = {
    "en": "English",
    "bg": "Bulgarian",
    "cs": "Czech",
    "da": "Danish",
    "de": "German",
    "el": "Greek",
    "es": "Spanish",
    "et": "Estonian",
    "fi": "Finish",
    "fr": "French",
    "hu": "Hungarian",
    "it": "Italian",
    "lt": "Lithuanian",
    "lv": "Latvian",
    "nl": "Dutch",
    "pl": "Polish",
    "pt": "Portuguese",
    "ro": "Romanian",
    "sk": "Slovak",
    "sl": "Slovene",
    "sv": "Swedish",
}


def generate_test_europarl(
    europarl_corpus_path: str,
    output_dir: str,
    encoding: str = "iob2",
):
    index_path: str = os.path.join(output_dir, "index.json")
    json_path: str = os.path.join(output_dir, "test.json")

    rows_dict: Dict[str, List[str, int, int]] = {}
    json_line: int = 0

    with open(json_path, "w+", encoding="utf8") as json_file:

        for lang in lang_codes.keys():
            lang_dir: str = europarl_corpus_path
            test_name: str = f"{lang}.tsv"
            test_tsv_path: str = os.path.join(lang_dir, test_name)
            test_json_path: str = os.path.join(
                output_dir, os.path.splitext(test_name)[0] + ".json"
            )
            lines_no: int = tsv2json(
                input_path=test_tsv_path, output_path=test_json_path, encoding=encoding
            )
            rows_dict[test_json_path] = [
                test_tsv_path,
                os.path.basename(os.path.splitext(test_tsv_path)[0]),
                json_line,
                json_line + lines_no,
            ]
            json_line += lines_no

            print(
                open(test_json_path, "r", encoding="utf8").read(),
                end="",
                file=json_file,
            )

    with open(index_path, "w+", encoding="utf8") as index_file:
        json.dump(rows_dict, index_file)


def eval_europarl(output_dir: str, encoding: str = "iob2", merge_tags: bool = False):
    index_path: str = os.path.join(output_dir, "index.json")
    tags_path: str = os.path.join(output_dir, "predictions.txt")
    current_line: int = 0

    with open(index_path, "r") as index_file:
        rows_dict: Dict[str, List[str, int, int]] = json.load(index_file)

    with open(tags_path, "r", encoding="utf8") as tags_file:
        for json_dataset_path, (
            tsv_dataset_path,
            output_name,
            start_line,
            end_line,
        ) in rows_dict.items():
            currentfile_tags_path = os.path.splitext(json_dataset_path)[0] + ".tags"
            with open(
                currentfile_tags_path, "w+", encoding="utf8"
            ) as currentfile_tags_file:
                while current_line != end_line:
                    print(tags_file.readline(), end="", file=currentfile_tags_file)
                    current_line += 1

            if "evalita" in tsv_dataset_path or "msra" in tsv_dataset_path:
                remove_misc = True
            else:
                remove_misc = False

            evaluate_file(
                original_dataset_path=tsv_dataset_path,
                json_path=json_dataset_path,
                tags_path=currentfile_tags_path,
                output_dir=output_dir,
                output_name=output_name,
                encoding=encoding,
                merge_tags=merge_tags,
                remove_misc=remove_misc,
            )

            os.remove(currentfile_tags_path)
            os.remove(json_dataset_path)
