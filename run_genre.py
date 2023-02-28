import sys

sys.path.append("GENRE/")
sys.path.append("fairseq/")

import os
from typing import List, Dict, Union, TextIO
import json
import torch
from GENRE.genre.fairseq_model import mGENRE
import pickle
from GENRE.genre.trie import Trie
from questions import fine2general
from tqdm.auto import tqdm
import argparse


def read_sentence(
    file: TextIO,
) -> Dict[str, Union[List[str], Dict[int, Dict[str, str]]]]:
    """
    Output:
    {
        'words': ["Obama","went","to","New","York"]
        'labels': ["B-OtherPer","O","O","B-HumanSettlement","I-HumanSettlement"]
        'entities':{
                    0:{
                        "text": "Obama"
                        "start": 0
                        "end": 1
                        "general_cat": Person
                        "fine_cat": OtherPer
                        "genre_prediction": None
                        "wikidata_summary": None
                        "wikidata_arguments": None
                        "wikipedia_title": None
                        "wikipedia_summary": None
                        }
                    1:{
                        "text": "New York"
                        "start": 3
                        "end": 5
                        "general_cat": Location
                        "fine_cat": HumanSettlement
                        "genre_prediction": None
                        "wikidata_summary": None
                        "wikidata_arguments": None
                        "wikipedia_title": None
                        "wikipedia_summary": None
                        }
                    }
    }
    """
    words: List[str] = []
    labels: List[str] = []
    entities: Dict[int, Dict[str, str]] = {}

    current_entity = []
    current_entity_start = None
    current_entity_end = None
    current_entity_fine_cat = None
    line: str = file.readline().rstrip().strip()
    while line:

        word: str
        label: str
        try:
            word, label = line.split()  # CONLL
        except ValueError:
            try:
                word, label, _ = line.split()  # Manual-projection
            except ValueError:
                try:
                    word, _, _, label = line.split()  # MultiCoNER
                except ValueError:
                    raise ValueError(f"Error splitting line: {line}")

        if (label.startswith("B-") or label == "O") and len(current_entity) > 0:
            current_entity_end = len(words)
            entities[len(entities)] = {
                "text": " ".join(current_entity),
                "start": current_entity_start,
                "end": current_entity_end,
                "general_cat": fine2general[current_entity_fine_cat]
                if current_entity_fine_cat != "ENTITY"
                else "ENTITY",
                "fine_cat": current_entity_fine_cat,
                "genre_prediction": None,
                "wikidata_summary": None,
                "wikidata_arguments": None,
                "wikipedia_title": None,
                "wikipedia_summary": None,
            }
            current_entity = []
            current_entity_start = None
            current_entity_end = None

        if label.startswith("B-"):
            current_entity_fine_cat = label[2:]
            current_entity.append(word)
            current_entity_start = len(words)
        elif label.startswith("I-"):
            current_entity.append(word)
        elif label == "O":
            pass
        else:
            raise ValueError(f"Unknown label: {label}")

        words.append(word)
        labels.append(label)

        line = file.readline().rstrip().strip()

    if len(current_entity) > 0:
        current_entity_end = len(words)
        entities[len(entities)] = {
            "text": " ".join(current_entity),
            "start": current_entity_start,
            "end": current_entity_end,
            "general_cat": fine2general[current_entity_fine_cat]
            if current_entity_fine_cat != "ENTITY"
            else "ENTITY",
            "fine_cat": current_entity_fine_cat,
            "genre_prediction": None,
            "wikidata_summary": None,
            "wikidata_arguments": None,
            "wikipedia_title": None,
            "wikipedia_summary": None,
        }

    return {"words": words, "labels": labels, "entities": entities}


def create_dataset(
    entity_tsv_path,
) -> Dict[
    int, Dict[str, Union[List[str], Dict[int, Dict[str, Union[str, List[str]]]]]]
]:
    """
    sentence_dic:
    {
    0: {
    'words': ["Obama","went","to","New","York"]
    'labels': ["B-OtherPer","O","O","B-HumanSettlement","I-HumanSettlement"]
    'entities':{
                0:{
                    "text": "Obama"
                    "start": 0
                    "end": 1
                    "general_cat": Person
                    "fine_cat": OtherPer
                    "genre_prediction": None
                    "wikidata_summary": None
                    "wikidata_arguments": None
                    "wikipedia_title": None
                    "wikipedia_summary": None

                    }
                1:{
                    "text": "New York"
                    "start": 3
                    "end": 5
                    "general_cat": Location
                    "fine_cat": HumanSettlement
                    "genre_prediction": None
                    "wikidata_summary": None
                    "wikidata_arguments": None
                    "wikipedia_title": None
                    "wikipedia_summary": None
                    }
                }
        }
    }
    """

    sentence_dict: Dict[
        int, Dict[str, Union[List[str], Dict[int, Dict[str, Union[str, List[str]]]]]]
    ] = {}

    with open(entity_tsv_path, "r") as f:
        sentence_id = 0
        sentence = read_sentence(f)
        while len(sentence["words"]) > 0:
            sentence_dict[sentence_id] = sentence
            sentence_id += 1
            sentence = read_sentence(f)

    return sentence_dict


def generate_genre_inputs(
    sentence_dict: Dict[
        int, Dict[str, Union[List[str], Dict[int, Dict[str, Union[str, List[str]]]]]]
    ]
) -> List[List[Union[str, int]]]:
    genre_inputs: List[
        List[Union[str, int]]
    ] = []  # [START_ENT] Obama  [END_ENT] went to New York, 0, 0

    w_prediction = 0
    wo_prediction = 0
    for sentence_id, sentence in sentence_dict.items():
        for entity_id, entity in sentence["entities"].items():
            if entity["genre_prediction"] is None:
                start_ent = entity["start"]
                end_ent = entity["end"]
                input_text = (
                    sentence["words"][:start_ent]
                    + ["[START]"]
                    + sentence["words"][start_ent:end_ent]
                    + ["[END]"]
                    + sentence["words"][end_ent:]
                )
                genre_inputs.append([" ".join(input_text), sentence_id, entity_id])
                wo_prediction += 1
            else:
                w_prediction += 1

    print(f"Number of entities with genre prediction: {w_prediction}")
    print(f"Number of entities without genre prediction: {wo_prediction}")

    return genre_inputs


def generate_genre_predictions(
    dataset_path: str,
    batch_size: int,
    num_beams: int = 8,
):

    print("Loading dataset...")
    if dataset_path.endswith(".tsv"):
        sentence_dict = create_dataset(dataset_path)
    elif dataset_path.endswith(".json"):
        print(f"FOUND JSON DATASET; LOADING {dataset_path}")
        with open(dataset_path, "r", encoding="utf8") as f:
            sentence_dict = json.load(f)
    genre_inputs = generate_genre_inputs(sentence_dict)

    print("Loading TRIE")
    with open(
        "./genre_titles/lang_title2wikidataID-normalized_with_redirect.pkl",
        "rb",
    ) as f:
        lang_title2wikidataID = pickle.load(f)
    with open(
        "./genre_titles/titles_lang_all105_trie_with_redirect.pkl",
        "rb",
    ) as f:
        trie = Trie.load_from_dict(pickle.load(f))

    print("Loading model")
    model = (
        mGENRE.from_pretrained(
            "./genre_models/fairseq_multilingual_entity_disambiguation"
        )
        .eval()
        .cuda()
    )

    print("Model loaded")

    print("Dataset loaded")

    print("Generating genre predictions...")
    with torch.no_grad():
        for i in tqdm(
            range(0, len(genre_inputs), batch_size), desc="Generating genre predictions"
        ):
            batch = genre_inputs[i : i + batch_size]
            batch_text = [x[0] for x in batch]
            batch_sentence_ids = [x[1] for x in batch]
            batch_entity_ids = [x[2] for x in batch]
            try:
                model_outputs = model.sample(
                    batch_text,
                    prefix_allowed_tokens_fn=lambda batch_id, sent: [
                        e
                        for e in trie.get(sent.tolist())
                        if e < len(model.task.target_dictionary)
                    ],
                    text_to_id=lambda x: max(
                        lang_title2wikidataID[tuple(reversed(x.split(" >> ")))],
                        key=lambda y: int(y[1:]),
                    ),
                    marginalize=True,
                    beam=num_beams,
                )

                for sentence_id, entity_id, prediction in zip(
                    batch_sentence_ids, batch_entity_ids, model_outputs
                ):
                    sentence_dict[sentence_id]["entities"][entity_id][
                        "genre_prediction"
                    ] = [str(x["id"]) for x in prediction]

            except (IndexError, KeyError, ValueError):
                # I didn't find why it happens, but sometimes one of the beams in genre is not generated, so instead
                # of having a list of 8 predictions, we have a list of 7 predictions. This causes an IndexError in
                # genre code. As a workaround, we run the sentences in the batch one by one to find the problematic
                # sentence and we run it with num_beams=1, this usually solves the problem. Another workaround is to
                # run the model unconstrained, but it may generate invalid wikidata entries, which is bad. If it
                # still fails, we add the key Q0, which doesn't exits and in the next step will be replaced by
                # 'no wikidata summary found'.
                print(
                    f"Error with batch {i} - {i+batch_size}\n"
                    f"Batch_text: {batch_text}\n"
                    f"We will run the batch one by one"
                )
                for sentence_id, entity_id, text in zip(
                    batch_sentence_ids, batch_entity_ids, batch_text
                ):
                    try:
                        prediction = model.sample(
                            [text],
                            prefix_allowed_tokens_fn=lambda batch_id, sent: [
                                e
                                for e in trie.get(sent.tolist())
                                if e < len(model.task.target_dictionary)
                            ],
                            text_to_id=lambda x: max(
                                lang_title2wikidataID[tuple(reversed(x.split(" >> ")))],
                                key=lambda y: int(y[1:]),
                            ),
                            marginalize=True,
                            beam=num_beams,
                        )

                        sentence_dict[sentence_id]["entities"][entity_id][
                            "genre_prediction"
                        ] = [str(x["id"]) for x in prediction[0]]
                    except (IndexError, KeyError, ValueError):
                        print(
                            f"Error with sentence {sentence_id} - {text}\n"
                            f"We will run the sentence with a beam of 1 to attempt to get a prediction"
                        )
                        try:
                            prediction = model.sample(
                                [text],
                                prefix_allowed_tokens_fn=lambda batch_id, sent: [
                                    e
                                    for e in trie.get(sent.tolist())
                                    if e < len(model.task.target_dictionary)
                                ],
                                text_to_id=lambda x: max(
                                    lang_title2wikidataID[
                                        tuple(reversed(x.split(" >> ")))
                                    ],
                                    key=lambda y: int(y[1:]),
                                ),
                                marginalize=True,
                                beam=1,
                            )
                            sentence_dict[sentence_id]["entities"][entity_id][
                                "genre_prediction"
                            ] = [str(x["id"]) for x in prediction[0]]
                        except (IndexError, KeyError, ValueError) as err:
                            print(
                                f"Error with sentence {sentence_id} - {text}\n"
                                f"We will skip this sentence\n"
                                f"Error: {err}"
                            )
                            sentence_dict[sentence_id]["entities"][entity_id][
                                "genre_prediction"
                            ] = ["Q0"]

    print("Done!")
    return sentence_dict


def generate_dataset(
    tsv_path: str,
    batch_size: int,
    output_path: str,
):

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    sentence_dict = generate_genre_predictions(
        dataset_path=tsv_path,
        batch_size=batch_size,
    )
    # sentence_dict = get_wiki_summary(sentence_dict, language)
    print(f"Saving dataset to {output_path}")
    with open(output_path, "w", encoding="utf8") as f:
        json.dump(sentence_dict, f, indent=4, ensure_ascii=False)

    print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tsv_path",
        type=str,
        required=True,
        help="Path to the tsv file containing the dataset",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        required=True,
        help="Batch size for the model",
    )

    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to the output file",
    )

    args = parser.parse_args()

    generate_dataset(
        tsv_path=args.tsv_path,
        batch_size=args.batch_size,
        output_path=args.output_path,
    )
