from typing import List, Dict, TextIO

import numpy as np


def get_tags(file: TextIO) -> List[str]:
    sentence: List[str] = []
    tags: List[str] = []

    line: str = file.readline().rstrip().strip()
    while line:
        word: str
        tag: str

        try:
            word, tag = line.split(" ")
            sentence.append(word)
            tags.append(tag)
        except ValueError:
            try:
                word, tag, _ = line.split("\t")
                sentence.append(word)
                tags.append(tag)
            except ValueError:
                raise ValueError(f"Error splitting line: {line}")

        line = file.readline().rstrip().strip()

    return tags


def get_all_tags(file_path: str):

    sentence_tags = []
    with open(file_path, "r", encoding="utf8") as file:
        tags = get_tags(file)

        while tags:
            sentence_tags.append(tags)
            tags = get_tags(file)

    return sentence_tags


def get_all_tags_model(file_path: str):
    sentence_tags = []
    with open(file_path, "r", encoding="utf8") as file:
        for line in file:
            sentence_tags.append(line.rstrip().strip().split(" "))

    return sentence_tags


def get_entities(tags: List[str]) -> (List[List[int]], List[str]):
    entities: List[List[int]] = []
    entity_types: List[str] = []

    for tag_no, tag in enumerate(tags):

        if tag.startswith("B") or tag.startswith("U"):
            try:
                _, tag_type = tag.split("-")
            except ValueError:
                raise ValueError(f"Unable to split tag: {tag} \n {tags}")

            entities.append([tag_no])
            entity_types.append(tag_type)

        elif tag.startswith("I"):
            try:
                _, tag_type = tag.split("-")
            except ValueError:
                raise ValueError(f"Unable to split tag: {tag} \n {tags}")

            if entities[-1][-1] == (tag_no - 1) and entity_types[-1] == tag_type:
                entities[-1].append(tag_no)
            else:
                entities.append([tag_no])
                entity_types.append(tag_type)

    return entities, entity_types


def overlap(x: List[int], y: List[int]) -> int:
    return max(0, min(x[-1], y[-1]) + 1 - max(x[0], y[0]))


class RelaxEvaluator:

    entity2id: Dict[str, int]
    id2entity: Dict[int, str]
    tag_no: int
    matrix: np.array
    tokens: int
    phrases: int
    correct_phrases: int

    def __init__(self, entity_types: List[str]):
        self.entity2id = {
            entity: entity_no for entity_no, entity in enumerate(entity_types + ["O"])
        }
        self.id2entity = {
            entity_no: entity for entity, entity_no in self.entity2id.items()
        }

        self.tag_no: int = len(entity_types)

        self.matrix = np.zeros((self.tag_no + 1, self.tag_no + 1))
        self.tokens = 0
        self.phrases = 0
        self.correct_phrases = 0

    def add_sentence(self, gold_tags: List[str], pred_tags: List[str]):
        if len(gold_tags) > len(pred_tags):
            print(
                f"[WARNING Relax Eval] Pred tags len ({len(pred_tags)}) > Gold tags len ({len(gold_tags)}). "
                f"We will truncate gold tags. "
            )
            gold_tags = gold_tags[: len(pred_tags)]

        assert len(gold_tags) == len(pred_tags), (
            f"Gold tags and Pred tags must have the same len. "
            f"Gold tags ({len(gold_tags)}: {gold_tags}. "
            f"Pred tags ({len(pred_tags)}: {pred_tags}"
        )

        # gold_entities, gold_types = get_entities(gold_tags)
        # pred_entities, pred_types = get_entities(pred_tags)

        self.tokens += len(gold_tags)
        self.phrases += 1
        correct = True
        match_gold = False
        match_pred = False
        gold_type = "O"
        pred_type = "O"

        for gold_tag, pred_tag in zip(gold_tags + ["O"], pred_tags + ["O"]):
            if gold_tag == "O":
                if gold_type != "O" and not match_gold:
                    tid = self.entity2id[gold_type]
                    oid = self.entity2id["O"]
                    self.matrix[tid][oid] += 1
                    correct = False

                gold_type = "O"
                match_gold = False

            else:
                try:
                    gold_s, gold_t = gold_tag.split("-")
                except ValueError:
                    raise ValueError(
                        f"Unable to split gold_tag: {gold_tag} \n {gold_tags}"
                    )

                if (
                    gold_s == "B"
                    or gold_s == "U"
                    or (gold_s == "I" and gold_type != gold_t)
                ):
                    if gold_type != "O" and not match_gold:
                        tid = self.entity2id[gold_type]
                        oid = self.entity2id["O"]
                        self.matrix[tid][oid] += 1
                        correct = False
                    match_gold = False

                if not match_gold:
                    if pred_tag != "O":
                        try:
                            pred_s, pred_t = pred_tag.split("-")
                        except ValueError:
                            raise ValueError(
                                f"Unable to split pred_tag: {pred_tag} \n {pred_tags}"
                            )

                        if pred_t == gold_t:
                            match_gold = True
                            tid = self.entity2id[gold_t]
                            self.matrix[tid][tid] += 1

                gold_type = gold_t

            if pred_tag == "O":
                if pred_type != "O" and not match_pred:
                    tid = self.entity2id[pred_type]
                    oid = self.entity2id["O"]
                    self.matrix[oid][tid] += 1
                    correct = False

                pred_type = "O"
                match_pred = False

            else:
                try:
                    pred_s, pred_t = pred_tag.split("-")
                except ValueError:
                    raise ValueError(
                        f"Unable to split pred_tag: {pred_tag} \n {pred_tags}"
                    )

                if (
                    pred_s == "B"
                    or pred_s == "U"
                    or (pred_s == "I" and pred_type != pred_t)
                ):
                    if pred_type != "O" and not match_pred:
                        tid = self.entity2id[pred_type]
                        oid = self.entity2id["O"]
                        self.matrix[oid][tid] += 1
                        correct = False

                    match_pred = False

                if not match_pred and gold_tag != "O":
                    try:
                        gold_s, gold_t = gold_tag.split("-")
                    except ValueError:
                        raise ValueError(
                            f"Unable to split gold_tag: {gold_tag} \n {gold_tags}"
                        )

                    if gold_t == pred_t:
                        match_pred = True

                pred_type = pred_t

        if correct:
            self.correct_phrases += 1

    def add_corpus(self, gold_tags: List[List[str]], pred_tags: List[List[str]]):
        for gold, pred in zip(gold_tags, pred_tags):
            self.add_sentence(gold_tags=gold, pred_tags=pred)

    def get_report(self, output_file: str = None):
        macro_precision = 0
        macro_recall = 0
        macro_f1 = 0
        weighed_f1 = 0
        entity_no = np.sum(self.matrix, axis=1)[:-1]
        tags_result = []
        found_tags = 0
        for i in range(self.tag_no):
            tag = self.id2entity[i]
            div = np.sum(self.matrix[:, i])
            if div == 0:
                precision = -1
            else:
                precision = self.matrix[i][i] / div

            div = np.sum(self.matrix[i])
            if div == 0:
                recall = -1
            else:
                recall = self.matrix[i][i] / div

            if recall == -1 or precision == -1:
                f1 = -1
            else:
                f1 = 2 * ((precision * recall) / (precision + recall))

            if f1 != -1:
                macro_precision += precision
                macro_recall += recall
                macro_f1 += f1
                weighed_f1 += f1 * entity_no[i]
                tags_result.append(
                    f"{tag}: precision:  {round(precision*100,2)}%; "
                    f"recall:  {round(recall*100,2)}%; "
                    f"FB1:  {round(f1*100,2)}  {entity_no[i]}"
                )
                found_tags += 1

        if found_tags == 0:
            raise ValueError(
                f"No tag has been recorded, report cannot be generated.\n {self.matrix}"
            )

        micro_tp = np.sum(np.diagonal(self.matrix))
        micro_f1 = np.sum(micro_tp) / np.sum(self.matrix)

        fstr = (
            f"processed {self.tokens} tokens with {self.phrases} phrases; "
            f"found: {self.phrases} phrases; correct: {self.correct_phrases}."
        )

        rstr = (
            f"accuracy:  {round(micro_f1*100,2)}%; "
            f"precision:  {round((macro_precision/found_tags)*100,2)}%; "
            f"recall:  {round((macro_recall/found_tags)*100,2)}%; "
            f"FB1:  {round((macro_f1/found_tags)*100,2)} "
            f"wFB1: {round((weighed_f1/np.sum(entity_no))*100,2)}"
        )

        print("\n".join([fstr, rstr] + tags_result))
        if output_file:
            with open(output_file, "w+", encoding="utf8") as output_file:
                print("\n".join([fstr, rstr] + tags_result), file=output_file)


def relax_eval_file(gold_path: str, pred_path: str, output_path: str, tags: List[str]):
    gold_tags = get_all_tags(gold_path)
    pred_tags = get_all_tags_model(pred_path)
    reval = RelaxEvaluator(entity_types=tags)
    reval.add_corpus(gold_tags=gold_tags, pred_tags=pred_tags)
    reval.get_report(output_file=output_path)
