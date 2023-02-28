import re
from typing import List, TextIO, Optional
from tag_encoding import rewrite_labels
import sys
import string


def split_sentence(
    tag_regex,
    sentence: str,
    recursion_limit: int = 10,
) -> List[str]:

    sentence = sentence.strip().split()

    if recursion_limit == 0:
        return sentence

    new_sentence: List[str] = []

    for word in sentence:
        search_result = tag_regex.search(word)
        if search_result:
            span = search_result.span()

            l = word[: span[0]].strip()
            r = word[span[1] :].strip()
            t = word[span[0] : span[1]].strip()
            if l:
                new_sentence.extend(split_sentence(tag_regex, l, recursion_limit - 1))
            new_sentence.append(t)
            if r:
                new_sentence.extend(split_sentence(tag_regex, r, recursion_limit - 1))

        else:
            new_sentence.append(word)

    return new_sentence


def get_label_type(label: str) -> (str, bool):
    label = label.strip()
    is_start = not label.startswith("</")
    if is_start:
        label_type = label[1:-1]
    else:
        label_type = label[2:-1]

    return label_type, is_start


def get_labelled_words(words: List[str], labels: List[str]) -> (List[str], List[str]):
    assert len(words) == len(labels), (
        f"Words and labels must have the same length. "
        f"len(words)={len(words)}, "
        f"len(labels)={len(labels)}. "
        f"words: {words}, "
        f"labels: {labels}"
    )

    labelled_entities: List[str] = []
    labelled_entities_labels: List[str] = []
    current_label: List[str] = []

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

    assert len(labelled_entities) == len(labelled_entities_labels), (
        f"Something went wrong retrieving labelled words. "
        f"len(labelled_entities)={len(labelled_entities)}, "
        f"len(labelled_entities_labels)={len(labelled_entities_labels)}.\n"
        f"words: {words}\n"
        f"labels: {labels}\n"
        f"labelled_entities: {labelled_entities}\n"
        f"labelled_entities_labels: {labelled_entities_labels}.\n"
    )

    return labelled_entities, labelled_entities_labels


def labelled_sentence_2_iob2(
    prediction: str, possible_labels: List[str]
) -> (List[str], List[str]):
    """
    Input
    <Person>Obama</Person> went to <Location>New York</Location> .
    Output
    ["Obama", "went", "to", "New York", "."]
    ["B-PER","O","O","B-LOC","I-LOC","O"]
    ["Obama", "New York"]
    ["Person", "Location"]
    """
    inside_tag: bool = False
    current_label_type: str = ""
    tag_regex = re.compile(f"</?({'|'.join([p for p in possible_labels])})>")
    predicted_words: List[str] = split_sentence(tag_regex, prediction)

    first = True
    i = 0
    words: List[str] = []
    tags: List[str] = []
    for word in predicted_words:

        result = tag_regex.match(word)
        if result:
            label_type, is_start = get_label_type(word)
            if is_start:
                inside_tag = True
                current_label_type = label_type
                first = True
            else:
                inside_tag = False
        else:
            if inside_tag:
                if first:
                    tags.append(f"B-{current_label_type}")
                    first = False
                else:
                    tags.append(f"I-{current_label_type}")
            else:
                tags.append("O")
            words.append(word)
            i += 1

    labelled_entities, labelled_entities_labels = get_labelled_words(
        words=words, labels=tags
    )

    return words, tags, labelled_entities, labelled_entities_labels


def get_sentence(
    file: TextIO, set_unique_label: bool = False
) -> (List[str], List[str], List[str], List[str]):
    words: List[str] = []
    labels: List[str] = []

    line: str = file.readline().rstrip().strip()
    while line:
        # print(line)
        if line.startswith("-DOCSTART-"):
            next(file)
            line = file.readline().rstrip().strip()
            continue

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

        words.append(word)
        labels.append(label)

        line = file.readline().rstrip().strip()

    labels = rewrite_labels(labels, encoding="iob2")

    labelled_entities: List[str] = []
    labelled_entities_labels: List[str] = []
    current_label: List[str] = []

    if set_unique_label:
        new_labels = []
        for label in labels:
            if label != "O":
                new_labels.append(f"{label[:1]}-ENTITY")
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


def read_all_sentences_tsv(
    dataset_path: str, set_unique_label: bool = False
) -> (List[List[str]], List[List[str]], List[List[str]], List[List[str]]):
    print(f"Reading dataset from {dataset_path}.")
    sentences_words: List[List[str]] = []
    sentences_labels: List[List[str]] = []
    sentences_labelled_entities: List[List[str]] = []
    sentences_labelled_entities_labels: List[List[str]] = []

    with open(dataset_path, "r", encoding="utf-8") as dataset_file:
        words, labels, labelled_entities, labelled_entities_labels = get_sentence(
            file=dataset_file,
            set_unique_label=set_unique_label,
        )
        while words:
            sentences_words.append(words)
            sentences_labels.append(labels)
            sentences_labelled_entities.append(labelled_entities)
            sentences_labelled_entities_labels.append(labelled_entities_labels)
            words, labels, labelled_entities, labelled_entities_labels = get_sentence(
                file=dataset_file,
                set_unique_label=set_unique_label,
            )

    print(f"Read {len(sentences_words)} sentences from {dataset_path}.")

    return (
        sentences_words,  # ["Obama","went","to","New","York"]
        sentences_labels,  # ["B-PER","O","O","B-LOC","I-LOC"]
        sentences_labelled_entities,  # ["Obama","New York"]
        sentences_labelled_entities_labels,  # ["PER","LOC"]
    )


def subfinder(
    mylist: List[str],
    pattern: List[str],
    tags: Optional[List[str]] = None,
) -> List[int]:
    matches: List[int] = []

    if len(pattern) == 0:
        return matches

    for i in range(min(len(mylist), sys.maxsize if tags is None else len(tags))):
        if (
            (mylist[i] == pattern[0] and (tags is None or tags[i] == "O"))
            and mylist[i : i + len(pattern)] == pattern
            and (
                tags is None
                or all(
                    [tags[j] == "O" for j in range(i, min(i + len(pattern), len(tags)))]
                )
            )
        ):
            matches.append(i)

    if len(matches) == 0:
        # Lower everything and remove punctuation
        mylist = [
            x.lower().translate(str.maketrans("", "", string.punctuation))
            for x in mylist
        ]
        mylist = [x for x in mylist if x != ""]
        pattern = [
            x.lower().translate(str.maketrans("", "", string.punctuation))
            for x in pattern
        ]
        pattern = [x for x in pattern if x != ""]
        if len(pattern) == 0 or len(mylist) == 0:
            return matches
        for i in range(min(len(mylist), sys.maxsize if tags is None else len(tags))):
            if (
                (mylist[i] == pattern[0] and (tags is None or tags[i] == "O"))
                and mylist[i : i + len(pattern)] == pattern
                and (
                    tags is None
                    or all(
                        [
                            tags[j] == "O"
                            for j in range(i, min(i + len(pattern), len(tags)))
                        ]
                    )
                )
            ):
                matches.append(i)

        # print(mylist, pattern, matches)
    return matches
