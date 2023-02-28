from torch.utils.data import Dataset, DataLoader
import os
import json
from typing import List, Tuple
from questions import general_category2id, fine2id
from tqdm.auto import tqdm
from transformers import PreTrainedTokenizer, DataCollatorWithPadding


class FlanT5Dataset(Dataset):
    def __init__(
        self,
        json_path: str,
        tokenizer: PreTrainedTokenizer,
        max_len: int = 128,
        label_category: str = "fine",
        include_wikidata_description: bool = False,
        include_wikidata_arguments: bool = False,
        include_wikipedia_summary: bool = True,
    ):

        print(f"Loading sentences from {json_path}")
        json_dict = json.load(open(json_path, "r", encoding="utf8"))
        texts: List[str] = []
        labels: List[str] = []
        positions: List[Tuple[int, int]] = []
        for sentence_no, sentence_dict in json_dict.items():
            for entity_no, entity_dict in sentence_dict["entities"].items():
                if label_category == "fine":
                    labels.append(entity_dict["fine_cat"])
                else:
                    labels.append(entity_dict["general_cat"])

                # [START_ENT] Obama [END_ENT] went to New York [TAB] human, politician [TAB] Obama is a former .....

                words = sentence_dict["words"]

                start_ent = entity_dict["start"]
                end_ent = entity_dict["end"]
                words_label = (
                    words[:start_ent]
                    + ["[START_ENT]"]
                    + words[start_ent:end_ent]
                    + ["[END_ENT]"]
                    + words[end_ent:]
                )

                text = " ".join(words_label)

                if include_wikidata_description:
                    text += " [TAB] " + str(entity_dict["wikidata_summary"])
                if include_wikidata_arguments:
                    text += " [TAB] " + ", ".join(entity_dict["wikidata_arguments"])
                if include_wikipedia_summary:
                    text += " [TAB] " + str(entity_dict["wikipedia_summary"])

                texts.append(text)
                # print(text)
                positions.append((int(sentence_no), int(entity_no)))

        print(f"Tokenizing {len(texts)} sentences")
        self.dataset = []
        for x, y, p in zip(tqdm(texts, desc="Data tokenization"), labels, positions):
            model_inputs = tokenizer(x, max_length=max_len, truncation=True)
            if y == "ENTITY":
                label = -1  # Test set, no label
            else:
                label = (
                    fine2id[y] if label_category == "fine" else general_category2id[y]
                )
            model_inputs["label"] = label
            model_inputs["position"] = p
            self.dataset.append(model_inputs)
        print(f"Dataset loaded with {len(self.dataset)} sentences")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]


def get_dataloader(
    json_path: str,
    tokenizer: PreTrainedTokenizer,
    max_len: int = 512,
    label_category: str = "fine",
    batch_size: int = 32,
    num_workers: int = min(os.cpu_count(), 16),
    shuffle: bool = True,
    include_wikidata_description: bool = False,
    include_wikidata_arguments: bool = False,
    include_wikipedia_summary: bool = True,
):
    dataset = FlanT5Dataset(
        json_path=json_path,
        tokenizer=tokenizer,
        max_len=max_len,
        label_category=label_category,
        include_wikidata_description=include_wikidata_description,
        include_wikidata_arguments=include_wikidata_arguments,
        include_wikipedia_summary=include_wikipedia_summary,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8),
        shuffle=shuffle,
    )
    return dataloader
