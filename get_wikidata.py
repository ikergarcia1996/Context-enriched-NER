from wikidata.client import Client
from typing import List, Dict, Union
import os
import json
from mediawiki import MediaWiki, DisambiguationError, HTTPTimeoutError, PageError
from tqdm.auto import tqdm
import time
import argparse
from multiprocessing import Pool
import pickle
import multiprocessing
import urllib
from urllib.error import HTTPError
from requests.exceptions import ReadTimeout
import mediawiki

languages2save = [
    "bn",
    "de",
    "en",
    "es",
    "fa",
    "fr",
    "hi",
    "it",
    "pt",
    "sv",
    "uk",
    "zh",
    "eu",
]


class WikiDataRetriever:
    def __init__(self, ignore_cache=False):
        self.client = Client()
        self.cache = {}
        self.argument_cache = {}
        self.ignore_cache = ignore_cache
        if not self.ignore_cache:
            self.wikidata_cache_path = f"/scratch/igarcia945/GENRE/wikidata_cache.json"
            try:
                os.makedirs(os.path.dirname(self.wikidata_cache_path), exist_ok=True)
            except PermissionError:
                print(f"Could not create directory {self.wikidata_cache_path}")
                print("Using current directory instead")
                self.wikidata_cache_path = f"wikidata_cache.json"

            if os.path.exists(self.wikidata_cache_path):
                # print("Loading Wikidata cache...")
                with open(self.wikidata_cache_path, "r", encoding="utf8") as f:
                    self.cache = json.load(f)
                # print(f"Loaded {len(self.cache)} items from cache")

            self.argument_cache_path = f"/scratch/igarcia945/GENRE/argument_cache.json"
            try:
                os.makedirs(os.path.dirname(self.argument_cache_path), exist_ok=True)
            except PermissionError:
                print(f"Could not create directory {self.argument_cache_path}")
                print("Using current directory instead")
                self.argument_cache_path = f"argument_cache.json"

            if os.path.exists(self.argument_cache_path):
                # print("Loading Argument cache...")
                with open(self.argument_cache_path, "r", encoding="utf8") as f:
                    self.argument_cache = json.load(f)
                # print(f"Loaded {len(self.argument_cache)} items from cache")

        self.instance_of = self.client.get("P31")
        self.occupation = self.client.get("P106")
        self.subclass_of = self.client.get("P279")
        self.properties = [self.instance_of, self.occupation, self.subclass_of]

    def get_names(self, argument):
        key = argument.id
        if key not in self.argument_cache:
            item = None
            i = 0
            while item is None and i < 10:
                try:
                    item = self.client.get(key, load=True)
                except HTTPError:
                    print(f"HTTP Error, retrying {key} (get names)...")
                    time.sleep(1)
                    i += 1

            if item is None:
                print(f"Could not retrieve {key}, 404 error")
                self.argument_cache[key] = {}
            try:
                wikipedia_titles = {}
                for (lang, d) in item.attributes["labels"].items():
                    if lang in languages2save:
                        wikipedia_titles[str(lang)] = str(d["value"])
                self.argument_cache[key] = wikipedia_titles
            except (KeyError, AttributeError):
                print(f"Could not retrieve wikipedia tittles for argument {key}")
                self.argument_cache[key] = {}

        return self.argument_cache[key]

    def get_wikidata_info(self, keys: List[str], language: str):
        if language not in languages2save:
            raise ValueError(
                f"Language {language} not in languages2save, please add it"
            )
        # print("Retrieving Wikidata info...")
        for key in keys:
            # print(key)
            if key not in self.cache:
                item = None
                i = 0
                while item is None and i < 10:
                    try:
                        item = self.client.get(key, load=True)
                    except HTTPError:
                        print(f"HTTP Error, retrying {key} (get_wikidata_info)...")
                        time.sleep(1)
                        i += 1

                if item is None:
                    print(f"Could not retrieve {key}, 404 error")
                    self.cache[key] = {
                        "wikidata_descriptions": {},
                        "arguments": [],
                        "wikipedia_titles": {},
                    }
                    continue

                try:
                    description = str(item.description)
                    if (
                        description == "Wikimedia disambiguation page"
                        or description == "Wikimedia list article"
                    ):
                        self.cache[key] = {
                            "wikidata_descriptions": {},
                            "arguments": [],
                            "wikipedia_titles": {},
                        }
                        raise KeyError

                    arguments_keys = []

                    for prop in self.properties:
                        args = item.getlist(prop)
                        for a in args:
                            arguments_keys.append(a)

                    arguments = [self.get_names(a) for a in arguments_keys]

                    wikidata_descriptions = {}
                    for (lang, d) in item.attributes["descriptions"].items():
                        if lang in languages2save:
                            wikidata_descriptions[str(lang)] = str(d["value"])

                    wikipedia_titles = {}
                    try:
                        for (lang, d) in item.attributes["labels"].items():
                            if lang in languages2save:
                                wikipedia_titles[str(lang)] = str(d["value"])
                    except (KeyError, AttributeError):
                        print(f"Could not wikipedia tittles for key {key}")
                        pass

                    self.cache[key] = {
                        "wikidata_descriptions": wikidata_descriptions,
                        "arguments": arguments,
                        "wikipedia_titles": wikipedia_titles,
                    }
                except (KeyError, AssertionError):
                    self.cache[key] = {
                        "wikidata_descriptions": {},
                        "arguments": [],
                        "wikipedia_titles": {},
                    }

            if (
                key in self.cache
                and self.cache[key]["wikidata_descriptions"] is not None
                and language in self.cache[key]["wikipedia_titles"]
                and language in self.cache[key]["wikidata_descriptions"]
            ):
                description = self.cache[key]["wikidata_descriptions"][language]
                arguments = []
                for a in self.cache[key]["arguments"]:
                    if language in a:
                        arguments.append(a[language])
                wikipedia_title = self.cache[key]["wikipedia_titles"][language]
                return description, arguments, wikipedia_title

        # Fall back to English
        if language != "en":
            print(f"Could not retrieve {keys} in {language}, falling back to English")
            # print(keys[0] in self.cache)
            # print(self.cache[keys[0]]["wikidata_descriptions"] is not None)
            # print(language in self.cache[keys[0]]["wikipedia_titles"])
            # print(language in self.cache[keys[0]]["wikidata_descriptions"])
            # print(f"self.cache[{keys[0]}] = {self.cache[keys[0]]}")

            return self.get_wikidata_info(keys, "en")

        return "No wikidata summary found", [], "No wikipedia title"

    def save_cache(self):
        if not self.ignore_cache:
            with open(self.wikidata_cache_path, "w", encoding="utf8") as f:
                json.dump(self.cache, f, indent=4, ensure_ascii=False)
            with open(self.argument_cache_path, "w", encoding="utf8") as f:
                json.dump(self.argument_cache, f, indent=4, ensure_ascii=False)
        else:
            print(f"Cache not saved because ignore_cache is {self.ignore_cache}")


class MediaWikiRetriever:
    def __init__(self, language: str, ignore_cache: bool = False):
        self.wikipedia = MediaWiki()
        self.wikipedia.set_api_url(lang=language)
        self.cache = {"No wikipedia title": "No wikipedia summary found"}
        self.language = language
        self.ignore_cache = ignore_cache
        if not self.ignore_cache:
            self.media_wiki_cache_path = (
                f"/scratch/igarcia945/GENRE/mediawiki_cache_{language}.json"
            )
            try:
                os.makedirs(os.path.dirname(self.media_wiki_cache_path), exist_ok=True)
            except PermissionError:
                print(f"Could not create directory {self.media_wiki_cache_path}")
                print("Using current directory instead")
                self.media_wiki_cache_path = f"mediawiki_cache_{language}.json"

            if os.path.exists(self.media_wiki_cache_path):
                # print("Loading Mediawiki cache...")
                with open(self.media_wiki_cache_path, "r", encoding="utf8") as f:
                    self.cache = json.load(f)
            # print(f"Loaded {len(self.cache)} entries from cache")

    def get_wikipedia_summary(self, key: str):
        # print("Retrieving Wikipedia summary...")
        # print(key)
        if key not in self.cache:
            try:
                self.cache[key] = str(self.wikipedia.summary(key, auto_suggest=False))
            except (
                PageError,
                DisambiguationError,
                KeyError,
                mediawiki.exceptions.PageError,
                mediawiki.exceptions.MediaWikiException,
            ) as e:
                search = self.wikipedia.search(key, suggestion=False)
                if len(search) > 0:
                    for s in search:
                        try:
                            self.cache[key] = str(
                                self.wikipedia.summary(s, auto_suggest=False)
                            )
                            break
                        except (
                            PageError,
                            DisambiguationError,
                            KeyError,
                            mediawiki.exceptions.PageError,
                            mediawiki.exceptions.MediaWikiException,
                        ) as e:
                            continue
            except (HTTPTimeoutError, ReadTimeout):
                print(f"Timeout error. Retrying {key} (wikipedia summary)...")
                time.sleep(5)
                self.get_wikipedia_summary(key)

            if key not in self.cache:
                self.cache[key] = "No wikipedia summary found"

        return self.cache[key]

    def save_cache(self):
        if not self.ignore_cache:
            with open(self.media_wiki_cache_path, "w", encoding="utf8") as f:
                json.dump(self.cache, f, indent=4, ensure_ascii=False)
        else:
            print(f"Cache not saved because ignore_cache is {self.ignore_cache}")


def get_wikipedia_info_batch(language, ignore_cache, sentence_dict):
    # print(f"language: {language}")
    # print(f"sentence_dict PREV: {sentence_dict}")
    wikidata_retriever = WikiDataRetriever(ignore_cache=ignore_cache)
    mediawiki_retriever = MediaWikiRetriever(
        language=language, ignore_cache=ignore_cache
    )
    # wikidata_retriever.cache = {}  #!!!!!!!!!!
    # mediawiki_retriever.cache = {}  #!!!!!!!!!!

    for sentence_id, sentence in sentence_dict.items():
        for entity_id, entity in sentence["entities"].items():
            if entity["genre_prediction"] is not None:
                retrieved = False
                i = 10
                while not retrieved:
                    try:
                        if (
                            entity["wikidata_summary"] is None
                            or entity["wikidata_summary"] == "Wikimedia list article"
                        ):
                            (
                                description,
                                arguments,
                                wikipedia_title,
                            ) = wikidata_retriever.get_wikidata_info(
                                entity["genre_prediction"], language
                            )
                            entity["wikidata_summary"] = description
                            entity["wikidata_arguments"] = arguments
                            entity["wikipedia_title"] = wikipedia_title

                        if (
                            entity["wikipedia_summary"] is None
                            and entity["wikipedia_title"] is not None
                        ):
                            entity[
                                "wikipedia_summary"
                            ] = mediawiki_retriever.get_wikipedia_summary(
                                entity["wikipedia_title"]
                            )

                        retrieved = True
                    except Exception as e:
                        print(f"Exception: {e}")
                        print(f"Retrying in 5 seconds {entity['genre_prediction']}...")
                        time.sleep(5)
                        i -= 1
                        if i == 0:
                            print("Giving up")
                            raise e

    return sentence_dict, wikidata_retriever.cache, mediawiki_retriever.cache


def get_wikipedia_info_parallel(
    json_dict_path: str,
    language: str,
    batch_size: int,
    num_workers: int,
    ignore_cache: bool,
):

    with open(json_dict_path, "r", encoding="utf8") as f:
        sentence_dict: Dict[
            int,
            Dict[str, Union[List[str], Dict[int, Dict[str, Union[str, List[str]]]]]],
        ] = json.load(f)

    if language == "multi":
        print(
            "Multi-language dataset, we will retrieve the English wikidata/wikipedia data"
        )
        language = "en"

    wikidata_retriever = WikiDataRetriever(ignore_cache=ignore_cache)
    mediawiki_retriever = MediaWikiRetriever(
        language=language, ignore_cache=ignore_cache
    )
    if ignore_cache:
        print("Ignoring cache")

    print(f"Wikidata cache size: {len(wikidata_retriever.cache)}")
    print(f"Mediawiki cache size: {len(mediawiki_retriever.cache)}")

    sentence_ids = list(sentence_dict.keys())
    batches = [
        sentence_ids[i : i + batch_size]
        for i in range(0, len(sentence_ids), batch_size)
    ]

    chunks = []
    for i in range(0, len(batches), num_workers):
        chunks.append(batches[i : i + num_workers])

    with Pool(num_workers) as p:
        for chunk in tqdm(chunks, desc="Getting wikipedia summaries"):
            # print(f"Wikidata cache size 1: {len(wikidata_retriever.cache)}")
            i = 0
            results = []
            while len(results) == 0 and i < 10:
                try:
                    results = p.starmap(
                        get_wikipedia_info_batch,
                        [
                            (
                                language,
                                ignore_cache,
                                {k: sentence_dict[k] for k in batch},
                            )
                            for batch in chunk
                        ],
                    )
                except Exception as e:
                    i += 1
                    print(f"Exception: {e}, retrying multicore...")
                    continue

            if len(results) == 0:
                print(f"Running this chunk in single process mode")
                for batch_no, batch in enumerate(chunk):
                    r = get_wikipedia_info_batch(
                        language, ignore_cache, {k: sentence_dict[k] for k in batch}
                    )
                    # print(f"\nbatch_no: {batch_no}")
                    # print(f"r: {r[0]}")

                    results.append(r)

            for result in results:
                sentence_dict.update(result[0])
                if not ignore_cache:
                    wikidata_retriever.cache.update(result[1])
                    mediawiki_retriever.cache.update(result[2])

            # print(f"Wikidata cache size 2: {len(wikidata_retriever.cache)}")
            # Save cache checkpoints
            if not ignore_cache:
                wikidata_retriever.save_cache()
                mediawiki_retriever.save_cache()

    wikidata_retriever.save_cache()
    mediawiki_retriever.save_cache()

    with open(json_dict_path, "w", encoding="utf8") as f:
        json.dump(sentence_dict, f, indent=4, ensure_ascii=False)

    print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--json_dict_path",
        type=str,
        required=True,
        help="Path to the tsv file containing the dataset",
    )

    parser.add_argument(
        "--language",
        type=str,
        required=False,
        default=None,
        help="Language of the dataset, if not specified, we will use the predicted language from genre",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        required=False,
        default=1,
        help="Batch size for parallel processing",
    )

    parser.add_argument(
        "--num_workers",
        type=int,
        required=False,
        default=1,
        help="Number of workers for parallel processing",
    )

    parser.add_argument(
        "--ignore_cache",
        action="store_true",
        help="Ignore local wikipedia/wikidata cache. "
        "Using a local cache will reduce the load on the wikipedia/mediawiki servers, "
        "but it can be very slow if the cache gets too big. If you use a local cache "
        "make sure to use a large batch size or the script will be very slow. Better "
        "implementations of the cache may solve this issue, but currently I dont have time to implement it. "
        "Contributions are welcome!",
    )

    args = parser.parse_args()

    print(f"Running in parallel mode with {args.num_workers} workers")
    get_wikipedia_info_parallel(
        json_dict_path=args.json_dict_path,
        language=args.language,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        ignore_cache=args.ignore_cache,
    )
