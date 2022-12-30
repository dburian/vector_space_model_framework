import os
import re
from typing import Callable, Iterator

import bs4
import nltk
import pandas as pd

SEPS = r"-,.:;?! \n\t\[\]\(\)'\""
QUERY_MARKS = r"-#\{\}\(\)\^\+:~/\'\""


def paths_gen(docs_paths_file: str, docs_paths_prefix: str) -> Iterator[str]:
    with open(docs_paths_file, mode="r", encoding="utf-8") as paths_file:
        for line in paths_file:
            yield os.path.join(docs_paths_prefix, line.rstrip())


def take(iter_: Iterator, max_len=100) -> Iterator:
    for i, elem in enumerate(iter_):
        yield elem
        if i + 1 > max_len:
            break


def doc_text_transform(
    doc: dict[str, str], transform: Callable[[str], str]
) -> dict[str, str]:
    doc["text"] = transform(doc["text"])
    return doc


def basic_tokenizer(text: str) -> str:
    """Splits text by ',.:;?! \n\t[]()'"' and joins it by space."""
    words = re.split("[" + SEPS + "]", text)
    return " ".join(words)


def nlkt_tokenizer(text: str, language: str = "english") -> str:
    tokens = nltk.word_tokenize(text, language=language)
    return " ".join(tokens)


def read_topics(
    topic_file: str, get_query: Callable[[bs4.element.Tag], dict[str, str]]
) -> pd.DataFrame:
    topics = []
    with open(topic_file, mode="r", encoding="utf-8") as file_handle:
        soup = bs4.BeautifulSoup(file_handle, "xml")
        for query_tag in soup.find_all("top"):
            topics.append(get_query(query_tag))

    return pd.DataFrame(topics)


def get_query_title(tag: bs4.element.Tag) -> dict[str, str]:
    qid = str(tag.num.string)
    title = str(tag.title.string)
    return {"qid": qid, "query": title}


def sanitize_query(row: pd.Series) -> str:
    query_tokens = re.split("[" + QUERY_MARKS + "]", row.query)
    return " ".join(query_tokens)
