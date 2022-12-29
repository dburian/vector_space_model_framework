import os
from enum import Enum
from typing import Any, Callable, Iterable, Optional

import bs4
import pandas as pd

Documents = Iterable[dict[str, str]]
Topics = pd.DataFrame
IndexRef = Any
Results = pd.DataFrame


class LAN(Enum):
    EN = 1
    CS = 2


class Experiment:
    def __init__(self, *, lan: LAN, threads: Optional[int] = None) -> None:
        self._threads = threads if threads is not None else (os.cpu_count() // 2)
        self._lan = lan

    def get_index(self, index_path: str, documents: Documents) -> IndexRef:
        raise NotImplementedError()

    def get_results(self, index_ref: IndexRef, topics: Topics) -> Results:
        raise NotImplementedError()

    def get_query_parser(self) -> Callable[[bs4.element.Tag], dict[str, str]]:
        def query_parser(tag: bs4.element.Tag) -> dict[str, str]:
            qid = str(tag.num.string)
            title = str(tag.title.string)
            return {"qid": qid, "query": title}

        return query_parser
