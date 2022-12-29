import functools
import multiprocessing
from typing import Optional

import pyterrier as pt

from src import utils
from src.types import Documents, Experiment, IndexRef, Results, Topics


class Run0EN(Experiment):
    def get_index(self, index_path: str, documents: Documents) -> IndexRef:
        indexer = pt.IterDictIndexer(
            index_path,
            threads=self._threads,
            overwrite=True,
            stemmer="none",
            stopwords="none",
            tokeniser="whitespace",
        )

        tokeniser_partial = functools.partial(
            utils.tokenise, tokeniser=utils.basic_tokeniser
        )
        with multiprocessing.Pool(self._threads) as pool:
            return indexer.index(
                pool.imap(tokeniser_partial, documents), fields=["text"]
            )

    def get_results(self, index_ref: IndexRef, topics: Topics) -> Results:
        retrieve = pt.BatchRetrieve(index_ref, {"wmmodel": "TF_IDF"}) % 1000
        retrieve = pt.apply.query(utils.sanitize_query) >> retrieve
        retrieve = (
            pt.apply.query(lambda row: utils.basic_tokeniser(row.query)) >> retrieve
        )

        return retrieve.transform(topics)


class Run0CS(Experiment):
    def get_index(self, index_path: str, documents: Documents) -> IndexRef:
        indexer = pt.IterDictIndexer(
            index_path,
            threads=self._threads,
            overwrite=True,
            stemmer="none",
            stopwords="none",
            tokeniser="whitespace",
        )

        tokeniser_partial = functools.partial(
            utils.tokenise, tokeniser=utils.basic_tokeniser
        )
        with multiprocessing.Pool(self._threads) as pool:
            return indexer.index(
                pool.imap(tokeniser_partial, documents), fields=["text"]
            )

    def get_results(self, index_ref: IndexRef, topics: Topics) -> Results:
        retrieve = pt.BatchRetrieve(index_ref, {"wmmodel": "TF_IDF"}) % 1000
        retrieve = pt.apply.query(utils.sanitize_query) >> retrieve
        retrieve = (
            pt.apply.query(lambda row: utils.basic_tokeniser(row.query)) >> retrieve
        )

        return retrieve.transform(topics)
