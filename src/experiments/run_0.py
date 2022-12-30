import functools
import logging
import multiprocessing
import os

import nltk
import pyterrier as pt
from nltk import downloader

from src import udpipe, utils
from src.types import LAN, Documents, Experiment, IndexRef, Results, Topics


class Run0(Experiment):
    def __init__(self, *args, **kwargs) -> None:
        kwargs.setdefault("index_dir", f"run-0_{kwargs['lan'].value}")
        super().__init__(*args, **kwargs)
        self._weighting_model = "Tf"

    def get_index(self, documents: Documents) -> IndexRef:
        indexer = pt.IterDictIndexer(
            self._index_path,
            threads=self._threads,
            overwrite=True,
            stemmer="none",
            stopwords="none",
            tokeniser="whitespace",
        )

        doc_tokenizer = functools.partial(
            utils.doc_text_transform, transform=utils.basic_tokenizer
        )
        with multiprocessing.Pool(self._threads) as pool:
            return indexer.index(pool.imap(doc_tokenizer, documents), fields=["text"])

    def get_results(self, index_ref: IndexRef, topics: Topics) -> Results:
        retrieve = pt.BatchRetrieve(index_ref, wmodel=self._weighting_model) % 1000
        retrieve = pt.apply.query(utils.sanitize_query) >> retrieve
        retrieve = (
            pt.apply.query(lambda row: utils.basic_tokenizer(row.query)) >> retrieve
        )

        return retrieve.transform(topics)


class Run0TfIdf(Run0):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._weighting_model = "TF_IDF"


# -------------------------------------------------------------------------------------
#                                       TOKENIZATION
# -------------------------------------------------------------------------------------


class Run0PyTerrierTok(Run0TfIdf):
    def __init__(self, *args, **kwargs) -> None:
        kwargs.setdefault("index_dir", f"run-0-pyterrier-tok_{kwargs['lan'].value}")
        super().__init__(*args, **kwargs)

    def get_index(self, documents: Documents) -> IndexRef:
        indexer = pt.IterDictIndexer(
            self._index_path,
            threads=self._threads,
            overwrite=True,
            stemmer="none",
            stopwords="none",
            tokeniser="english" if self._lan == LAN.EN else "utf",
        )

        return indexer.index(documents)

    def get_results(self, index_ref: IndexRef, topics: Topics) -> Results:
        retrieve = pt.BatchRetrieve(index_ref, wmodel=self._weighting_model) % 1000
        retrieve = pt.apply.query(utils.sanitize_query) >> retrieve

        return retrieve.transform(topics)


class Run0NlktTok(Run0TfIdf):
    def __init__(self, *args, **kwargs) -> None:
        kwargs.setdefault("index_dir", f"run-0-nltk-tok_{kwargs['lan'].value}")
        super().__init__(*args, **kwargs)
        nltk_downloader = downloader.Downloader()
        if not nltk_downloader.is_installed("punkt"):
            logging.info("Downloading punkt package from nltk.")
            nltk.download("punkt")
            logging.info("Done")

        self._text_tokenizer = functools.partial(
            utils.nlkt_tokenizer,
            language="english" if self._lan == LAN.EN else "czech",
        )
        self._doc_tokenizer = functools.partial(
            utils.doc_text_transform,
            transform=self._text_tokenizer,
        )

    def get_index(self, documents: Documents) -> IndexRef:
        indexer = pt.IterDictIndexer(
            self._index_path,
            threads=self._threads,
            overwrite=True,
            stemmer="none",
            stopwords="none",
            tokeniser="whitespace",
        )

        # Set higher maximal term length
        pt.set_property("max.term.length", "40")
        with multiprocessing.Pool(self._threads) as pool:
            return indexer.index(
                pool.imap(self._doc_tokenizer, documents), fields=["text"]
            )

    def get_results(self, index_ref: IndexRef, topics: Topics) -> Results:
        retrieve = pt.BatchRetrieve(index_ref, wmodel=self._weighting_model) % 1000
        retrieve = pt.apply.query(utils.sanitize_query) >> retrieve
        retrieve = (
            pt.apply.query(lambda row: self._text_tokenizer(row.query)) >> retrieve
        )

        return retrieve.transform(topics)


# -------------------------------------------------------------------------------------
#                               STOPWORDS BEFORE STEMMING
# -------------------------------------------------------------------------------------


class Run0PyTerrierStopEN(Run0PyTerrierTok):
    def __init__(self, *args, **kwargs) -> None:
        kwargs.setdefault("index_dir", f"run-0-pyterrier-stop_{kwargs['lan'].value}")
        super().__init__(*args, **kwargs)

    def get_index(self, documents: Documents) -> IndexRef:
        indexer = pt.IterDictIndexer(
            self._index_path,
            threads=self._threads,
            overwrite=True,
            stemmer="none",
            stopwords="english" if self._lan == LAN.EN else "none",
            tokeniser="english" if self._lan == LAN.EN else "utf",
        )

        return indexer.index(documents)


class Run0NltkStopEN(Run0PyTerrierStopEN):
    def __init__(self, *args, **kwargs) -> None:
        kwargs.setdefault("index_dir", f"run-0-nltk-stop_{kwargs['lan'].value}")
        super().__init__(*args, **kwargs)

        if self._lan == LAN.EN:
            pt.set_property("stopwords.filename", "nltk_stopwords_en.txt")


class Run0KaggleStopCS(Run0PyTerrierTok):
    def __init__(self, *args, **kwargs) -> None:
        kwargs.setdefault("index_dir", f"run-0-kaggle-stop_{kwargs['lan'].value}")
        super().__init__(*args, **kwargs)

        if self._lan == LAN.CS:
            pt.set_property("stopwords.filename", "kaggle_stopwords_cs.txt")

    def get_index(self, documents: Documents) -> IndexRef:
        indexer = pt.IterDictIndexer(
            self._index_path,
            threads=self._threads,
            overwrite=True,
            stemmer="none",
            stopwords="english" if self._lan == LAN.CS else "none",
            tokeniser="english" if self._lan == LAN.EN else "utf",
        )

        return indexer.index(documents)


# -------------------------------------------------------------------------------------
#                               STEMMING AND LEMMATIZATION
# -------------------------------------------------------------------------------------


class Run0PorterStem(Run0KaggleStopCS):
    def __init__(self, *args, **kwargs) -> None:
        kwargs.setdefault("index_dir", f"run-0-porter-stemm_{kwargs['lan'].value}")
        super().__init__(*args, **kwargs)

    def get_index(self, documents: Documents) -> IndexRef:
        indexer = pt.IterDictIndexer(
            self._index_path,
            threads=self._threads,
            overwrite=True,
            stemmer="porter",
            stopwords="english" if self._lan == LAN.CS else "none",
            tokeniser="english" if self._lan == LAN.EN else "utf",
        )

        return indexer.index(documents)


class Run0SnowballStem(Run0KaggleStopCS):
    def __init__(self, *args, **kwargs) -> None:
        kwargs.setdefault("index_dir", f"run-0-snowball-stemm_{kwargs['lan'].value}")
        super().__init__(*args, **kwargs)

    def get_index(self, documents: Documents) -> IndexRef:
        indexer = pt.IterDictIndexer(
            self._index_path,
            threads=self._threads,
            overwrite=True,
            stemmer="EnglishSnowballStemmer",
            stopwords="english" if self._lan == LAN.CS else "none",
            tokeniser="english" if self._lan == LAN.EN else "utf",
        )

        return indexer.index(documents)


class Run0UDPipeLemm(Run0TfIdf):
    def __init__(self, *args, **kwargs) -> None:
        kwargs.setdefault("index_dir", f"run-0-udpipe-lemm_{kwargs['lan'].value}")
        super().__init__(*args, **kwargs)

    def get_index(self, documents: Documents) -> IndexRef:
        indexer = pt.IterDictIndexer(
            self._index_path,
            threads=self._threads,
            overwrite=True,
            stemmer="none",
            stopwords="none",
            tokeniser="whitespace",
        )

        lemmatize_docs = functools.partial(udpipe.lemmatize_docs, lan=self._lan)

        # with multiprocessing.Pool(self._threads) as pool:
        lemmatized_docs = udpipe.unbatchify(
            map(lemmatize_docs, udpipe.batchify(documents, 2))
        )
        return indexer.index(lemmatized_docs)

    def get_results(self, index_ref: IndexRef, topics: Topics) -> Results:
        retriever = pt.BatchRetrieve(index_ref, wmodel=self._weighting_model) % 1000
        retriever = pt.apply.query(utils.sanitize_query) >> retriever

        lemmatized_topics = udpipe.lemmatize_topics(topics, self._lan)

        return retriever.transform(lemmatized_topics)


# -------------------------------------------------------------------------------------
#                               WEIGHTING SCHEMES
# -------------------------------------------------------------------------------------


# TODO: Rewrite these with single base class and self._wmodel
class Run0TfIdfPivotedPython(Run0SnowballStem):
    def get_results(self, index_ref: IndexRef, topics: Topics) -> Results:
        pivoted_tfidf = pt.autoclass(
            "org.terrier.matching.models.WeightingModelLibrary"
        )().tf_pivoted

        def my_wmodel(key_freq, posting, entry_stats, collection_stats):
            return pivoted_tfidf(
                posting.getFrequency(),
                0.6,
                posting.getDocumentLength(),
                collection_stats.getAverageDocumentLength(),
            )

        retriever = pt.BatchRetrieve(index_ref, wmodel=my_wmodel) % 1000
        retriever = pt.apply.query(utils.sanitize_query) >> retriever

        return retriever.transform(topics)


class Run0TfIdfPivoted(Run0SnowballStem):
    def get_results(self, index_ref: IndexRef, topics: Topics) -> Results:
        pivoted_tfidf = pt.autoclass("cz.dburian.TfIdfPivoted")(0.6)

        retriever = pt.BatchRetrieve(index_ref, wmodel=pivoted_tfidf) % 1000
        retriever = pt.apply.query(utils.sanitize_query) >> retriever

        return retriever.transform(topics)


class Run0TfIdfPivoted(Run0SnowballStem):
    def get_results(self, index_ref: IndexRef, topics: Topics) -> Results:
        pivoted_tfidf = pt.autoclass("cz.dburian.TfIdfRobertsonPivoted")(0.6)

        retriever = pt.BatchRetrieve(index_ref, wmodel=pivoted_tfidf) % 1000
        retriever = pt.apply.query(utils.sanitize_query) >> retriever

        return retriever.transform(topics)
