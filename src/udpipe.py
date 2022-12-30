import email.mime.multipart
import email.mime.nonmultipart
import email.policy
import json
import logging
import sys
import urllib.error
import urllib.request
from typing import Iterable

import conllu

from src.types import LAN, Document, Topics

UD_PIPE_SERVICE = "https://lindat.mff.cuni.cz/services/udpipe/api"


def _perform_request(*, params, server=UD_PIPE_SERVICE, method="process"):
    if not params:
        request_headers, request_data = {}, None
    else:
        message = email.mime.multipart.MIMEMultipart(
            "form-data", policy=email.policy.HTTP
        )

        for name, value in params.items():
            payload = email.mime.nonmultipart.MIMENonMultipart("text", "plain")
            payload.add_header("Content-Disposition", f'form-data; name="{name}"')
            payload.add_header("Content-Transfer-Encoding", "8bit")
            payload.set_payload(value, charset="utf-8")
            message.attach(payload)

        request_data = message.as_bytes().split(b"\r\n\r\n", maxsplit=1)[1]
        request_headers = {"Content-Type": message["Content-Type"]}

    try:
        with urllib.request.urlopen(
            urllib.request.Request(
                url=f"{server}/{method}",
                headers=request_headers,
                data=request_data,
            )
        ) as request:
            return json.loads(request.read())
    except urllib.error.HTTPError as err:
        print(
            "An exception was raised during UDPipe 'process' REST request.\n"
            "The service returned the following error:\n"
            f"  {err.fp.read().decode('utf-8')}",
            file=sys.stderr,
        )
        raise
    except json.JSONDecodeError as err:
        print(
            "Cannot parse the JSON response of UDPipe 'process' REST request.\n"
            f"  {err.msg}",
            file=sys.stderr,
        )
        raise


def _lemmatize_list(texts: list[str], lan: LAN) -> list[str]:
    text_data = ""
    for text in texts:
        text_data += text
        text_data += "\n\n"

    params = {
        "input": "horizontal",
        "output": "conllu",
        "model": "czech" if lan == LAN.CS else "english",
        "tokenizer": "",
        "tagger": "",
        "data": text_data.rstrip(),
    }
    response = _perform_request(params=params)
    parsed_texts = []
    for parsed_text in conllu.parse(response["result"]):
        lemmas = [token["lemma"] for token in parsed_text]
        parsed_texts.append(" ".join(lemmas))

    return parsed_texts


def lemmatize_docs(docs: list[Document], lan: LAN) -> list[Document]:
    texts = [doc["text"] for doc in docs]

    logging.info("Lemmatizing %s docs with UDPipe...", len(docs))
    parsed_docs = _lemmatize_list(texts, lan)
    logging.info("Lemmatizing %s docs with UDPipe...Done", len(docs))
    for doc, parsed_doc in zip(docs, parsed_docs):
        doc["text"] = parsed_doc

    return docs


def lemmatize_topics(topics: Topics, lan: LAN) -> Topics:
    queries = list(topics["query"])
    logging.info("Lemmatizing %s queries with UDPipe...", len(topics))
    parsed_topics = _lemmatize_list(queries, lan)
    logging.info("Lemmatizing %s queries with UDPipe...Done", len(topics))

    topics["query"] = parsed_topics
    return topics


def batchify(iterable: Iterable, batch_size: int) -> Iterable[list]:
    batch = []
    for elem in iterable:
        batch.append(elem)
        if len(batch) == batch_size:
            yield batch
            batch = []

    if len(batch) > 0:
        yield batch


def unbatchify(iterable: Iterable[list]) -> Iterable:
    for list_ in iterable:
        for elem in list_:
            yield elem
