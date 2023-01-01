import email.mime.multipart
import email.mime.nonmultipart
import email.policy
import json
import logging
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
        logging.error(
            "An exception was raised during UDPipe 'process' REST request.\n"
            "The service returned the following error:\n"
            "  %s",
            err.fp.read().decode("utf-8"),
        )
        raise
    except json.JSONDecodeError as err:
        logging.error(
            "Cannot parse the JSON response of UDPipe 'process' REST request.\n  %s",
            err.msg,
        )
        raise


def _lemmatize_list(texts: list[str], lan: LAN, udpipe_service) -> list[str]:
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
    logging.info(
        "Lemmatizing %s texts with UDPipe at %s...", len(texts), udpipe_service
    )
    response = _perform_request(params=params, server=udpipe_service)
    logging.info(
        "Lemmatizing %s texts with UDPipe at %s...Done", len(texts), udpipe_service
    )
    parsed_texts = []
    for parsed_text in conllu.parse(response["result"]):
        lemmas = [token["lemma"] for token in parsed_text]
        parsed_texts.append(" ".join(lemmas))

    return parsed_texts


def lemmatize_docs(
    docs: list[Document], lan: LAN, udpipe_service: str
) -> list[Document]:
    texts = [doc["text"] for doc in docs]

    parsed_docs = _lemmatize_list(texts, lan, udpipe_service)
    for doc, parsed_doc in zip(docs, parsed_docs):
        doc["text"] = parsed_doc

    return docs


def lemmatize_topics(topics: Topics, lan: LAN, udpipe_service) -> Topics:
    queries = list(topics["query"])
    parsed_topics = _lemmatize_list(queries, lan, udpipe_service)

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
