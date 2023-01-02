import argparse
import logging
import os
import sys

import jnius_config
import pyterrier as pt

from src import run, types, utils
from src.experiments import run_0, run_1, run_2

parser = argparse.ArgumentParser()

parser.add_argument(
    "-q",
    "--queries",
    type=str,
    help="XML topic file in TREC format.",
    default=None,
)
parser.add_argument(
    "-d",
    "--documents",
    type=str,
    help=(
        "File containing list of documents, where each path is prepended with given"
        " filename without extension as a containing directory."
    ),
    default=None,
)
parser.add_argument(
    "-r",
    "--run",
    type=str,
    default=None,
    help="Identifier of the run/experiment.",
)
parser.add_argument(
    "-o",
    "--output",
    type=str,
    help="Name of the output results file.",
    default=None,
)
parser.add_argument(
    "--threads",
    type=int,
    default=None,
    help=(
        "Number of threads to use. Default is to use half of the available processors."
    ),
)
parser.add_argument(
    "--slopes",
    type=float,
    default=[],
    nargs="+",
    action="extend",
    help="Slopes for which to run experiments fot Pivoted unique normalization.",
)
parser.add_argument(
    "--qrels",
    type=str,
    default=None,
    help="Qrels for given topics and document set. Required with '--slopes'.",
)
parser.add_argument(
    "--pivoted_wmodel",
    type=str,
    default=None,
    help="Java class to perform pivoted weighting for slopes experimentation.",
)
parser.add_argument(
    "--udpipe_service_url",
    type=str,
    default="http://127.0.0.1:6666",
    help="Url to UDPipe2 service.",
)
parser.add_argument(
    "--list_runs",
    # type=bool,
    default=False,
    action="store_true",
    help="List available runs.",
)

AVAILABLE_RUNS = {
    "run-0_en": run_0.Run0,
    "run-0_cs": run_0.Run0,
    "run-0-tfidf_en": run_0.Run0TfIdf,
    "run-0-tfidf_cs": run_0.Run0TfIdf,
    "run-0-pyterrier-tok_en": run_0.Run0PyTerrierTok,
    "run-0-pyterrier-tok_cs": run_0.Run0PyTerrierTok,
    "run-0-nltk-tok_en": run_0.Run0NlktTok,
    "run-0-nltk-tok_cs": run_0.Run0NlktTok,
    "run-0-pyterrier-stop_en": run_0.Run0PyTerrierStopEN,
    "run-0-nltk-stop_en": run_0.Run0NltkStopEN,
    "run-0-kaggle-stop_cs": run_0.Run0KaggleStopCS,
    "run-0-porter-stemm_en": run_0.Run0PorterStem,
    "run-0-porter-stemm_cs": run_0.Run0PorterStem,
    "run-0-snowball-stemm_en": run_0.Run0SnowballStem,
    "run-0-snowball-stemm_cs": run_0.Run0SnowballStem,
    "run-0-udpipe-lemm_en": run_0.Run0UDPipeLemm,
    "run-0-udpipe-lemm_cs": run_0.Run0UDPipeLemm,
    "run-0-czech-stemm_cs": run_0.Run0CzechStemmer,
    "run-0-czech-stemm_en": run_0.Run0CzechStemmer,
    "run-0-tfidf-pivoted_python_en": run_0.Run0TfIdfPivotedPython,
    "run-0-tfidf-pivoted_python_cs": run_0.Run0TfIdfPivotedPython,
    "run-0-tfidf-pivoted_en": run_0.Run0TfIdfPivoted,
    "run-0-tfidf-pivoted_cs": run_0.Run0TfIdfPivoted,
    "run-0-tfidf-pivoted-robertson_en": run_0.Run0TfIdfRobertsonPivoted,
    "run-0-tfidf-pivoted-robertson_cs": run_0.Run0TfIdfRobertsonPivoted,
    "run-0-bm25_cs": run_0.Run0BM25,
    "run-0-bm25_en": run_0.Run0BM25,
    "run-0-pl2_cs": run_0.Run0PL2,
    "run-0-pl2_en": run_0.Run0PL2,
    "run-0-lemur-tfidf_cs": run_0.Run0LemurTfIdf,
    "run-0-lemur-tfidf_en": run_0.Run0LemurTfIdf,
    "run-1_cs": run_1.Run1,
    "run-1_en": run_1.Run1,
    "run-2_cs": run_2.Run2,
    "run-2_en": run_2.Run2,
}


def main(args: argparse.Namespace) -> None:
    logging.basicConfig(
        format="%(asctime)s : [%(levelname)s] %(message)s", level=logging.INFO
    )

    if args.list_runs:
        print("\n".join(AVAILABLE_RUNS.keys()))
        sys.exit(0)

    if args.run is None or args.run not in AVAILABLE_RUNS:
        logging.error("Specify one of available runs.")
        sys.exit(1)

    if args.queries is None:
        logging.error("Specify queries to search for.")
        sys.exit(1)

    if args.documents is None:
        logging.error("Specify documents to index.")
        sys.exit(1)

    if args.output is None:
        logging.error("Specify output file.")
        sys.exit(1)

    cwd = os.path.abspath(".")

    # Adding my java code before jnius starts
    jnius_config.add_classpath(
        os.path.join(cwd, "java", "terrier_ir", "target", "terrier_ir-1.0.jar")
    )

    if not pt.started():
        pt.init(tqdm="tqdm")

    pt.set_property("terrier.home", os.path.join(cwd, "pyterrier_home"))

    doc_dir = args.documents[: args.documents.rfind(".")]
    logging.info("Loading documents from %s.", doc_dir)
    document_paths = list(utils.paths_gen(args.documents, doc_dir))
    documents = pt.index.treccollection2textgen(
        document_paths, tag_text_length=4096 * 2
    )

    lan = types.LAN.EN if args.run.endswith("_en") else types.LAN.CS
    experiment = AVAILABLE_RUNS[args.run](
        udpipe_service=args.udpipe_service_url,
        threads=args.threads,
        lan=lan,
    )

    logging.info("Loading topics from %s.", args.queries)
    topics = utils.read_topics(args.queries, experiment.get_query_parser())

    if len(args.slopes) > 0:
        if args.qrels is None:
            logging.error(
                "Qrels must be specified for Pivoted unique normalization slope"
                " experiments."
            )
            sys.exit(1)

        if args.pivoted_wmodel is None:
            logging.error(
                "Pivoted weight model must be specified for Pivoted unique"
                " normalization slope experiments."
            )
            sys.exit(1)

        qrels = pt.io.read_qrels(args.qrels)
        run.run_tfidf_pivoted_slope_experiment(
            args.slopes,
            experiment,
            documents,
            topics,
            qrels,
            args.output,
            args.pivoted_wmodel,
        )
        sys.exit(0)

    run.run_experiment(args.run, experiment, documents, topics, args.output)


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
