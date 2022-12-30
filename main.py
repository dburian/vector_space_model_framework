import argparse
import logging
import os
import sys

import jnius_config
import pyterrier as pt

from src import run, types, utils
from src.experiments import run_0

parser = argparse.ArgumentParser()

parser.add_argument(
    "-q", "--queries", type=str, help="XML topic file in TREC format.", required=True
)
parser.add_argument(
    "-d",
    "--documents",
    type=str,
    help=(
        "File containing list of documents, where each path is prepended with given"
        " filename without extension as a containing directory."
    ),
    required=True,
)
parser.add_argument(
    "-r",
    "--run",
    type=str,
    default=None,
    help="Identifier of the run/experiment.",
    required=True,
)
parser.add_argument(
    "-o",
    "--output",
    type=str,
    help="Name of the output results file.",
    required=True,
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
    "run-0-tfidf-pivoted_python_en": run_0.Run0TfIdfPivotedPython,
    "run-0-tfidf-pivoted_python_cs": run_0.Run0TfIdfPivotedPython,
    "run-0-tfidf-pivoted_en": run_0.Run0TfIdfPivoted,
    "run-0-tfidf-pivoted_cs": run_0.Run0TfIdfPivoted,
}


def main(args: argparse.Namespace) -> None:
    logging.basicConfig(
        format="%(asctime)s : [%(levelname)s] %(message)s", level=logging.INFO
    )

    if args.run not in AVAILABLE_RUNS:
        logging.error(
            "Run %s is not available. Choose one of %s.",
            args.run,
            AVAILABLE_RUNS.keys(),
        )
        sys.exit(1)

    cwd = os.path.abspath(".")

    # Adding my java code before jnius starts
    jnius_config.add_classpath(
        os.path.join(cwd, "java", "ir", "target", "ir-1.0-SNAPSHOT.jar")
    )

    if not pt.started():
        pt.init(tqdm="tqdm")

    pt.set_property("terrier.home", os.path.join(cwd, "pyterrier_home"))

    doc_dir = args.documents[: args.documents.rfind(".")]
    logging.info("Loading documents from %s.", doc_dir)
    document_paths = list(utils.paths_gen(args.documents, doc_dir))
    documents = pt.index.treccollection2textgen(document_paths)

    lan = types.LAN.EN if args.run.endswith("_en") else types.LAN.CS
    experiment = AVAILABLE_RUNS[args.run](threads=args.threads, lan=lan)

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
