import argparse
import importlib
import logging
import os
import sys

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

AVAILABLE_RUNS: dict[str, types.Experiment] = {
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

    if not pt.started():
        pt.init(tqdm="tqdm")

    cwd = os.path.abspath(".")
    pt.set_property("terrier.home", os.path.join(cwd, "pyterrier_home"))

    doc_dir = args.documents[: args.documents.rfind(".")]
    logging.info("Loading documents from %s.", doc_dir)
    document_paths = list(utils.paths_gen(args.documents, doc_dir))
    documents = pt.index.treccollection2textgen(document_paths)

    lan = types.LAN.EN if args.run.endswith("_en") else types.LAN.CS
    experiment = AVAILABLE_RUNS[args.run](threads=args.threads, lan=lan)

    logging.info("Loading topics from %s.", args.queries)
    topics = utils.read_topics(args.queries, experiment.get_query_parser())

    run.run_experiment(args.run, experiment, documents, topics, args.output)


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
