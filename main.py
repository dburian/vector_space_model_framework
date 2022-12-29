import argparse
import importlib
import logging
import os
import sys

import pyterrier as pt

from src import run, types, utils

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
    "--index_path", type=str, help="Directory where to store/load an index."
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
    "run-0_en": importlib.import_module("src.experiments.run_0").Run0EN,
    "run-0_cs": importlib.import_module("src.experiments.run_0").Run0CS,
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

    doc_dir = args.documents[: args.documents.rfind(".")]
    logging.info("Loading documents from %s.", doc_dir)
    document_paths = list(utils.paths_gen(args.documents, doc_dir))
    documents = pt.index.treccollection2textgen(document_paths)

    lan = types.LAN.EN if args.run.endswith("_en") else types.LAN.CS
    experiment = AVAILABLE_RUNS[args.run](threads=args.threads, lan=lan)

    logging.info("Loading topics from %s.", args.queries)
    topics = utils.read_topics(args.queries, experiment.get_query_parser())

    index_path = os.path.join("./indices", args.run)
    run.run_experiment(args.run, experiment, documents, topics, index_path, args.output)


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
