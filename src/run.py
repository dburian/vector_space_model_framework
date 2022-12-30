import logging
import os
import time

import pyterrier as pt

from src.types import Documents, Experiment, Topics


def run_experiment(
    run_id: str,
    experiment: Experiment,
    documents: Documents,
    topics: Topics,
    results_path: str,
) -> None:
    logging.info("Running experiment %s.", run_id)
    index_ref = None
    if os.path.exists(experiment.index_path):
        index_creation_time = os.path.getctime(experiment.index_path)
        logging.info(
            "Index read loaded from disk at %s from %s.",
            experiment.index_path,
            time.strftime("%Y %d.%M. %H:%M:%S", time.gmtime(index_creation_time)),
        )
        index_ref = pt.IndexRef.of(experiment.index_path)
    else:
        logging.info("Indexing...")
        index_ref = experiment.get_index(documents)
        logging.info("Done")

    logging.info("Getting results...")
    res = experiment.get_results(index_ref, topics)
    logging.info("Done")

    logging.info("Writing results to %s.", results_path)
    pt.io.write_results(res, results_path, run_name=run_id)
