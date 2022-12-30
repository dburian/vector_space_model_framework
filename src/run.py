import logging
import os
import time

import pyterrier as pt

from src import utils
from src.types import Documents, Experiment, IndexRef, QRels, Topics


def run_experiment(
    run_id: str,
    experiment: Experiment,
    documents: Documents,
    topics: Topics,
    results_path: str,
) -> None:
    logging.info("Running experiment %s.", run_id)
    index_ref = _get_index_ref(experiment, documents)

    logging.info("Getting results...")
    res = experiment.get_results(index_ref, topics)
    logging.info("Getting results...Done")

    logging.info("Writing results to %s.", results_path)
    pt.io.write_results(res, results_path, run_name=run_id)


def run_tfidf_pivoted_slope_experiment(
    slopes: list[float],
    experiment: Experiment,
    documents: Documents,
    topics: Topics,
    qrels: QRels,
    results_path: str,
    weight_model_java_class: str,
) -> None:
    logging.info("Running TF-IDF unique normalization slope experiment")
    slope_names = [f"{slope:.2f}" for slope in slopes]
    logging.info("Trying slopes %s.", " ,".join(slope_names))
    logging.info("Using weighting model %s.", weight_model_java_class)

    index_ref = _get_index_ref(experiment, documents)

    WModel = pt.autoclass(weight_model_java_class)

    def create_retriever(slope):
        wmodel = WModel(slope)
        retriever = pt.BatchRetrieve(index_ref, wmodel=wmodel) % 1000
        retriever = pt.apply.query(utils.sanitize_query) >> retriever

        return retriever

    results = pt.Experiment(
        [create_retriever(slope) for slope in slopes],
        topics,
        qrels,
        names=slope_names,
        eval_metrics=["map"],
    ).T

    logging.info("Resulting MAP scores:")
    print(results)
    results.to_csv(results_path, header=True)


def _get_index_ref(experiment: Experiment, documents: Documents) -> IndexRef:
    if os.path.exists(experiment.index_path):
        index_creation_time = os.path.getctime(experiment.index_path)
        logging.info(
            "Index loaded from disk at %s from %s.",
            experiment.index_path,
            time.strftime("%Y %d.%M. %H:%M:%S", time.gmtime(index_creation_time)),
        )
        return pt.IndexRef.of(experiment.index_path)

    logging.info("Indexing to %s...", experiment.index_path)
    index_ref = experiment.get_index(documents)
    logging.info("Indexing to %s...Done", experiment.index_path)
    return index_ref
