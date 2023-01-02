import pyterrier as pt

from src import utils
from src.experiments import run_0
from src.types import IndexRef, Results, Topics


class Run2(run_0.Run0WeightingModel):
    def get_results(self, index_ref: IndexRef, topics: Topics) -> Results:
        retriever = (
            pt.BatchRetrieve(index_ref, wmodel=self._wmodel, controls={"qe": "on"})
            % 1000
        )
        retriever = pt.apply.query(utils.sanitize_query) >> retriever

        return retriever.transform(topics)
