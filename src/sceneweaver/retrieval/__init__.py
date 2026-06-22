"""Experience-card retrieval service."""

from sceneweaver.retrieval.models import QueryPlan, QueryUseCase, RetrievalRequest, RetrievalWeights
from sceneweaver.retrieval.query_plan import build_query_plan
from sceneweaver.retrieval.usecase import build_script_usecase, infer_query_usecase


def retrieve_experience_matches(*args, **kwargs):
    from sceneweaver.retrieval.service import retrieve_experience_matches as _retrieve_experience_matches

    return _retrieve_experience_matches(*args, **kwargs)


__all__ = [
    "QueryPlan",
    "QueryUseCase",
    "RetrievalRequest",
    "RetrievalWeights",
    "build_script_usecase",
    "build_query_plan",
    "infer_query_usecase",
    "retrieve_experience_matches",
]
