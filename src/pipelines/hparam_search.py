"""Entry point: hyperparameter search.

    python -m src.pipelines.hparam_search
"""

from src.models.hparam import run_hparam_search
from src.utils.log import log_run

if __name__ == "__main__":
    with log_run("hparam_search"):
        run_hparam_search()
