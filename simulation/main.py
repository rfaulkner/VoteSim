"""Main entry point for the VoteSim simulation pipeline and personalized query functions.
"""

import logging
import os
import sys

import hydra
import omegaconf
from simulation.mode_compare import run_compare
from simulation.mode_judge import run_judge
from simulation.run import run_pipeline
from simulation.run import run_query


# Add the current directory to path to allow imports if run as script
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: omegaconf.DictConfig):
  debug = cfg.get("debug", False)
  log_level = logging.DEBUG if debug else logging.INFO
  logging.basicConfig(
      level=log_level,
      format="%(asctime)s[%(name)s][%(levelname)s] - %(message)s",
  )
  if debug:
    logging.info("*** DEBUG MODE ENABLED ***")
  logging.info("Config:\n%s", omegaconf.OmegaConf.to_yaml(cfg))

  mode = cfg.get("mode", "pipeline")
  logging.info("Running in '%s' mode.", mode)

  if mode == "query":
    run_query(cfg)
  elif mode == "pipeline":
    run_pipeline(cfg)
  elif mode == "compare":
    run_compare(cfg)
  elif mode == "judge":
    run_judge(cfg)
  else:
    raise ValueError(
        f"Unknown mode '{mode}'. Expected 'pipeline', 'query',"
        f" 'compare', or 'judge'."
    )


if __name__ == "__main__":
  main()

