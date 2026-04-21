import hydra
from omegaconf import DictConfig, OmegaConf
import logging
import sys
import os

# Add the current directory to path to allow imports if run as script
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from run import run_query

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info("Starting simple model query.")
    logging.info("Config:\n%s", OmegaConf.to_yaml(cfg))
    
    run_query(cfg)

if __name__ == "__main__":
    main()
