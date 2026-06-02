"""Hydra entrypoint for gene expression imputation."""

import logging
import os

import hydra
from omegaconf import DictConfig, OmegaConf

from transcriptformer.model.imputation import load_and_merge_with_checkpoint, run_imputation

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


@hydra.main(
    config_path=os.path.join(os.path.dirname(__file__), "conf"),
    config_name="imputation_config.yaml",
    version_base=None,
)
def main(cfg: DictConfig):
    logging.info("Imputation config:\n%s", OmegaConf.to_yaml(cfg))
    cfg = load_and_merge_with_checkpoint(cfg)

    adata_output = run_imputation(cfg, data_files=cfg.model.imputation_config.data_files)

    output_dir = cfg.model.imputation_config.output_path
    os.makedirs(output_dir, exist_ok=True)

    output_filename = getattr(cfg.model.imputation_config, "output_filename", "imputed_query_genes.h5ad")
    if not output_filename.endswith(".h5ad"):
        output_filename = f"{output_filename}.h5ad"

    save_file = os.path.join(output_dir, output_filename)
    adata_output.write_h5ad(save_file)
    logging.info("Saved imputation output to %s", save_file)


if __name__ == "__main__":
    main()
