"""Tests for constrained vs unconstrained imputation scaling behavior."""

import torch

from transcriptformer.data.dataclasses import BatchData, DataConfig, LossConfig, ModelConfig
from transcriptformer.model.model import Transcriptformer


def _tiny_model() -> Transcriptformer:
    gene_vocab = {
        "unknown": 0,
        "[PAD]": 1,
        "[START]": 2,
        "[END]": 3,
        "[RD]": 4,
        "[CELL]": 5,
        "[MASK]": 6,
        "g1": 7,
        "g2": 8,
        "g3": 9,
    }
    aux_vocab = {"assay": {"unknown": 0, "new_assay": 1}}

    data_config = DataConfig(
        aux_vocab_path=".",
        pin_memory=False,
        aux_cols=["assay"],
        gene_col_name="ensembl_id",
        clip_counts=30,
        filter_to_vocabs=True,
        filter_outliers=0.0,
        pad_zeros=True,
        normalize_to_scale=0,
        n_data_workers=0,
        sort_genes=False,
        randomize_genes=False,
        min_expressed_genes=0,
        gene_pad_token="[PAD]",
        aux_pad_token="unknown",
    )

    model_config = ModelConfig(
        log_counts_eps=1e-6,
        num_heads=2,
        num_layers=1,
        model_dim=16,
        embed_dim=8,
        dropout=0.0,
        activation="gelu",
        attn_bias=False,
        fw_bias=False,
        mu_link_fn="softplus",
        softcap=0,
        seq_len=3,
        aux_len=1,
        block_len=2,
        compile_block_mask=False,
    )

    loss_config = LossConfig(gene_id_loss_weight=1.0, softplus_approx=True)
    emb_matrix = torch.randn(len(gene_vocab), model_config.embed_dim)

    model = Transcriptformer(
        data_config=data_config,
        model_config=model_config,
        loss_config=loss_config,
        gene_vocab_dict=gene_vocab,
        aux_vocab_dict=aux_vocab,
        emb_matrix=emb_matrix,
    )
    model.eval()
    return model


def _tiny_batch() -> BatchData:
    return BatchData(
        gene_counts=torch.tensor([[5.0, 3.0, 2.0], [2.0, 1.0, 4.0]], dtype=torch.float32),
        gene_token_indices=torch.tensor([[7, 8, 9], [8, 7, 9]], dtype=torch.int64),
        aux_token_indices=torch.tensor([[0], [1]], dtype=torch.int64),
    )


def test_constrained_imputation_ignores_explicit_total():
    model = _tiny_model()
    batch = _tiny_batch()

    observed_mask = torch.tensor([[True, True, False], [True, False, True]], dtype=torch.bool)

    no_total = model.impute_gene_expression(
        batch=batch,
        observed_mask=observed_mask,
        num_iters=1,
    )
    with_total = model.impute_gene_expression(
        batch=batch,
        observed_mask=observed_mask,
        total_counts=10000.0,
        num_iters=1,
    )

    assert torch.allclose(no_total["imputed_counts"], with_total["imputed_counts"], atol=1e-5, rtol=1e-5)


def test_unconstrained_imputation_honors_explicit_total():
    model = _tiny_model()
    batch = _tiny_batch()

    observed_mask = torch.zeros_like(batch.gene_counts, dtype=torch.bool)
    result = model.impute_gene_expression(
        batch=batch,
        observed_mask=observed_mask,
        total_counts=10000.0,
        num_iters=1,
    )

    row_totals = result["imputed_counts"].sum(dim=1)
    expected = torch.full_like(row_totals, 10000.0)
    assert torch.allclose(row_totals, expected, atol=1e-3, rtol=1e-5)


"""Tests for imputation config merge behavior with legacy checkpoints."""

import json
from omegaconf import OmegaConf
from transcriptformer.model.imputation import load_and_merge_with_checkpoint


def test_load_and_merge_with_checkpoint_handles_missing_inference_config(tmp_path):
    """Legacy checkpoints without model.inference_config should still merge cleanly."""
    ckpt_dir = tmp_path / "ckpt"
    ckpt_dir.mkdir()

    config_json = {
        "model": {
            "data_config": {
                "aux_vocab_path": None,
                "pin_memory": True,
                "aux_cols": "assay",
                "gene_col_name": "ensembl_id",
                "clip_counts": None,
                "filter_to_vocabs": None,
                "filter_outliers": None,
                "pad_zeros": True,
                "normalize_to_scale": 0,
                "n_data_workers": 1,
                "sort_genes": None,
                "randomize_genes": None,
                "min_expressed_genes": None,
                "gene_pad_token": "[PAD]",
                "aux_pad_token": "unknown",
                "use_raw": None,
                "remove_duplicate_genes": False,
            },
            "model_config": {
                "log_counts_eps": 1e-6,
                "num_heads": 16,
                "num_layers": 12,
                "model_dim": 256,
                "embed_dim": 256,
                "dropout": 0.1,
                "activation": "gelu",
                "attn_bias": False,
                "fw_bias": False,
                "mu_link_fn": "softmax",
                "softcap": 10,
                "seq_len": 128,
                "aux_len": 1,
                "block_len": 1,
            },
            "loss_config": {
                "gene_id_loss_weight": 1.0,
                "softplus_approx": True,
            },
        }
    }
    (ckpt_dir / "config.json").write_text(json.dumps(config_json), encoding="utf-8")

    cfg = OmegaConf.create(
        {
            "model": {
                "checkpoint_path": str(ckpt_dir),
                "model_type": "transcriptformer",
                "inference_config": {},
                "data_config": {
                    "gene_col_name": "ensembl_id",
                    "filter_to_vocabs": True,
                    "sort_genes": False,
                    "randomize_genes": False,
                    "min_expressed_genes": 0,
                    "use_raw": None,
                    "remove_duplicate_genes": False,
                },
                "imputation_config": {
                    "data_files": ["dummy.h5ad"],
                    "output_path": "./tmp_impute",
                    "batch_size": 4,
                    "obs_keys": ["all"],
                    "device": "cpu",
                    "pretrained_embedding": None,
                    "query_genes": ["ENSG000001"],
                },
            }
        }
    )

    merged = load_and_merge_with_checkpoint(cfg)

    assert merged.model.imputation_config.device == "cpu"
    assert merged.model.imputation_config.pretrained_embedding is None
    assert merged.model.imputation_config.batch_size == 4
    assert merged.model.imputation_config.load_checkpoint == str(ckpt_dir / "model_weights.pt")
    assert merged.model.model_config.compile_block_mask is True
