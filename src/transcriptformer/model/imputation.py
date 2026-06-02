"""Imputation workflow for Transcriptformer models."""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Sequence
from dataclasses import asdict

import anndata
import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig

from transcriptformer.data.dataloader import get_counts_layer, load_gene_features, to_dense
from transcriptformer.data.dataclasses import BatchData, DataConfig, LossConfig, ModelConfig
from transcriptformer.model.embedding_surgery import change_embedding_layer
from transcriptformer.tokenizer.tokenizer import BatchObsTokenizer
from transcriptformer.tokenizer.vocab import load_vocabs_and_embeddings
from transcriptformer.utils.device import resolve_checkpoint_map_location

logger = logging.getLogger(__name__)


def _as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


def _bool_or_default(value: object, default: bool) -> bool:
    if value is None:
        return default
    return _as_bool(value)


def _int_or_default(value: object, default: int) -> int:
    if value is None:
        return default
    return int(value)


def _normalize_query_genes(impute_cfg: DictConfig) -> list[str]:
    query_genes: list[str] = []

    if getattr(impute_cfg, "query_genes", None) is not None:
        for gene in list(impute_cfg.query_genes):
            if gene is None:
                continue
            gene = str(gene).strip()
            if gene:
                query_genes.append(gene)

    query_file = getattr(impute_cfg, "query_genes_file", None)
    if query_file:
        with open(query_file) as f:
            for line in f:
                gene = line.strip()
                if gene and not gene.startswith("#"):
                    query_genes.append(gene)

    deduped: list[str] = []
    seen = set()
    for gene in query_genes:
        if gene in seen:
            continue
        seen.add(gene)
        deduped.append(gene)
    return deduped


def _order_observed_indices(
    counts_row: torch.Tensor,
    candidate_indices: torch.Tensor,
    order_mode: str,
    rng: torch.Generator | None,
) -> torch.Tensor:
    if candidate_indices.numel() == 0:
        return candidate_indices

    if order_mode == "count_desc":
        rank = torch.argsort(counts_row[candidate_indices], descending=True)
        return candidate_indices[rank]

    if order_mode == "random":
        if rng is None:
            perm = torch.randperm(candidate_indices.numel(), device=candidate_indices.device)
        else:
            perm = torch.randperm(candidate_indices.numel(), generator=rng, device=candidate_indices.device)
        return candidate_indices[perm]

    # input order fallback
    return candidate_indices


def _build_cell_sequence(
    counts_row: torch.Tensor,
    token_row: torch.Tensor,
    query_token_ids: torch.Tensor,
    observed_pool_indices: torch.Tensor,
    query_source_positions: torch.Tensor,
    seq_len: int,
    pad_idx: int,
    sort_genes: bool,
    randomize_genes: bool,
    treat_query_as_missing: bool,
    seed_query_with_observed_counts: bool,
    include_zero_observed: bool,
    rng: torch.Generator | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if include_zero_observed:
        observed_candidates = observed_pool_indices
    else:
        observed_nonzero_mask = counts_row[observed_pool_indices] > 0
        observed_candidates = observed_pool_indices[observed_nonzero_mask]

    if sort_genes:
        observed_candidates = _order_observed_indices(counts_row, observed_candidates, "count_desc", rng)
    elif randomize_genes:
        observed_candidates = _order_observed_indices(counts_row, observed_candidates, "random", rng)

    if query_token_ids.numel() > seq_len:
        raise ValueError(
            f"Number of query genes ({query_token_ids.numel()}) exceeds model sequence length ({seq_len})."
        )

    observed_budget = max(seq_len - query_token_ids.numel(), 0)
    observed_candidates = observed_candidates[:observed_budget]

    out_tokens = torch.full((seq_len,), pad_idx, dtype=token_row.dtype)
    out_counts = torch.zeros((seq_len,), dtype=counts_row.dtype)
    observed_mask = torch.zeros((seq_len,), dtype=torch.bool)

    obs_len = observed_candidates.numel()
    if obs_len > 0:
        out_tokens[:obs_len] = token_row[observed_candidates]
        out_counts[:obs_len] = counts_row[observed_candidates]
        observed_mask[:obs_len] = True

    query_start = obs_len
    query_end = query_start + query_token_ids.numel()
    if query_token_ids.numel() > 0:
        out_tokens[query_start:query_end] = query_token_ids

        if not treat_query_as_missing:
            for i, src_pos in enumerate(query_source_positions):
                if src_pos >= 0:
                    out_counts[query_start + i] = counts_row[src_pos]
                    observed_mask[query_start + i] = True
        elif seed_query_with_observed_counts:
            for i, src_pos in enumerate(query_source_positions):
                if src_pos >= 0:
                    out_counts[query_start + i] = counts_row[src_pos]

    return out_tokens, out_counts, observed_mask


def _prepare_sequence_maps(
    token_row: torch.Tensor,
    query_token_ids: torch.Tensor,
    pad_idx: int,
    unknown_idx: int,
    filter_to_vocabs: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    valid = token_row != pad_idx
    if filter_to_vocabs:
        valid = valid & (token_row != unknown_idx)

    non_query = ~torch.isin(token_row, query_token_ids) if query_token_ids.numel() > 0 else torch.ones_like(valid)
    observed_pool_indices = torch.nonzero(valid & non_query, as_tuple=True)[0]

    source_positions = []
    for qid in query_token_ids.tolist():
        idx = torch.nonzero((token_row == qid) & valid, as_tuple=True)[0]
        source_positions.append(int(idx[0].item()) if idx.numel() > 0 else -1)
    query_source_positions = torch.tensor(source_positions, dtype=torch.long)

    return observed_pool_indices, query_source_positions


def _eligible_cell_indices(
    X: np.ndarray,
    observed_pool_indices: torch.Tensor,
    include_zero_observed: bool,
    min_expressed_genes: int,
) -> np.ndarray:
    n_cells = X.shape[0]
    if min_expressed_genes <= 0:
        return np.arange(n_cells, dtype=np.int64)

    if observed_pool_indices.numel() == 0:
        return np.array([], dtype=np.int64)

    if include_zero_observed:
        if observed_pool_indices.numel() >= min_expressed_genes:
            return np.arange(n_cells, dtype=np.int64)
        return np.array([], dtype=np.int64)

    obs_idx = observed_pool_indices.cpu().numpy()
    nnz = np.count_nonzero(X[:, obs_idx] > 0, axis=1)
    return np.nonzero(nnz >= min_expressed_genes)[0].astype(np.int64)


def _extract_query_matrix(imputed_counts: torch.Tensor, gene_tokens: torch.Tensor, query_token_ids: torch.Tensor) -> torch.Tensor:
    batch_size = imputed_counts.shape[0]
    n_query = query_token_ids.numel()
    out = torch.full((batch_size, n_query), float("nan"), device=imputed_counts.device, dtype=imputed_counts.dtype)

    if n_query == 0:
        return out

    for j, qid in enumerate(query_token_ids):
        mask = gene_tokens == qid
        denom = mask.sum(dim=1)
        has_value = denom > 0
        if has_value.any():
            values = (imputed_counts * mask.float()).sum(dim=1) / torch.clamp(denom.float(), min=1.0)
            out[has_value, j] = values[has_value]

    return out


def _build_model(cfg: DictConfig):
    (gene_vocab, aux_vocab), emb_matrix = load_vocabs_and_embeddings(cfg)

    model_type = cfg.model.get("model_type", "transcriptformer")
    if model_type == "esm2ce":
        from transcriptformer.model.model import ESM2CE as ModelClass
    elif model_type == "transcriptformer":
        from transcriptformer.model.model import Transcriptformer as ModelClass
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    impute_cfg = cfg.model.imputation_config

    model = ModelClass(
        data_config=cfg.model.data_config,
        model_config=cfg.model.model_config,
        loss_config=cfg.model.loss_config,
        inference_config=None,
        gene_vocab_dict=gene_vocab,
        aux_vocab_dict=aux_vocab,
        emb_matrix=emb_matrix,
    )

    if not getattr(impute_cfg, "load_checkpoint", None):
        raise ValueError("Missing model.imputation_config.load_checkpoint")

    map_location = resolve_checkpoint_map_location(getattr(impute_cfg, "device", "auto"))
    state_dict = torch.load(impute_cfg.load_checkpoint, weights_only=True, map_location=map_location)
    model.load_state_dict(state_dict)

    if getattr(impute_cfg, "pretrained_embedding", None) is not None:
        pretrained = impute_cfg.pretrained_embedding
        pretrained_paths = pretrained if isinstance(pretrained, list) else [pretrained]
        model, gene_vocab = change_embedding_layer(model, pretrained_paths)

    model.eval()
    return model, gene_vocab, aux_vocab


def run_imputation(cfg: DictConfig, data_files: Sequence[str] | None = None) -> anndata.AnnData:
    """Run imputation end-to-end from AnnData input to query-gene output AnnData."""
    data_files = list(data_files or cfg.model.imputation_config.data_files)
    if not data_files:
        raise ValueError("No data files provided for imputation")

    model, gene_vocab, aux_vocab = _build_model(cfg)

    device_name = resolve_checkpoint_map_location(getattr(cfg.model.imputation_config, "device", "auto"))
    device = torch.device(device_name)
    model = model.to(device)

    impute_cfg = cfg.model.imputation_config
    query_genes = _normalize_query_genes(impute_cfg)
    if not query_genes:
        raise ValueError("No query genes provided. Set model.imputation_config.query_genes or query_genes_file")

    unknown_idx = gene_vocab["unknown"]
    query_token_ids = []
    missing_from_vocab = []
    for gene in query_genes:
        token_id = gene_vocab.get(gene, unknown_idx)
        if token_id == unknown_idx:
            missing_from_vocab.append(gene)
            continue
        query_token_ids.append(token_id)

    if not query_token_ids:
        raise ValueError("None of the query genes are in model vocabulary")

    query_token_tensor = torch.tensor(query_token_ids, dtype=torch.long, device=device)

    if missing_from_vocab:
        logger.warning("Skipping %s query genes not found in vocab", len(missing_from_vocab))

    rng = None
    if getattr(impute_cfg, "seed", None) is not None:
        rng = torch.Generator(device="cpu")
        rng.manual_seed(int(impute_cfg.seed))

    all_outputs = []
    all_obs = []

    batch_size = int(impute_cfg.batch_size)
    seq_len = int(cfg.model.model_config.seq_len)
    treat_query_as_missing = _as_bool(getattr(impute_cfg, "treat_query_as_missing", True))
    seed_query_with_observed = _as_bool(getattr(impute_cfg, "seed_query_with_observed_counts", False))
    include_zero_observed = _as_bool(getattr(impute_cfg, "include_zero_observed", False))
    sort_genes = _bool_or_default(getattr(cfg.model.data_config, "sort_genes", None), False)
    randomize_genes = _bool_or_default(getattr(cfg.model.data_config, "randomize_genes", None), False)
    filter_to_vocabs = _bool_or_default(getattr(cfg.model.data_config, "filter_to_vocabs", None), True)
    min_expressed_genes = _int_or_default(getattr(cfg.model.data_config, "min_expressed_genes", None), 0)

    obs_keys = list(getattr(impute_cfg, "obs_keys", []) or [])

    for file_path in data_files:
        adata, success = (anndata.read_h5ad(file_path), True)
        if not success:
            logger.warning("Skipping failed file: %s", file_path)
            continue

        gene_names, success, adata = load_gene_features(
            adata,
            cfg.model.data_config.gene_col_name,
            cfg.model.data_config.remove_duplicate_genes,
            use_raw=cfg.model.data_config.use_raw,
        )
        if not success:
            logger.warning("Skipping file due to gene feature load failure: %s", file_path)
            continue

        X = to_dense(get_counts_layer(adata, cfg.model.data_config.use_raw))
        X = np.asarray(X)

        token_ids_np = np.array([gene_vocab.get(gene, unknown_idx) for gene in gene_names], dtype=np.int64)
        token_ids = torch.from_numpy(token_ids_np)

        observed_pool_indices, query_source_positions = _prepare_sequence_maps(
            token_row=token_ids,
            query_token_ids=query_token_tensor.cpu(),
            pad_idx=model.gene_vocab.pad_idx,
            unknown_idx=unknown_idx,
            filter_to_vocabs=filter_to_vocabs,
        )

        eligible_rows = _eligible_cell_indices(
            X=X,
            observed_pool_indices=observed_pool_indices,
            include_zero_observed=include_zero_observed,
            min_expressed_genes=min_expressed_genes,
        )

        if eligible_rows.size == 0:
            logger.warning("No eligible cells after min_expressed_genes/filter settings for file: %s", file_path)
            continue

        X = X[eligible_rows]

        n_cells = X.shape[0]
        obs_df = adata.obs.iloc[eligible_rows].copy()

        if obs_keys and "all" not in obs_keys:
            keep = [key for key in obs_keys if key in obs_df.columns]
            obs_df = obs_df[keep]

        for start in range(0, n_cells, batch_size):
            end = min(start + batch_size, n_cells)
            counts_block = torch.tensor(X[start:end], dtype=torch.float32)
            cell_count = counts_block.shape[0]

            batch_tokens = []
            batch_counts = []
            batch_obs_mask = []
            kept_local_rows = []
            for i in range(cell_count):
                tokens_i, counts_i, obs_mask_i = _build_cell_sequence(
                    counts_row=counts_block[i],
                    token_row=token_ids,
                    query_token_ids=query_token_tensor.cpu(),
                    observed_pool_indices=observed_pool_indices,
                    query_source_positions=query_source_positions,
                    seq_len=seq_len,
                    pad_idx=model.gene_vocab.pad_idx,
                    sort_genes=sort_genes,
                    randomize_genes=randomize_genes,
                    treat_query_as_missing=treat_query_as_missing,
                    seed_query_with_observed_counts=seed_query_with_observed,
                    include_zero_observed=include_zero_observed,
                    rng=rng,
                )
                batch_tokens.append(tokens_i)
                batch_counts.append(counts_i)
                batch_obs_mask.append(obs_mask_i)

                kept_local_rows.append(i)

            gene_token_indices = torch.stack(batch_tokens).to(device)
            gene_counts = torch.stack(batch_counts).to(device)
            observed_mask = torch.stack(batch_obs_mask).to(device)

            aux_token_indices = None
            if aux_vocab is not None:
                aux_tokenizer = BatchObsTokenizer(aux_vocab)
                obs_batch = adata.obs.iloc[start:end]
                aux_all = torch.stack([aux_tokenizer(obs) for _, obs in obs_batch.iterrows()])
                aux_token_indices = aux_all[kept_local_rows].to(device)

            batch = BatchData(
                gene_counts=gene_counts,
                gene_token_indices=gene_token_indices,
                aux_token_indices=aux_token_indices,
                file_path=file_path,
                obs=None,
            )

            total_counts = None
            total_count_obs_key = getattr(impute_cfg, "total_count_obs_key", None)
            if total_count_obs_key and total_count_obs_key in adata.obs.columns:
                total_counts = torch.tensor(
                    adata.obs.iloc[start:end][total_count_obs_key].to_numpy(dtype=np.float32)[kept_local_rows],
                    device=device,
                )
            elif getattr(impute_cfg, "total_count_value", None) is not None:
                total_counts = float(impute_cfg.total_count_value)

            result = model.impute_gene_expression(
                batch=batch,
                observed_mask=observed_mask,
                query_gene_ids=query_token_tensor,
                total_counts=total_counts,
                count_scale=getattr(impute_cfg, "count_scale", None),
                observed_fraction=getattr(impute_cfg, "observed_fraction", None),
                initial_missing_counts=(gene_counts if seed_query_with_observed else None),
                num_iters=int(getattr(impute_cfg, "num_iters", 2)),
                eps=float(getattr(impute_cfg, "eps", 1e-6)),
            )

            query_matrix = _extract_query_matrix(result["imputed_counts"], batch.gene_token_indices, query_token_tensor)
            all_outputs.append(query_matrix.detach().cpu().numpy())
            all_obs.append(obs_df.iloc[start:end].iloc[kept_local_rows].copy())

    if not all_outputs:
        raise ValueError("No imputation outputs produced")

    output_matrix = np.vstack(all_outputs)
    output_obs = pd.concat(all_obs, axis=0)
    output_obs.index = output_obs.index.astype(str)

    output = anndata.AnnData(X=output_matrix, obs=output_obs)
    kept_query_genes = [gene for gene in query_genes if gene not in missing_from_vocab]
    output.var_names = np.array(kept_query_genes, dtype=str)
    output.uns["imputation"] = {
        "num_iters": int(getattr(impute_cfg, "num_iters", 2)),
        "count_scale": None if getattr(impute_cfg, "count_scale", None) is None else float(impute_cfg.count_scale),
        "observed_fraction": (
            None if getattr(impute_cfg, "observed_fraction", None) is None else float(impute_cfg.observed_fraction)
        ),
        "treat_query_as_missing": bool(treat_query_as_missing),
        "sort_genes": bool(sort_genes),
        "randomize_genes": bool(randomize_genes),
        "missing_query_genes_from_vocab": missing_from_vocab,
        "data_files": list(data_files),
    }

    return output


def load_and_merge_with_checkpoint(cfg: DictConfig) -> DictConfig:
    """Build imputation config from checkpoint base plus explicit runtime overrides."""
    config_path = os.path.join(cfg.model.checkpoint_path, "config.json")
    with open(config_path) as f:
        config_dict = json.load(f)

    from omegaconf import OmegaConf

    checkpoint_model_cfg = config_dict["model"]

    # Runtime YAML is the source of user-facing runtime knobs (inference/imputation).
    merged = OmegaConf.create(OmegaConf.to_container(cfg, resolve=False))

    # Materialize ModelConfig through dataclass construction so optional defaults
    # (e.g. compile_block_mask=True) are populated for older checkpoints.
    model_cfg_json = dict(checkpoint_model_cfg["model_config"])
    model_cfg_json.pop("_target_", None)
    merged.model.model_config = OmegaConf.create(asdict(ModelConfig(**model_cfg_json)))

    # Materialize LossConfig through dataclass construction.
    loss_cfg_json = dict(checkpoint_model_cfg["loss_config"])
    loss_cfg_json.pop("_target_", None)
    merged.model.loss_config = OmegaConf.create(asdict(LossConfig(**loss_cfg_json)))

    # Keep top-level model selectors from runtime.
    merged.model.checkpoint_path = cfg.model.checkpoint_path
    if getattr(cfg.model, "model_type", None) is not None:
        merged.model.model_type = cfg.model.model_type

    # Materialize DataConfig from checkpoint base, then apply non-null YAML overrides.
    data_cfg_json = dict(checkpoint_model_cfg["data_config"])
    data_cfg_json.pop("_target_", None)
    runtime_data = OmegaConf.to_container(cfg.model.data_config, resolve=False)
    for key, value in runtime_data.items():
        if key.startswith("_"):
            continue
        if value is not None:
            data_cfg_json[key] = value
    merged.model.data_config = OmegaConf.create(asdict(DataConfig(**data_cfg_json)))

    # Imputation controls are fully runtime-managed.
    merged.model.imputation_config = cfg.model.imputation_config

    # Required imputation runtime fields.
    if getattr(merged.model.imputation_config, "data_files", None) is None:
        merged.model.imputation_config.data_files = []
    if getattr(merged.model.imputation_config, "output_path", None) is None:
        merged.model.imputation_config.output_path = "./imputation_results"
    if getattr(merged.model.imputation_config, "output_filename", None) is None:
        merged.model.imputation_config.output_filename = "imputed_query_genes.h5ad"
    if getattr(merged.model.imputation_config, "batch_size", None) is None:
        merged.model.imputation_config.batch_size = 8
    if getattr(merged.model.imputation_config, "obs_keys", None) is None:
        merged.model.imputation_config.obs_keys = ["all"]
    if getattr(merged.model.imputation_config, "device", None) is None:
        merged.model.imputation_config.device = "auto"
    if "pretrained_embedding" not in merged.model.imputation_config:
        merged.model.imputation_config.pretrained_embedding = None

    merged.model.imputation_config.load_checkpoint = os.path.join(merged.model.checkpoint_path, "model_weights.pt")
    merged.model.data_config.aux_vocab_path = os.path.join(merged.model.checkpoint_path, "vocabs")
    merged.model.data_config.esm2_mappings_path = os.path.join(merged.model.checkpoint_path, "vocabs")

    return merged
