"""Tests for the TranscriptFormer CLI module."""

import argparse
import sys
from unittest import mock

import pytest
from omegaconf import OmegaConf

from transcriptformer.cli import (
    main,
    run_impute_cli,
    run_train_cli,
    setup_impute_parser,
    setup_inference_parser,
    setup_train_parser,
)


class TestCLIMain:
    """Tests for the main CLI entry point."""

    def test_main_no_args(self, monkeypatch, capsys):
        """Test CLI with no arguments prints help and exits."""
        monkeypatch.setattr(sys, "argv", ["transcriptformer"])
        with mock.patch("sys.exit") as mock_exit:
            main()
            mock_exit.assert_called_once_with(1)

        captured = capsys.readouterr()
        assert "usage: " in captured.out
        assert "TranscriptFormer command-line interface" in captured.out

    def test_main_help(self, monkeypatch, capsys):
        """Test CLI with --help argument prints help."""
        monkeypatch.setattr(sys, "argv", ["transcriptformer", "--help"])
        with pytest.raises(SystemExit):
            main()

        captured = capsys.readouterr()
        assert "usage: " in captured.out
        assert "TranscriptFormer command-line interface" in captured.out


class TestInferenceCommand:
    """Tests for the inference command."""

    @mock.patch("transcriptformer.cli.run_inference_cli")
    def test_inference_command(self, mock_run_inference, monkeypatch):
        """Test that inference command runs with required arguments."""
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "transcriptformer",
                "inference",
                "--checkpoint-path",
                "/path/to/checkpoint",
                "--data-file",
                "/path/to/data.h5ad",
            ],
        )

        main()
        mock_run_inference.assert_called_once()


class TestTrainCommand:
    """Tests for the train command."""

    @mock.patch("transcriptformer.cli.run_train_cli")
    def test_train_command(self, mock_run_train, monkeypatch):
        """Test that train command runs with required arguments."""
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "transcriptformer",
                "train",
                "--checkpoint-dir",
                "/path/to/checkpoint",
                "--output-dir",
                "/path/to/output",
                "--train-file",
                "/path/to/train.h5ad",
            ],
        )

        main()
        mock_run_train.assert_called_once()

    @mock.patch("transcriptformer.cli.run_train_from_dict")
    @mock.patch("transcriptformer.cli.setup_runtime_for_training")
    def test_run_train_cli(self, mock_setup_runtime, mock_run_train_from_dict):
        """Test run_train_cli parameter mapping."""
        args = mock.MagicMock()
        args.checkpoint_dir = "/path/to/checkpoint"
        args.resume_artifact_dir = None
        args.resume_mode = "weights"
        args.output_dir = "/path/to/output"
        args.train_file = ["/path/to/train.h5ad"]
        args.val_file = []
        args.expanded_assay_vocab = None
        args.obs_assay_col = "assay"
        args.gene_col_name = "ensembl_id"
        args.filter_to_vocabs = True
        args.filter_outliers = 0.0
        args.sort_genes = False
        args.randomize_genes = False
        args.min_expressed_genes = 0
        args.n_data_workers = 4
        args.batch_size = 2
        args.num_workers = 0
        args.max_epochs = 1
        args.precision = "32"
        args.device = "cpu"
        args.num_gpus = 1
        args.num_nodes = 1
        args.lr = 1e-4
        args.weight_decay = 0.0
        args.adam_beta1 = 0.9
        args.adam_beta2 = 0.95
        args.adam_eps = 1e-8
        args.warmup_ratio = 0.1
        args.min_lr_ratio = 0.1
        args.gene_id_loss_weight = 1.0
        args.softplus_approx = True
        args.init_default_source = "unknown"
        args.assay_init_map = []
        args.freeze_transformer = False
        args.unfreeze_last_n_transformer_blocks = 0
        args.freeze_gene_embeddings = False
        args.freeze_count_head = False
        args.freeze_gene_head = False
        args.train_aux_only = False
        args.shuffle_expressed_each_batch = False
        args.clip_counts = 30.0
        args.normalize_to_scale = 0.0
        args.use_raw = False
        args.remove_duplicate_genes = False
        args.use_oom_dataloader = False
        args.file_aware_batching = True
        args.oom_batches_per_file = 1
        args.seed = 42

        # Mock return value
        mock_run_train_from_dict.return_value = {"output_dir": "/path/to/output"}

        # Call the function
        run_train_cli(args)

        # Verify setup was called
        mock_setup_runtime.assert_called_once()

        # Verify run_train_from_dict was called with correct config
        mock_run_train_from_dict.assert_called_once()
        call_args = mock_run_train_from_dict.call_args[0][0]
        assert call_args["checkpoint_dir"] == "/path/to/checkpoint"
        assert call_args["output_dir"] == "/path/to/output"
        assert call_args["train_files"] == ["/path/to/train.h5ad"]
        assert call_args["data_config"]["gene_col_name"] == "ensembl_id"
        assert call_args["device"] == "cpu"
        assert call_args["num_gpus"] == 1
        assert call_args["unfreeze_last_n_transformer_blocks"] == 0
        assert call_args["enable_file_aware_batching"] is True
        assert call_args["oom_batches_per_file"] == 1
        assert call_args["loss_config"]["gene_id_loss_weight"] == 1.0


class TestImputeCommand:
    """Tests for the impute command."""

    @mock.patch("transcriptformer.cli.run_impute_cli")
    def test_impute_command(self, mock_run_impute, monkeypatch):
        """Test that impute command runs with required arguments."""
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "transcriptformer",
                "impute",
                "--checkpoint-path",
                "/path/to/checkpoint",
                "--data-file",
                "/path/to/data.h5ad",
            ],
        )

        main()
        mock_run_impute.assert_called_once()

    @mock.patch("transcriptformer.model.imputation.run_imputation")
    @mock.patch("transcriptformer.model.imputation.load_and_merge_with_checkpoint")
    @mock.patch("transcriptformer.cli.OmegaConf.load")
    @mock.patch("transcriptformer.cli.os.makedirs")
    def test_run_impute_cli(self, mock_makedirs, mock_cfg_load, mock_load_merge, mock_run_imputation):
        """Test run_impute_cli parameter mapping and output writing."""
        cfg = {
            "model": {
                "checkpoint_path": "",
                "imputation_config": {
                    "data_files": [],
                    "output_path": "",
                    "output_filename": "imputed_query_genes.h5ad",
                    "batch_size": 8,
                    "obs_keys": ["all"],
                    "device": "auto",
                    "pretrained_embedding": None,
                    "num_iters": 3,
                    "treat_query_as_missing": True,
                    "seed_query_with_observed_counts": False,
                    "include_zero_observed": False,
                    "count_scale": None,
                    "observed_fraction": None,
                    "total_count_obs_key": None,
                    "total_count_value": None,
                    "query_genes_file": None,
                    "query_genes": [],
                },
                "data_config": {
                    "gene_col_name": "ensembl_id",
                    "use_raw": None,
                    "remove_duplicate_genes": False,
                    "filter_to_vocabs": True,
                    "min_expressed_genes": 0,
                    "sort_genes": None,
                    "randomize_genes": None,
                },
                "inference_config": {},
            }
        }
        mock_cfg_load.return_value = OmegaConf.create(cfg)
        mock_load_merge.side_effect = lambda input_cfg: input_cfg

        mock_adata = mock.MagicMock()
        mock_run_imputation.return_value = mock_adata

        args = mock.MagicMock()
        args.checkpoint_path = "/path/to/checkpoint"
        args.data_file = ["/path/to/data.h5ad"]
        args.query_genes = "ENSG1,ENSG2"
        args.query_genes_file = None
        args.output_path = "/path/to/output"
        args.output_filename = "my_imputed"
        args.batch_size = 4
        args.num_iters = 5
        args.treat_query_as_missing = True
        args.seed_query_with_observed_counts = True
        args.include_zero_observed = True
        args.count_scale = 1.5
        args.observed_fraction = 0.8
        args.total_count_obs_key = "library_size"
        args.total_count_value = None
        args.gene_col_name = "ensembl_id"
        args.filter_to_vocabs = False
        args.min_expressed_genes = 10
        args.sort_genes = True
        args.randomize_genes = False
        args.use_raw = None
        args.remove_duplicate_genes = True
        args.device = "cpu"
        args.config_override = [
            "model.data_config.min_expressed_genes=12",
            "model.imputation_config.total_count_value=2000",
        ]

        run_impute_cli(args)

        mock_load_merge.assert_called_once()
        mapped_cfg = mock_load_merge.call_args[0][0]
        assert mapped_cfg.model.checkpoint_path == "/path/to/checkpoint"
        assert mapped_cfg.model.imputation_config.data_files == ["/path/to/data.h5ad"]
        assert mapped_cfg.model.imputation_config.query_genes == ["ENSG1", "ENSG2"]
        assert mapped_cfg.model.imputation_config.num_iters == 5
        assert mapped_cfg.model.imputation_config.total_count_value == 2000
        assert mapped_cfg.model.data_config.filter_to_vocabs is False
        assert mapped_cfg.model.data_config.min_expressed_genes == 12
        assert mapped_cfg.model.data_config.sort_genes is True
        assert mapped_cfg.model.data_config.randomize_genes is False
        assert mapped_cfg.model.imputation_config.device == "cpu"
        assert mapped_cfg.model.imputation_config.batch_size == 4
        assert mapped_cfg.model.imputation_config.output_path == "/path/to/output"
        assert mapped_cfg.model.imputation_config.output_filename == "my_imputed"

        mock_run_imputation.assert_called_once_with(mapped_cfg, data_files=["/path/to/data.h5ad"])
        mock_makedirs.assert_called_once_with("/path/to/output", exist_ok=True)
        mock_adata.write_h5ad.assert_called_once_with("/path/to/output/my_imputed.h5ad")


class TestCLIParsers:
    """Tests for CLI parsers setup."""

    def test_inference_parser_setup(self):
        """Test that inference parser is set up correctly."""
        parser = mock.MagicMock()
        subparsers = mock.MagicMock()
        subparsers.add_parser.return_value = parser

        setup_inference_parser(subparsers)

        subparsers.add_parser.assert_called_once_with(
            "inference",
            help="Run inference with a TranscriptFormer model",
            description="Run inference with a TranscriptFormer model on scRNA-seq data.",
        )

        parser.add_argument.assert_any_call(
            "--checkpoint-path",
            required=True,
            help="Path to the model checkpoint directory",
        )
        parser.add_argument.assert_any_call(
            "--data-file",
            required=True,
            help="Path to input AnnData file to run inference on",
        )

        parser.add_argument.assert_any_call(
            "--emb-type",
            default="cell",
            choices=["cell", "cge"],
            help="Type of embeddings to extract: 'cell' for mean-pooled cell embeddings or 'cge' for contextual gene embeddings (default: cell)",
        )

    def test_train_parser_setup(self):
        """Test that train parser is set up correctly."""
        parser = mock.MagicMock()
        subparsers = mock.MagicMock()
        subparsers.add_parser.return_value = parser

        setup_train_parser(subparsers)

        subparsers.add_parser.assert_called_once_with(
            "train",
            help="Train or continue training with expanded assay vocab",
            description="Fine-tune TranscriptFormer with expanded assay tokens and optional freezing.",
        )

        parser.add_argument.assert_any_call(
            "--checkpoint-dir",
            required=True,
            help="Base artifact directory with config.json/model_weights.pt",
        )
        parser.add_argument.assert_any_call(
            "--output-dir",
            required=True,
            help="Output artifact directory",
        )
        parser.add_argument.assert_any_call(
            "--train-file",
            action="append",
            required=True,
            help="Training .h5ad file (repeatable)",
        )
        parser.add_argument.assert_any_call(
            "--file-aware-batching",
            action=argparse.BooleanOptionalAction,
            default=True,
            help="Keep OOM batches mostly file-local; disable to fall back to Lightning DistributedSampler batching",
        )

    def test_impute_parser_setup(self):
        """Test that impute parser is set up correctly."""
        parser = mock.MagicMock()
        subparsers = mock.MagicMock()
        subparsers.add_parser.return_value = parser

        setup_impute_parser(subparsers)

        subparsers.add_parser.assert_called_once_with(
            "impute",
            help="Run gene expression imputation for selected query genes",
            description="Impute expression values for query genes using TranscriptFormer.",
        )

        parser.add_argument.assert_any_call(
            "--checkpoint-path",
            required=True,
            help="Path to model checkpoint directory",
        )
        parser.add_argument.assert_any_call(
            "--data-file",
            action="append",
            required=True,
            help="Input .h5ad file (repeatable)",
        )
        parser.add_argument.assert_any_call(
            "--num-iters",
            type=int,
            default=3,
            help="Number of iterative update steps",
        )
