"""Rerun MrAP on the exact test rows this package scores.

MrAP (Bayram et al., 2021) is run from the LitEm authors' script,
``MrAP/run_mrap.py`` on the ``paper-reproduction`` branch of
dice-group/literal-embeddings::

    git clone -b paper-reproduction https://github.com/dice-group/literal-embeddings

It follows the released protocol: the per-relation offsets are estimated from
every literal, validation and test included, over the graph of all relational
triples, and the test rows are scored.

``torch_scatter`` is used by MrAP only as a 1-D sum; when it is not installed an
exact ``index_add_`` stand-in is registered.
"""

from __future__ import annotations

import dataclasses
import os
import sys
import tempfile
import types
from pathlib import Path

import pandas as pd

from .featurizer import KGIndex, load_literal_splits

#: Dataset directory name -> the key ``run_mrap.DATASETS`` uses.
MRAP_KEYS = {"DB15K": "db15k", "FB15k-237": "fb15k-237", "YAGO15k": "yago15k"}


def _install_scatter_stand_in() -> None:
    import torch

    try:
        import torch_scatter  # noqa: F401
    except ImportError:
        def scatter(src, index, dim=-1, dim_size=None, reduce="sum"):
            assert src.dim() == 1 and reduce == "sum", "stand-in covers MrAP's 1-D sum only"
            out = torch.zeros(dim_size, dtype=src.dtype, device=src.device)
            return out.index_add_(0, index, src)

        sys.modules["torch_scatter"] = types.SimpleNamespace(scatter=scatter)


def _import_run_mrap(mrap_dir: str):
    _install_scatter_stand_in()
    sys.path.insert(0, os.path.abspath(mrap_dir))
    import run_mrap

    return run_mrap


def _triples(dataset_dir: str) -> pd.DataFrame:
    return pd.concat(
        [pd.read_csv(os.path.join(dataset_dir, f), sep="\t", header=None, dtype=str)
         for f in ("train.txt", "valid.txt", "test.txt")],
        ignore_index=True,
    )


def run_mrap(mrap_dir: str, dataset_dir: str, device: str = "cpu") -> pd.DataFrame:
    """Per-attribute test MAE and MASE of MrAP on the shipped splits."""
    import torch

    run_mrap_module = _import_run_mrap(mrap_dir)
    name = os.path.basename(os.path.normpath(dataset_dir))
    split = load_literal_splits(
        os.path.join(dataset_dir, "literals"), kg_index=KGIndex.from_dataset_dir(dataset_dir)
    )
    config = dataclasses.replace(
        run_mrap_module.DATASETS[MRAP_KEYS[name]], kg_dir=name, triples_file="triples.txt"
    )
    with tempfile.TemporaryDirectory() as root:
        literal_dir = os.path.join(root, "KGs", name, "literals")
        os.makedirs(literal_dir)
        for part in ("train", "valid", "test"):
            getattr(split, part)[["head", "attribute", "value"]].to_csv(
                os.path.join(literal_dir, f"{part}.txt"), sep="\t", header=False, index=False
            )
        _triples(dataset_dir).to_csv(
            os.path.join(root, "KGs", name, "triples.txt"), sep="\t", header=False, index=False
        )
        scores, _ = run_mrap_module.run_dataset(Path(root), config, torch.device(device))

    # MASE against the training median on the same test rows, as for the backbones.
    medians = split.train.groupby("attribute")["value"].median()
    test = split.test
    baseline = (test["value"] - test["attribute"].map(medians)).abs().groupby(test["attribute"]).mean()
    scores = scores.rename(columns={"relation": "attribute"})
    scores["n"] = scores["attribute"].map(test.groupby("attribute").size())
    scores["MASE"] = scores["MAE"] / scores["attribute"].map(baseline)
    return scores[["attribute", "n", "MAE", "MASE"]].assign(
        dataset=name, seed=1, backbone="mrap", split="test",
    )
