"""Metric formulas, target masking and the result tables."""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.evaluation import (  # noqa: E402
    LITEM,
    llm_direct_path,
    llm_table,
    mrap_path,
    per_attribute_metrics,
    result_path,
    shipped_table,
)
from src.featurizer import KGIndex, build_featuriser  # noqa: E402
from src.runner import build_predictor  # noqa: E402


class TestPerAttributeMetrics:
    def test_mae_matches_hand_computation(self):
        frame = pd.DataFrame(
            {"attribute": ["x", "x", "x"], "value": [1.0, 2.0, 3.0], "predictions": [1.0, 2.0, 5.0]}
        )
        assert per_attribute_metrics(frame).loc[0, "MAE"] == pytest.approx(2 / 3)

    def test_mase_is_one_for_the_training_median(self):
        train = pd.DataFrame({"attribute": ["x"] * 3, "value": [1.0, 2.0, 9.0]})
        frame = pd.DataFrame({"attribute": ["x"] * 3, "value": [0.0, 2.0, 7.0], "predictions": [2.0] * 3})
        assert per_attribute_metrics(frame, train).loc[0, "MASE"] == pytest.approx(1.0)


class TestTargetMasking:
    def test_own_column_of_the_target_is_masked(self):
        triples = [("a", "r", "b"), ("b", "r", "c"), ("c", "r", "a")] + [
            (f"e{i}", "r", "a") for i in range(30)
        ]
        kg = KGIndex(triples)
        entities = ["a", "b", "c"] + [f"e{i}" for i in range(30)]
        literals = pd.DataFrame(
            [(e, "x", float(i)) for i, e in enumerate(entities)]
            + [(e, "y", float(i) + 0.5) for i, e in enumerate(entities)],
            columns=["head", "attribute", "value"],
        )
        predictor = build_predictor("gbt", build_featuriser(("own_literal",)), verbose=False)
        predictor.fit(kg, literals)
        x_column = predictor.feature_names_.index("own_literal::x")
        design = predictor._design_matrix(np.arange(3), "x")
        assert np.isnan(design[:, x_column]).all()
        assert not np.isnan(design[:, predictor.feature_names_.index("own_literal::y")]).all()

    def test_only_the_targets_answer_column_is_visible(self, tmp_path):
        kg = KGIndex([(f"e{i}", "r", f"e{i + 1}") for i in range(40)])
        entities = [f"e{i}" for i in range(40)]
        literals = pd.DataFrame(
            [(e, "x", float(i)) for i, e in enumerate(entities)]
            + [(e, "y", float(i) + 0.5) for i, e in enumerate(entities)],
            columns=["head", "attribute", "value"],
        )
        answers = tmp_path / "answers.csv"
        pd.DataFrame([(e, a, 1.0) for e in entities for a in ("x", "y")],
                     columns=["head", "attribute", "llm_value"]).to_csv(answers, index=False)
        featuriser = build_featuriser(("own_literal", "llm_answer"), llm_answers=str(answers))
        predictor = build_predictor("gbt", featuriser, verbose=False)
        predictor.fit(kg, literals)
        design = predictor._design_matrix(np.arange(3), "x")
        names = predictor.feature_names_
        assert not np.isnan(design[:, names.index("llm::x")]).any()
        assert np.isnan(design[:, names.index("llm::y")]).all()
        assert np.isnan(design[:, names.index("own_literal::x")]).all()


def _write(tmp_path, dataset, backbone, conversion, rows):
    frame = pd.DataFrame(rows).assign(dataset=dataset, split="test", backbone=backbone,
                                      conversion=conversion)
    frame.to_csv(result_path(str(tmp_path), dataset, backbone, conversion), index=False)


def _mrap(tmp_path, dataset, rows):
    pd.DataFrame(rows).assign(dataset=dataset, split="test", backbone="mrap", seed=1).to_csv(
        mrap_path(str(tmp_path), dataset), index=False)


class TestTables:
    def test_shipped_table_counts_wins_against_litem_and_mrap(self, tmp_path):
        litem = LITEM["DB15K"]["birthDate"]
        _mrap(tmp_path, "DB15K", [{"attribute": "birthDate", "n": 502, "MAE": litem + 5}])
        _write(tmp_path, "DB15K", "tabpfn", "full",
               [{"attribute": "birthDate", "n": 502, "MAE": litem - 1, "seed": s} for s in (1, 2, 3)])
        # Other conversions must not leak into the table.
        _write(tmp_path, "DB15K", "tabpfn", "full_llm",
               [{"attribute": "birthDate", "n": 502, "MAE": 0.0, "seed": s} for s in (1, 2, 3)])
        text = shipped_table(str(tmp_path), ["tabpfn"], {"tabpfn": "TabPFN-3.5"})
        assert f"**{litem - 1:.3f}**" in text
        assert "TabPFN-3.5 1/1" in text

    def test_incomplete_seeds_are_not_reported(self, tmp_path):
        _mrap(tmp_path, "DB15K", [{"attribute": "birthDate", "n": 502, "MAE": 20.0}])
        _write(tmp_path, "DB15K", "kumo", "full", [{"attribute": "birthDate", "n": 502, "MAE": 9.0, "seed": 1}])
        _write(tmp_path, "DB15K", "tabpfn", "full",
               [{"attribute": "birthDate", "n": 502, "MAE": 10.0, "seed": s} for s in (1, 2, 3)])
        text = shipped_table(str(tmp_path), ["tabpfn", "kumo"], {"tabpfn": "TabPFN-3.5", "kumo": "Kumo-Tabular"})
        row = [line for line in text.splitlines() if line.startswith("| birthDate")][0]
        assert row.endswith("| **10.000** | -- |")

    def test_mixed_test_rows_raise(self, tmp_path):
        _mrap(tmp_path, "DB15K", [{"attribute": "birthDate", "n": 500, "MAE": 20.0}])
        _write(tmp_path, "DB15K", "tabpfn", "full",
               [{"attribute": "birthDate", "n": 502, "MAE": 10.0, "seed": s} for s in (1, 2, 3)])
        with pytest.raises(ValueError):
            shipped_table(str(tmp_path), ["tabpfn"], {"tabpfn": "TabPFN-3.5"})

    def test_llm_table_medians_and_counts(self, tmp_path):
        for conversion, mase in (("full", (0.5, 0.9)), ("full_llm", (0.3, 1.2))):
            _write(tmp_path, "DB15K", "gbt", conversion,
                   [{"attribute": a, "n": 100, "MASE": m, "seed": s}
                    for a, m in zip(("birthDate", "deathDate"), mase) for s in (1, 2, 3)])
        pd.DataFrame([{"arm": arm, "attribute": a, "n": 100, "MASE": 0.8}
                      for arm in ("name", "graph", "graph_values") for a in ("birthDate", "deathDate")]
                     ).assign(dataset="DB15K").to_csv(llm_direct_path(str(tmp_path), "DB15K"), index=False)
        text = llm_table(str(tmp_path), ["gbt"], {"gbt": "GBT"})
        assert "| conversion | GBT | 0.700 | 2 | 2 |" in text
        assert "| + answer column | GBT | 0.750 | 1 | 2 |" in text
        assert "GBT 1/2" in text
