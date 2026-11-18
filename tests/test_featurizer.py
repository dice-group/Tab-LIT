"""Shape and leakage smoke checks for the KG -> tabular conversion."""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.featurizer import (  # noqa: E402
    CONVERSIONS,
    AssociationFeaturiser,
    CompositeFeaturiser,
    FeaturisationContext,
    KGIndex,
    OwnLiteralFeaturiser,
    LLMAnswerFeaturiser,
    StructuralFeaturiser,
    build_featuriser,
    load_literal_splits,
)


def _toy_triples():
    # A small ring plus a couple of chords, large enough for a giant
    # component and a handful of relations.
    return [
        ("a", "r1", "b"), ("b", "r1", "c"), ("c", "r1", "d"), ("d", "r1", "a"),
        ("a", "r2", "c"), ("b", "r2", "d"), ("e", "r1", "a"), ("f", "r2", "e"),
    ]


@pytest.fixture
def kg() -> KGIndex:
    return KGIndex(_toy_triples())


@pytest.fixture
def literals() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "head": ["a", "b", "c", "d", "a", "b"],
            "attribute": ["x", "x", "x", "x", "y", "y"],
            "value": [1.0, 2.0, 3.0, 4.0, 10.0, 20.0],
        }
    )


class TestKGIndex:
    def test_shapes(self, kg: KGIndex):
        assert kg.num_entities == 6
        assert kg.num_relations == 2
        assert kg.out_csr.shape == (6, 6)
        assert kg.sym_csr.shape == (6, 6)

    def test_self_loops_stripped(self):
        kg = KGIndex([("a", "r", "a"), ("a", "r", "b")])
        assert kg.out_csr[kg.entity_to_idx["a"], kg.entity_to_idx["a"]] == 0

    def test_degrees_match_triples(self, kg: KGIndex):
        degrees = kg.degrees()
        # 'a' appears as head in (a,r1,b) and (a,r2,c): out-degree 2.
        assert degrees[kg.entity_to_idx["a"], 0] == 2

    def test_laplacian_eigenvectors_deterministic(self, kg: KGIndex):
        first = kg.laplacian_eigenvectors(k=3)
        second = kg.laplacian_eigenvectors(k=3)
        np.testing.assert_array_equal(first, second)


class TestFeaturisers:
    def test_own_literal_masks_nothing_by_default(self, kg: KGIndex, literals: pd.DataFrame):
        ctx = FeaturisationContext(kg_index=kg, literals=literals)
        block = OwnLiteralFeaturiser().fit_transform(ctx)
        assert block.shape == (kg.num_entities, 2)  # attributes x, y
        assert block[kg.entity_to_idx["a"], 0] == pytest.approx(1.0)
        assert np.isnan(block[kg.entity_to_idx["e"], 0])  # no literal for 'e'

    def test_association_carries_no_literals(self, kg: KGIndex, literals: pd.DataFrame):
        ctx = FeaturisationContext(kg_index=kg, literals=literals)
        block = AssociationFeaturiser(dim=4, seed=1).fit_transform(ctx)
        assert block.shape[0] == kg.num_entities
        assert block.shape[1] > 0

    def test_structural_width(self, kg: KGIndex, literals: pd.DataFrame):
        ctx = FeaturisationContext(kg_index=kg, literals=literals)
        block = StructuralFeaturiser(num_eigenvectors=3).fit_transform(ctx)
        assert block.shape == (kg.num_entities, 3 + 1 + 3)  # degrees + pagerank + eigenvectors

    def test_composite_pca_reduces_wide_groups(self, kg: KGIndex, literals: pd.DataFrame):
        ctx = FeaturisationContext(kg_index=kg, literals=literals)
        composite = CompositeFeaturiser(
            [AssociationFeaturiser(dim=4, seed=1), StructuralFeaturiser(num_eigenvectors=3)],
            max_group_dim=2,
        )
        block = composite.fit_transform(ctx)
        # association is pre-reduced (its own SVD), so it must pass through
        # untouched even though it exceeds max_group_dim=2.
        assert composite.group_widths_["association"] > 2
        assert "association" not in composite.reduced_groups_
        # structural is not pre-reduced and exceeds the budget, so it must shrink.
        assert composite.reduced_groups_["structural"] == 2
        assert block.shape[0] == kg.num_entities


@pytest.fixture
def answers(tmp_path) -> str:
    path = tmp_path / "answers.csv"
    pd.DataFrame({"head": ["a", "b", "e"], "attribute": ["x", "y", "x"],
                  "answer": ["1", "19", "?"], "llm_value": [1.5, 19.0, np.nan]}).to_csv(path, index=False)
    return str(path)


class TestBuildFeaturiser:
    @pytest.mark.parametrize("conversion", sorted(CONVERSIONS))
    def test_every_conversion_builds(self, kg: KGIndex, literals: pd.DataFrame, answers, conversion):
        ctx = FeaturisationContext(kg_index=kg, literals=literals)
        feat = build_featuriser(CONVERSIONS[conversion], association_dim=4, seed=1, llm_answers=answers)
        block = feat.fit_transform(ctx)
        assert block.shape[0] == kg.num_entities
        assert block.shape[1] > 0

    def test_full_llm_is_full_plus_the_answer_column(self):
        assert CONVERSIONS["full_llm"] == (*CONVERSIONS["full"], "llm_answer")

    def test_unknown_group_raises(self):
        with pytest.raises(ValueError):
            build_featuriser(("not_a_real_group",))


class TestLLMAnswers:
    def test_one_column_per_attribute(self, kg: KGIndex, literals: pd.DataFrame, answers):
        ctx = FeaturisationContext(kg_index=kg, literals=literals)
        featuriser = LLMAnswerFeaturiser(answers)
        block = featuriser.fit_transform(ctx)
        assert featuriser.feature_names(ctx) == ["llm::x", "llm::y"]
        assert block[kg.entity_to_idx["a"], 0] == pytest.approx(1.5)
        assert block[kg.entity_to_idx["b"], 1] == pytest.approx(19.0)
        assert np.isnan(block[kg.entity_to_idx["e"], 0])  # no parsable answer
        assert np.isnan(block[kg.entity_to_idx["c"]]).all()  # never asked


class TestShippedSplits:
    def _write(self, root, rows):
        root.mkdir()
        for name, part in rows.items():
            pd.DataFrame(part).to_csv(root / f"{name}.txt", sep="\t", header=False, index=False)

    def test_files_are_used_verbatim(self, tmp_path, kg: KGIndex):
        # A pair in both train and test with different values is a source
        # contradiction and stays; the entity outside the graph is dropped.
        self._write(tmp_path / "literals", {
            "train": [("a", "x", 1.0), ("b", "x", 2.0), ("zz", "x", 9.0)],
            "valid": [("c", "x", 3.0)],
            "test": [("a", "x", 1.5), ("d", "x", 4.0)],
        })
        splits = load_literal_splits(str(tmp_path / "literals"), kg)
        assert list(splits.train["head"]) == ["a", "b"]
        assert list(splits.test["value"]) == [1.5, 4.0]

    def test_repeated_test_values_raise(self, tmp_path):
        self._write(tmp_path / "literals", {
            "train": [("a", "x", 1.0)], "valid": [("b", "x", 2.0)], "test": [("a", "x", 1.0)],
        })
        with pytest.raises(ValueError):
            load_literal_splits(str(tmp_path / "literals"))
