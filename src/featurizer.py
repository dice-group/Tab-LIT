"""Convert a train-only knowledge graph and numeric literals into entity features."""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from scipy import sparse

# ----------------------------------------------------------------------
# KGIndex: sparse adjacency over the training graph
# ----------------------------------------------------------------------


def _read_triples(file_path: str, separator: str = "\t") -> List[Tuple[str, str, str]]:
    """Read ``(head, relation, tail)`` triples from a TSV/CSV/whitespace file."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Triple file not found: {file_path}")
    frame = pd.read_csv(file_path, sep=separator, header=None)
    if frame.shape[1] < 3:
        for fallback in (r"\s+", "\t", ","):
            retry = pd.read_csv(file_path, sep=fallback, engine="python", header=None)
            if retry.shape[1] >= 3:
                frame = retry
                break
        else:
            raise ValueError(
                f"Expected at least 3 columns in {file_path}, got {frame.shape[1]}."
            )
    return list(frame.iloc[:, :3].itertuples(index=False, name=None))


def segment_reduce(
    values: np.ndarray, indptr: np.ndarray, reduction: str, num_rows: int
) -> np.ndarray:
    """Reduce ``values`` over the CSR segments described by ``indptr``.

    Rows with no entries are filled with ``NaN`` so a downstream consumer can
    distinguish "no neighbours" from "neighbours whose aggregate is zero".
    """
    dim = values.shape[1] if values.ndim == 2 else 1
    out = np.full((num_rows, dim), np.nan, dtype=np.float64)

    counts = np.diff(indptr)
    non_empty = np.flatnonzero(counts > 0)
    if non_empty.size == 0:
        return out

    starts = indptr[non_empty]
    if reduction == "sum":
        out[non_empty] = np.add.reduceat(values, starts, axis=0)
    elif reduction == "mean":
        out[non_empty] = (
            np.add.reduceat(values, starts, axis=0) / counts[non_empty][:, None]
        )
    elif reduction == "max":
        out[non_empty] = np.maximum.reduceat(values, starts, axis=0)
    elif reduction == "min":
        out[non_empty] = np.minimum.reduceat(values, starts, axis=0)
    else:
        raise ValueError(
            f"Unknown reduction '{reduction}'. Use 'mean', 'max', 'min', or 'sum'."
        )
    return out


class KGIndex:
    """Sparse, train-only view of a knowledge graph."""

    def __init__(self, triples: Sequence[Tuple[str, str, str]]):
        if len(triples) == 0:
            raise ValueError("Cannot build a KGIndex from an empty triple set.")

        entities = sorted({e for h, _, t in triples for e in (h, t)})
        relations = sorted({r for _, r, _ in triples})

        self.entity_to_idx: Dict[str, int] = {e: i for i, e in enumerate(entities)}
        self.idx_to_entity: Dict[int, str] = {i: e for e, i in self.entity_to_idx.items()}
        self.relation_to_idx: Dict[str, int] = {r: i for i, r in enumerate(relations)}
        self.idx_to_relation: Dict[int, str] = {
            i: r for r, i in self.relation_to_idx.items()
        }

        self.triples = np.array(
            [
                (self.entity_to_idx[h], self.relation_to_idx[r], self.entity_to_idx[t])
                for h, r, t in triples
            ],
            dtype=np.int32,
        )

        self.num_entities = len(self.entity_to_idx)
        self.num_relations = len(self.relation_to_idx)

        self._out_csr: Optional[sparse.csr_matrix] = None
        self._in_csr: Optional[sparse.csr_matrix] = None
        self._sym_csr: Optional[sparse.csr_matrix] = None
        self._relation_csr: Dict[Tuple[int, str], sparse.csr_matrix] = {}

    @classmethod
    def from_file(cls, file_path: str, separator: str = "\t") -> "KGIndex":
        return cls(_read_triples(file_path, separator=separator))

    @classmethod
    def from_dataset_dir(cls, dataset_dir: str, separator: str = "\t") -> "KGIndex":
        """Build an index from ``<dataset_dir>/train.txt`` only.

        Graph-derived features must never see valid/test edges.
        """
        return cls.from_file(os.path.join(dataset_dir, "train.txt"), separator=separator)

    def _build_csr(
        self, rows: np.ndarray, cols: np.ndarray, drop_self_loops: bool = True
    ) -> sparse.csr_matrix:
        """Binary adjacency from an edge list, parallel edges collapsed.

        Self-loops are dropped by default: a self-loop makes an entity its own
        neighbour, which turns any aggregate over a literal matrix into a
        direct read of the target.
        """
        data = np.ones(len(rows), dtype=np.float32)
        matrix = sparse.csr_matrix(
            (data, (rows, cols)), shape=(self.num_entities, self.num_entities)
        )
        matrix.data[:] = 1.0
        matrix.sum_duplicates()
        matrix.data[:] = 1.0
        if drop_self_loops:
            matrix.setdiag(0)
            matrix.eliminate_zeros()
        return matrix.tocsr()

    @property
    def out_csr(self) -> sparse.csr_matrix:
        if self._out_csr is None:
            self._out_csr = self._build_csr(self.triples[:, 0], self.triples[:, 2])
        return self._out_csr

    @property
    def in_csr(self) -> sparse.csr_matrix:
        if self._in_csr is None:
            self._in_csr = self._build_csr(self.triples[:, 2], self.triples[:, 0])
        return self._in_csr

    @property
    def sym_csr(self) -> sparse.csr_matrix:
        if self._sym_csr is None:
            combined = self.out_csr + self.in_csr
            combined.data[:] = 1.0
            combined.setdiag(0)
            combined.eliminate_zeros()
            self._sym_csr = combined.tocsr()
        return self._sym_csr

    def relation_csr(self, relation: int, direction: str = "out") -> sparse.csr_matrix:
        """Binary adjacency restricted to a single relation.

        ``direction='sym'`` treats the relation as undirected.
        """
        if direction not in {"out", "in", "sym"}:
            raise ValueError(f"direction must be 'out', 'in' or 'sym', got '{direction}'.")
        key = (int(relation), direction)
        cached = self._relation_csr.get(key)
        if cached is not None:
            return cached

        mask = self.triples[:, 1] == relation
        heads = self.triples[mask, 0]
        tails = self.triples[mask, 2]
        if direction == "out":
            matrix = self._build_csr(heads, tails)
        elif direction == "in":
            matrix = self._build_csr(tails, heads)
        else:
            matrix = self._build_csr(
                np.concatenate([heads, tails]), np.concatenate([tails, heads])
            )
            matrix.setdiag(0)
            matrix.eliminate_zeros()
        matrix = matrix.tocsr()
        self._relation_csr[key] = matrix
        return matrix

    def degrees(self) -> np.ndarray:
        """``(|E|, 3)`` of out-degree, in-degree, total degree."""
        out_deg = np.bincount(self.triples[:, 0], minlength=self.num_entities)
        in_deg = np.bincount(self.triples[:, 2], minlength=self.num_entities)
        return np.stack([out_deg, in_deg, out_deg + in_deg], axis=1).astype(np.float32)

    def pagerank(
        self, damping: float = 0.85, max_iter: int = 100, tol: float = 1e-6,
        directed: bool = True,
    ) -> np.ndarray:
        """PageRank via power iteration.

        Directed by default: on the symmetrised graph the stationary
        distribution is proportional to degree (measured Spearman 0.92-0.95),
        which makes the undirected version a near-duplicate of a column
        already in the feature matrix.
        """
        adjacency = self.out_csr if directed else self.sym_csr
        degree = np.asarray(adjacency.sum(axis=1)).ravel()
        dangling = degree == 0
        inv_degree = np.zeros_like(degree)
        inv_degree[~dangling] = 1.0 / degree[~dangling]

        n = self.num_entities
        rank = np.full(n, 1.0 / n, dtype=np.float64)
        teleport = (1.0 - damping) / n

        for _ in range(max_iter):
            contribution = adjacency.T @ (rank * inv_degree)
            dangling_mass = rank[dangling].sum() / n
            updated = teleport + damping * (contribution + dangling_mass)
            if np.abs(updated - rank).sum() < tol:
                rank = updated
                break
            rank = updated
        return rank.astype(np.float32)

    def connected_components(self) -> Tuple[int, np.ndarray]:
        from scipy.sparse.csgraph import connected_components as _components

        count, labels = _components(self.sym_csr, directed=False)
        return int(count), labels

    def laplacian_eigenvectors(
        self, k: int = 14, giant_component_only: bool = True,
        trivial_tolerance: float = 1e-5,
    ) -> np.ndarray:
        """First ``k`` non-trivial eigenvectors of the symmetric normalized Laplacian.

        Deterministic (fixed ARPACK start vector, ``tol=0``) and restricted to
        entities inside the graph's giant component: eigenvalue 1 of the
        normalized adjacency has multiplicity equal to the number of connected
        components, so its eigenspace is degenerate and an arbitrary basis of
        it is mostly round-off noise for entities outside the giant component.
        """
        n = self.num_entities
        result = np.zeros((n, k), dtype=np.float32)
        if k < 1:
            return result

        rows = np.arange(n)
        if giant_component_only:
            count, labels = self.connected_components()
            if count > 1:
                sizes = np.bincount(labels)
                rows = np.flatnonzero(labels == int(np.argmax(sizes)))

        adjacency = self.sym_csr[rows][:, rows]
        size = adjacency.shape[0]
        if size < 3:
            return result

        degree = np.asarray(adjacency.sum(axis=1)).ravel()
        inv_sqrt = np.zeros_like(degree)
        nonzero = degree > 0
        inv_sqrt[nonzero] = 1.0 / np.sqrt(degree[nonzero])
        scaling = sparse.diags(inv_sqrt)
        normalized = scaling @ adjacency @ scaling

        margin = 8
        num_requested = min(k + margin, size - 1)
        if num_requested < 1:
            return result

        try:
            if size <= 512:
                eigenvalues, eigenvectors = np.linalg.eigh(np.asarray(normalized.todense()))
                order = np.argsort(eigenvalues)[::-1][:num_requested]
                values, vectors = eigenvalues[order], eigenvectors[:, order]
            else:
                from scipy.sparse.linalg import eigsh

                start = np.random.default_rng(0).normal(size=size)
                values, vectors = eigsh(
                    normalized, k=num_requested, which="LA", tol=0.0, v0=start
                )
                order = np.argsort(values)[::-1]
                values, vectors = values[order], vectors[:, order]
        except Exception as error:  # pragma: no cover - ARPACK convergence
            print(f"[KGIndex] Laplacian eigendecomposition failed ({error}); using zeros.")
            return result

        non_trivial = values < 1.0 - trivial_tolerance
        vectors = vectors[:, non_trivial]

        if vectors.size:
            pivot = np.argmax(np.abs(vectors), axis=0)
            signs = np.sign(vectors[pivot, np.arange(vectors.shape[1])])
            signs[signs == 0] = 1.0
            vectors = vectors * signs

        usable = min(k, vectors.shape[1])
        if usable:
            result[np.ix_(rows, np.arange(usable))] = vectors[:, :usable]
        return result

    def __repr__(self) -> str:
        return (
            f"KGIndex(entities={self.num_entities}, relations={self.num_relations}, "
            f"triples={len(self.triples)})"
        )


# ----------------------------------------------------------------------
# Featurisers: KGIndex + literals -> per-entity feature matrix
# ----------------------------------------------------------------------


@dataclass
class FeaturisationContext:
    """Everything a featuriser may read.

    ``literals`` must be **training** literals only.
    """

    kg_index: KGIndex
    literals: Optional[pd.DataFrame] = None
    attribute_to_idx: Optional[Dict[str, int]] = field(default=None)

    def __post_init__(self):
        if self.literals is not None and self.attribute_to_idx is None:
            self.attribute_to_idx = {
                attribute: idx
                for idx, attribute in enumerate(sorted(self.literals["attribute"].unique()))
            }

    def require_literals(self, who: str) -> pd.DataFrame:
        if self.literals is None:
            raise ValueError(
                f"{who} requires literal training data, but "
                "FeaturisationContext.literals is None."
            )
        return self.literals


class EntityFeaturiser(ABC):
    """Map a knowledge graph to a per-entity feature matrix.

    Implementations return ``(kg_index.num_entities, d)`` with rows ordered by
    entity index. ``NaN`` is a legitimate output -- the backbones used here
    handle missing values natively.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier used in feature names."""

    @property
    def is_pre_reduced(self) -> bool:
        """Whether this group is already a linear compression of its own input.

        :class:`CompositeFeaturiser` PCA-reduces any group wider than its
        budget; for a group that is already a truncated SVD that would be a
        second, lossy pass over an orthogonal basis. Groups that do their own
        reduction set this and are passed through at full width.
        """
        return False

    @abstractmethod
    def fit_transform(self, ctx: FeaturisationContext) -> np.ndarray:
        """Build the feature matrix."""

    def feature_names(self, ctx: FeaturisationContext) -> List[str]:
        width = self.fit_transform(ctx).shape[1]
        return [f"{self.name}_{i}" for i in range(width)]


class StructuralFeaturiser(EntityFeaturiser):
    """Out/in/total degree, PageRank, and Laplacian eigenvectors (G2T-FM's SF)."""

    def __init__(self, num_eigenvectors: int = 14):
        self.num_eigenvectors = num_eigenvectors

    @property
    def name(self) -> str:
        return "structural"

    def fit_transform(self, ctx: FeaturisationContext) -> np.ndarray:
        kg = ctx.kg_index
        return np.hstack(
            [kg.degrees(), kg.pagerank()[:, None], kg.laplacian_eigenvectors(self.num_eigenvectors)]
        ).astype(np.float32)

    def feature_names(self, ctx: FeaturisationContext) -> List[str]:
        return (
            ["structural::out_degree", "structural::in_degree", "structural::total_degree"]
            + ["structural::pagerank"]
            + [f"structural::laplacian_{i}" for i in range(self.num_eigenvectors)]
        )


class OwnLiteralFeaturiser(EntityFeaturiser):
    """The entity's own **training** literal values, one column per attribute.

    ``NaN`` where an entity has no training value for an attribute.

    Column ``a`` must be masked to ``NaN`` when predicting attribute ``a`` --
    :class:`~src.runner.PerAttributeLiteralPredictor` does this via
    :meth:`column_index_for_attribute`.
    """

    @property
    def name(self) -> str:
        return "own_literal"

    def fit_transform(self, ctx: FeaturisationContext) -> np.ndarray:
        literals = ctx.require_literals("OwnLiteralFeaturiser")
        kg = ctx.kg_index
        attribute_to_idx = ctx.attribute_to_idx

        width = len(attribute_to_idx)
        features = np.full((kg.num_entities, width), np.nan, dtype=np.float32)
        heads = literals["head"].map(kg.entity_to_idx)
        attributes = literals["attribute"].map(attribute_to_idx)
        known = heads.notna() & attributes.notna()
        rows = heads[known].to_numpy(dtype=np.int64)
        columns = attributes[known].to_numpy(dtype=np.int64)
        values = literals.loc[known, "value"].to_numpy(dtype=np.float64)

        # An (entity, attribute) pair can repeat with conflicting values.
        # Collapse repeats to their median so this matrix -- and everything
        # built on it -- does not depend on input row order.
        if width and len(np.unique(rows * width + columns)) < len(rows):
            collapsed = (
                pd.DataFrame({"cell": rows * width + columns, "value": values})
                .groupby("cell", sort=False)["value"]
                .median()
            )
            rows, columns = np.divmod(collapsed.index.to_numpy(dtype=np.int64), width)
            values = collapsed.to_numpy(dtype=np.float64)

        features[rows, columns] = values.astype(np.float32)
        return features

    def feature_names(self, ctx: FeaturisationContext) -> List[str]:
        return [f"own_literal::{a}" for a in _ordered_attributes(ctx)]

    @staticmethod
    def column_index_for_attribute(ctx: FeaturisationContext, attribute: str) -> List[int]:
        idx = ctx.attribute_to_idx.get(attribute)
        return [] if idx is None else [idx]


def _sparse_extreme(adjacency, values: np.ndarray, mode: str) -> np.ndarray:
    """Row-wise max/min of ``values`` over each row's sparse support."""
    adjacency = adjacency.tocsr()
    gathered = values[adjacency.indices]
    result = segment_reduce(gathered, adjacency.indptr, mode, adjacency.shape[0])
    result[np.isinf(result)] = np.nan
    return result


def _ordered_attributes(ctx: FeaturisationContext) -> List[str]:
    if ctx.attribute_to_idx is None:
        return []
    return [a for a, _ in sorted(ctx.attribute_to_idx.items(), key=lambda kv: kv[1])]


class LLMAnswerFeaturiser(EntityFeaturiser):
    """A language model's answer for each (entity, attribute) cell, one column per attribute.

    Column ``llm::<a>`` holds the parsed answer the model gave for attribute
    ``a`` of the entity, asked from the entity's name alone (``NaN`` where the
    cell was not asked). The answer depends on the name only, so it is the same
    for context and query rows and reads no literal value. Which cells were
    asked does reveal which attributes an entity holds, so the predictor masks
    every ``llm::<b>`` with ``b`` other than the target.
    """

    def __init__(self, answers_path: Optional[str]):
        if answers_path is None:
            raise ValueError("The llm_answer group needs the path of an answer file.")
        self.answers_path = answers_path

    @property
    def name(self) -> str:
        return "llm_answer"

    def fit_transform(self, ctx: FeaturisationContext) -> np.ndarray:
        kg = ctx.kg_index
        column = {a: j for j, a in enumerate(_ordered_attributes(ctx))}
        table = pd.read_csv(self.answers_path).dropna(subset=["llm_value"])
        table = table[table["attribute"].isin(column) & table["head"].isin(kg.entity_to_idx)]
        out = np.full((kg.num_entities, len(column)), np.nan, dtype=np.float32)
        out[table["head"].map(kg.entity_to_idx).to_numpy(),
            table["attribute"].map(column).to_numpy()] = table["llm_value"].to_numpy(dtype=np.float32)
        return out

    def feature_names(self, ctx: FeaturisationContext) -> List[str]:
        return [f"llm::{a}" for a in _ordered_attributes(ctx)]


class NeighborLiteralFeaturiser(EntityFeaturiser):
    """Aggregated **training** literal values of an entity's 1-hop neighbours.

    For every attribute: mean/max/min/count of that attribute's values over
    an entity's neighbours. This is the knowledge-graph analogue of
    neighbourhood aggregation applied to the target variable's own modality
    rather than to structural features, and the single most load-bearing
    group in this pipeline. An entity is never its own neighbour: self-loop
    triples are stripped from every adjacency by :class:`KGIndex`.
    """

    @property
    def name(self) -> str:
        return "neighbor_literal"

    def _value_matrix(self, ctx: FeaturisationContext) -> np.ndarray:
        return OwnLiteralFeaturiser().fit_transform(ctx)

    @staticmethod
    def _aggregate(adjacency, values, present, filled):
        counts = adjacency @ present
        sums = adjacency @ filled
        with np.errstate(invalid="ignore", divide="ignore"):
            means = np.where(counts > 0, sums / counts, np.nan)
        high = np.where(np.isnan(values), -np.inf, values)
        low = np.where(np.isnan(values), np.inf, values)
        maxima = _sparse_extreme(adjacency, high, "max")
        minima = _sparse_extreme(adjacency, low, "min")
        maxima[counts == 0] = np.nan
        minima[counts == 0] = np.nan
        return means, maxima, minima, counts

    def fit_transform(self, ctx: FeaturisationContext) -> np.ndarray:
        ctx.require_literals("NeighborLiteralFeaturiser")
        kg = ctx.kg_index
        values = self._value_matrix(ctx)
        present = (~np.isnan(values)).astype(np.float32)
        filled = np.nan_to_num(values, nan=0.0)
        means, maxima, minima, counts = self._aggregate(kg.sym_csr, values, present, filled)
        return np.hstack([means, maxima, minima, counts]).astype(np.float32)

    def feature_names(self, ctx: FeaturisationContext) -> List[str]:
        attributes = _ordered_attributes(ctx)
        names: List[str] = []
        for statistic in ("mean", "max", "min", "count"):
            names.extend([f"neighbor_literal::1hop::{statistic}::{a}" for a in attributes])
        return names


class AssociationFeaturiser(EntityFeaturiser):
    """Spectral compression of *which* entities each entity links to, by relation.

    Every other literal feature group here answers "what are my neighbours'
    values?"; this answers "who are my neighbours?" and reads no literals at
    all, so it cannot leak a target. The incidence matrix is one column per
    ``(relation, neighbour, direction)`` triple, reduced by truncated SVD.
    """

    def __init__(self, dim: int = 128, seed: int = 1):
        if dim < 1:
            raise ValueError(f"dim must be >= 1, got {dim}.")
        self.dim = dim
        self.seed = seed

    @property
    def name(self) -> str:
        return "association"

    @property
    def is_pre_reduced(self) -> bool:
        return True

    @staticmethod
    def _incidence(kg):
        heads, relations, tails = kg.triples[:, 0], kg.triples[:, 1], kg.triples[:, 2]
        keep = heads != tails
        heads, relations, tails = heads[keep], relations[keep], tails[keep]
        stride = np.int64(kg.num_entities)
        span = np.int64(kg.num_relations) * stride
        out_keys = relations.astype(np.int64) * stride + tails
        in_keys = relations.astype(np.int64) * stride + heads + span
        rows = np.concatenate([heads, tails])
        _, columns = np.unique(np.concatenate([out_keys, in_keys]), return_inverse=True)
        matrix = sparse.csr_matrix(
            (np.ones(rows.size, dtype=np.float32), (rows, columns)),
            shape=(kg.num_entities, int(columns.max()) + 1 if columns.size else 1),
        )
        matrix.sum_duplicates()
        matrix.data[:] = 1.0
        return matrix

    def fit_transform(self, ctx: FeaturisationContext) -> np.ndarray:
        from sklearn.decomposition import TruncatedSVD

        kg = ctx.kg_index
        matrix = self._incidence(kg)
        budget = min(self.dim, matrix.shape[0] - 1, matrix.shape[1] - 1)
        if budget < 1:
            return np.zeros((kg.num_entities, 0), dtype=np.float32)
        svd = TruncatedSVD(n_components=budget, random_state=self.seed)
        return svd.fit_transform(matrix).astype(np.float32)

    def feature_names(self, ctx: FeaturisationContext) -> List[str]:
        return [f"association::rel{i}" for i in range(self.dim)]


def _entity_folds(num_entities: int, folds: int, seed: int) -> np.ndarray:
    """Balanced, deterministic fold id per entity."""
    return np.random.default_rng(seed).permutation(num_entities) % folds


def _cross_fit_blocks(
    values: np.ndarray, attributes: Sequence[str], blocks: List[np.ndarray],
    folds: Optional[int], seed: int, fit_channels, emit_block,
) -> List[np.ndarray]:
    """Cross-fit prior estimates for entities with observed training values."""
    if not folds or folds < 2:
        return blocks
    assignment = _entity_folds(values.shape[0], folds, seed)
    for fold in range(folds):
        held = assignment == fold
        if not held.any():
            continue
        reduced = values.copy()
        reduced[held, :] = np.nan
        channels = fit_channels(reduced)
        for index, attribute in enumerate(attributes):
            rows = held & ~np.isnan(values[:, index])
            if not rows.any():
                continue
            blocks[index][rows] = emit_block(attribute, channels)[rows]
    return blocks


def _combine_estimates(
    num_entities: int, estimates, exact_tol: Optional[float] = None,
) -> np.ndarray:
    """Combine ``(fires, prediction, MAD, count)`` channel estimates."""
    weighted, weight, total = (np.zeros(num_entities) for _ in range(3))
    best, best_mad = (np.full(num_entities, np.nan) for _ in range(2))
    exact = np.full(num_entities, np.nan) if exact_tol is not None else None

    for fires, prediction, mad, count in estimates:
        channel_weight = np.where(fires, 1.0 / (mad + 1e-3), 0.0)
        weighted += np.where(fires, prediction * channel_weight, 0.0)
        weight += channel_weight
        total += count
        first = fires & np.isnan(best)
        best[first], best_mad[first] = prediction[first], mad
        if exact is not None and mad <= exact_tol:
            first_exact = fires & np.isnan(exact)
            exact[first_exact] = prediction[first_exact]

    with np.errstate(invalid="ignore", divide="ignore"):
        prediction = np.where(weight > 0, weighted / weight, np.nan)
    columns = [prediction, best, best_mad, total]
    if exact is not None:
        columns.append(exact)
    return np.column_stack(columns).astype(np.float32)


class MrAPPriorFeaturiser(EntityFeaturiser):
    """Propagate each attribute over relation/direction channels with MAD offsets."""

    def __init__(
        self, top_k: int = 12, min_edges: int = 15,
        log_space: Union[str, Sequence[str]] = "auto", exact_tol: float = 1e-6,
        cross_fit_folds: Optional[int] = 5, cross_fit_seed: int = 0,
    ):
        if top_k < 1:
            raise ValueError(f"top_k must be >= 1, got {top_k}.")
        if min_edges < 1:
            raise ValueError(f"min_edges must be >= 1, got {min_edges}.")
        self.top_k = top_k
        self.min_edges = min_edges
        self.log_space = log_space
        self.exact_tol = exact_tol
        self.cross_fit_folds = cross_fit_folds
        self.cross_fit_seed = cross_fit_seed
        self.channels_: Dict[str, List[Tuple[float, float, int, str, int]]] = {}
        self.log_attributes_: List[str] = []

    @property
    def name(self) -> str:
        return "mrap_prior"

    def _select_log_attributes(self, values: np.ndarray, attributes: Sequence[str]) -> List[str]:
        if self.log_space != "auto":
            return list(self.log_space)
        chosen = []
        for idx, attribute in enumerate(attributes):
            column = values[:, idx]
            observed = column[~np.isnan(column)]
            if observed.size < 2 or observed.min() <= 0:
                continue
            median = float(np.median(observed))
            if median > 0 and float(observed.max()) / median > 100.0:
                chosen.append(attribute)
        return chosen

    def _fit_channels(self, kg, values: np.ndarray, attributes: Sequence[str]):
        triples = kg.triples
        channels: Dict[str, List[Tuple[float, float, int, int, str]]] = {}
        for idx, attribute in enumerate(attributes):
            column = values[:, idx]
            if attribute in self.log_attributes_:
                with np.errstate(invalid="ignore", divide="ignore"):
                    column = np.where(column > 0, np.log(column), np.nan)
            found = []
            for relation in range(kg.num_relations):
                mask = triples[:, 1] == relation
                if not mask.any():
                    continue
                heads, tails = triples[mask, 0], triples[mask, 2]
                both = ~np.isnan(column[heads]) & ~np.isnan(column[tails])
                if int(both.sum()) < self.min_edges:
                    continue
                deltas = column[heads[both]] - column[tails[both]]
                offset = float(np.median(deltas))
                mad = float(np.median(np.abs(deltas - offset)))
                support = int(both.sum())
                found.append((mad, offset, relation, "out", support))
                found.append((mad, -offset, relation, "in", support))
            found.sort(key=lambda item: (item[0], item[2], item[3]))
            channels[attribute] = found[: self.top_k]
        return channels

    def _emit_block(self, kg, values, index, attribute, channels):
        column = values[:, index]
        logged = attribute in self.log_attributes_
        if logged:
            with np.errstate(invalid="ignore", divide="ignore"):
                column = np.where(column > 0, np.log(column), np.nan)
        present = (~np.isnan(column)).astype(np.float64)
        filled = np.nan_to_num(column, nan=0.0)

        def estimates():
            for mad, offset, relation, direction, _ in channels:
                adjacency = kg.relation_csr(relation, direction=direction)
                counts = adjacency @ present
                sums = adjacency @ filled
                fires = counts > 0
                with np.errstate(invalid="ignore", divide="ignore"):
                    prediction = np.where(fires, sums / counts + offset, np.nan)
                yield fires, prediction, mad, counts

        block = _combine_estimates(kg.num_entities, estimates(), self.exact_tol)
        if logged:
            block[:, [0, 1, 4]] = np.exp(block[:, [0, 1, 4]])
        return block

    def fit_transform(self, ctx: FeaturisationContext) -> np.ndarray:
        ctx.require_literals("MrAPPriorFeaturiser")
        kg = ctx.kg_index
        attributes = _ordered_attributes(ctx)
        values = OwnLiteralFeaturiser().fit_transform(ctx).astype(np.float64)
        self.log_attributes_ = self._select_log_attributes(values, attributes)
        self.channels_ = self._fit_channels(kg, values, attributes)

        blocks = [
            self._emit_block(kg, values, index, attribute, self.channels_[attribute])
            for index, attribute in enumerate(attributes)
        ]
        indices = {attribute: index for index, attribute in enumerate(attributes)}
        blocks = _cross_fit_blocks(
            values, attributes, blocks, self.cross_fit_folds, self.cross_fit_seed,
            fit_channels=lambda reduced: self._fit_channels(kg, reduced, attributes),
            emit_block=lambda attribute, channels: self._emit_block(
                kg, values, indices[attribute], attribute, channels[attribute]
            ),
        )
        if not blocks:
            return np.zeros((kg.num_entities, 0), dtype=np.float32)
        return np.hstack(blocks).astype(np.float32)

    def feature_names(self, ctx: FeaturisationContext) -> List[str]:
        names: List[str] = []
        for attribute in _ordered_attributes(ctx):
            names.extend(
                f"mrap_prior::{statistic}::{attribute}"
                for statistic in ("pred", "best", "mad", "count", "exact")
            )
        return names


class CompositeFeaturiser(EntityFeaturiser):
    """Concatenate feature groups, PCA-reducing any group that exceeds the budget.

    Each group is reduced *independently* rather than on the concatenation,
    so preserving each group's contribution instead of letting the widest one
    dominate the principal components. Groups are standardised before PCA
    because raw scales are incommensurable. Groups left un-reduced keep their
    ``NaN`` values, which is why the small literal groups retain the
    missingness signal the backbones read natively.
    """

    def __init__(
        self, featurisers: Sequence[EntityFeaturiser], max_group_dim: Optional[int] = 64,
        random_state: int = 0,
    ):
        if not featurisers:
            raise ValueError("CompositeFeaturiser needs at least one featuriser.")
        self.featurisers = list(featurisers)
        self.max_group_dim = max_group_dim
        self.random_state = random_state
        self.group_widths_: Dict[str, int] = {}
        self.reduced_groups_: Dict[str, int] = {}
        self._feature_names: List[str] = []

    @property
    def name(self) -> str:
        return "+".join(f.name for f in self.featurisers)

    def fit_transform(self, ctx: FeaturisationContext) -> np.ndarray:
        blocks: List[np.ndarray] = []
        names: List[str] = []
        self.group_widths_ = {}
        self.reduced_groups_ = {}

        for featuriser in self.featurisers:
            block = featuriser.fit_transform(ctx)
            self.group_widths_[featuriser.name] = block.shape[1]

            if (
                self.max_group_dim is not None
                and block.shape[1] > self.max_group_dim
                and not featuriser.is_pre_reduced
            ):
                block, width = self._reduce(block, featuriser.name)
                names.extend([f"{featuriser.name}::pca_{i}" for i in range(width)])
                self.reduced_groups_[featuriser.name] = width
            else:
                group_names = featuriser.feature_names(ctx)
                if len(group_names) != block.shape[1]:
                    group_names = [f"{featuriser.name}_{i}" for i in range(block.shape[1])]
                names.extend(group_names)
            blocks.append(block)

        self._feature_names = names
        return np.hstack(blocks).astype(np.float32)

    def _reduce(self, block: np.ndarray, group: str):
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler

        finite = np.where(np.isfinite(block), block, np.nan)
        medians = np.nanmedian(finite, axis=0)
        medians = np.where(np.isfinite(medians), medians, 0.0)
        imputed = np.where(np.isfinite(block), block, medians)

        scaled = StandardScaler().fit_transform(imputed)
        budget = min(self.max_group_dim, scaled.shape[0], scaled.shape[1])
        pca = PCA(n_components=budget, random_state=self.random_state).fit(scaled)
        reduced = pca.transform(scaled)
        print(f"[CompositeFeaturiser] {group}: {block.shape[1]} -> {budget} dims (PCA)")
        return reduced.astype(np.float32), budget

    def feature_names(self, ctx: FeaturisationContext) -> List[str]:
        if not self._feature_names:
            self.fit_transform(ctx)
        return list(self._feature_names)


#: The five blocks of the conversion and the feature group that builds each.
BLOCKS: Dict[str, str] = {
    "struct": "structural",
    "assoc": "association",
    "value": "neighbor_literal",
    "offset": "mrap_prior",
    "own": "own_literal",
}

#: Named conversions. ``full`` is the conversion of the main results table;
#: ``full_llm`` adds the language model's answers (:class:`LLMAnswerFeaturiser`)
#: and the rest are the ablation arms. Group order matters: it fixes the column order, and the backbones' outputs
#: depend on column order.
CONVERSIONS: Dict[str, Sequence[str]] = {
    "full": ("structural", "association", "neighbor_literal", "mrap_prior", "own_literal"),
    "full_llm": ("structural", "association", "neighbor_literal", "mrap_prior", "own_literal",
                 "llm_answer"),
    # Ablation, adding one block at a time.
    "own": ("own_literal",),
    "struct_own": ("structural", "own_literal"),
    "struct_assoc_own": ("structural", "association", "own_literal"),
    "struct_assoc_value_own": ("structural", "association", "neighbor_literal", "own_literal"),
    # Ablation, removing one block from ``full``.
    "full_less_struct": ("association", "neighbor_literal", "mrap_prior", "own_literal"),
    "full_less_assoc": ("structural", "neighbor_literal", "mrap_prior", "own_literal"),
    "full_less_value": ("structural", "association", "mrap_prior", "own_literal"),
}


def build_featuriser(
    groups: Sequence[str],
    num_eigenvectors: int = 14,
    max_group_dim: Optional[int] = 64,
    association_dim: int = 128,
    mrap_top_k: int = 12,
    mrap_min_edges: int = 15,
    prior_cross_fit_folds: Optional[int] = 5,
    seed: int = 1,
    llm_answers: Optional[str] = None,
) -> CompositeFeaturiser:
    """Assemble a :class:`CompositeFeaturiser` from feature-group names.

    The defaults are those of every reported run: ``k = 14`` eigenvectors, as
    in G2T-FM, and a ``q = 128`` association SVD. ``llm_answers`` is the answer
    file the ``llm_answer`` group reads.
    """
    registry = {
        "structural": lambda: StructuralFeaturiser(num_eigenvectors=num_eigenvectors),
        "association": lambda: AssociationFeaturiser(
            dim=association_dim, seed=seed
        ),
        "neighbor_literal": lambda: NeighborLiteralFeaturiser(),
        "mrap_prior": lambda: MrAPPriorFeaturiser(
            top_k=mrap_top_k, min_edges=mrap_min_edges, cross_fit_folds=prior_cross_fit_folds
        ),
        "own_literal": lambda: OwnLiteralFeaturiser(),
        "llm_answer": lambda: LLMAnswerFeaturiser(llm_answers),
    }
    unknown = [g for g in groups if g not in registry]
    if unknown:
        raise ValueError(f"Unknown feature group(s) {unknown}. Available: {sorted(registry)}")
    return CompositeFeaturiser([registry[g]() for g in groups], max_group_dim=max_group_dim)


# ----------------------------------------------------------------------
# Literal splits
# ----------------------------------------------------------------------


@dataclass
class LiteralSplits:
    """Train / valid / test frames of ``(head, attribute, value)`` literals."""

    train: pd.DataFrame
    valid: pd.DataFrame
    test: pd.DataFrame


def load_literal_frame(file_path: str) -> pd.DataFrame:
    """Load a tab-separated ``entity, attribute, value`` file."""
    frame = pd.read_csv(file_path, sep="\t", header=None, index_col=False,
                        names=["head", "attribute", "value"], dtype={0: str, 1: str})
    frame["value"] = pd.to_numeric(frame["value"], errors="raise")
    return frame


def _shared_pairs(a: pd.DataFrame, b: pd.DataFrame) -> int:
    pairs = set(zip(a["head"], a["attribute"]))
    return sum(pair in pairs for pair in zip(b["head"], b["attribute"]))


def _repeated_values(a: pd.DataFrame, b: pd.DataFrame) -> int:
    seen = set(zip(a["head"], a["attribute"], np.round(a["value"].to_numpy(float), 9)))
    return sum(row in seen for row in zip(b["head"], b["attribute"], np.round(b["value"].to_numpy(float), 9)))


def load_literal_splits(literal_dir: str, kg_index: Optional[KGIndex] = None) -> LiteralSplits:
    """The dataset's shipped ``train/valid/test.txt`` literal splits, used verbatim.

    These are the rows published baselines were scored on. Rows whose entity is
    not in the training graph are dropped. The shipped files share some
    (entity, attribute) pairs across splits, almost all with different values
    (contradictions in the source data); the counts are printed, and loading
    fails only if more than 1% of test rows repeat a training value exactly.
    """
    splits = LiteralSplits(*(load_literal_frame(os.path.join(literal_dir, f"{name}.txt"))
                             for name in ("train", "valid", "test")))
    if kg_index is not None:
        for name in ("train", "valid", "test"):
            frame = getattr(splits, name)
            keep = frame["head"].isin(kg_index.entity_to_idx)
            if not keep.all():
                print(f"[literal splits] {name}: dropped {int((~keep).sum())}/{len(frame)} literals "
                      "whose entity is absent from the training graph.")
            setattr(splits, name, frame[keep].reset_index(drop=True))

    repeated = _repeated_values(splits.train, splits.test) + _repeated_values(splits.valid, splits.test)
    print(f"[literal splits] shipped splits: {_shared_pairs(splits.train, splits.test)} "
          f"(entity, attribute) pairs shared by train and test, {repeated} test rows "
          "repeating a known value.")
    if repeated > 0.01 * max(len(splits.test), 1):
        raise ValueError(f"{repeated} test rows repeat a training value; too many to score against.")
    return splits
