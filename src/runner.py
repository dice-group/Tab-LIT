"""Backbone wrappers and the experiment loop.

Four frozen tabular foundation models -- TabPFN-3.5, TabICLv2, Kumo-Tabular and
Nori-1.5 -- sit behind one shared fit/predict loop
(:class:`PerAttributeLiteralPredictor`), next to a gradient-boosting backbone
that needs no external weights. All of them read the same feature table built
by :mod:`src.featurizer`; only the backbone differs.

The foundation models are not trained on this data. "Fitting" one hands an
attribute's training rows to it as in-context examples, and "predicting" is one
forward pass over that context and the query rows. Every backbone is decoded at
its predictive **median**, the minimiser of the absolute error that every
reported metric is built on.

:func:`run_experiment` runs the reported protocol: one fixed conversion, applied to
every attribute, scored on test (and on validation, for reference), with no
selection over configurations.
"""

from __future__ import annotations

import gc
import os
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .evaluation import evaluate_predictor
from .featurizer import (
    CONVERSIONS,
    CompositeFeaturiser,
    EntityFeaturiser,
    FeaturisationContext,
    KGIndex,
    build_featuriser,
    load_literal_splits,
)

#: The language model's name-only answers, one file per dataset (see :mod:`src.llm`).
LLM_ANSWERS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "llm",
                           "{dataset}_answers.csv")


def _default_device() -> str:
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:  # pragma: no cover
        return "cpu"


def _free_gpu() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:  # pragma: no cover
        pass


# ----------------------------------------------------------------------
# The shared per-attribute loop
# ----------------------------------------------------------------------


class PerAttributeLiteralPredictor:
    """One in-context regression per attribute over a shared entity table.

    Subclasses supply :meth:`_make_model` (an object with ``fit(X, y)``) and
    :meth:`_point_predict` (its median). Everything else -- the table, the
    masking of the target's own column, the fallback to the training mean for
    attributes too small to fit -- lives here, so backbones differ in nothing
    but the model.
    """

    backbone_name = "per-attribute"

    def __init__(
        self, featuriser: EntityFeaturiser, max_train_samples: Optional[int] = 10000,
        min_rows_per_attribute: int = 10, random_seed: int = 0, verbose: bool = True,
        n_threads: Optional[int] = None,
    ):
        self.featuriser = featuriser
        self.max_train_samples = max_train_samples
        self.min_rows_per_attribute = min_rows_per_attribute
        self.random_seed = random_seed
        self.verbose = verbose
        self.n_threads = n_threads
        #: Attributes to fit; ``None`` fits every attribute with training rows.
        self.fit_attributes: Optional[frozenset] = None

        self.entity_to_idx: Dict[str, int] = {}
        self.train_means_: Dict[str, float] = {}
        self._global_mean = 0.0
        self.features_: Optional[np.ndarray] = None
        self.feature_names_: List[str] = []
        self.regressors_: Dict[str, object] = {}
        self.fallback_attributes_: List[str] = []
        self.usable_columns_: Dict[str, np.ndarray] = {}
        self._own_columns: Dict[str, List[int]] = {}

    # -- backbone hooks --------------------------------------------------

    def _make_model(self, attribute: str, num_rows: int):
        raise NotImplementedError

    def _point_predict(self, model, design: np.ndarray) -> np.ndarray:
        return np.asarray(model.predict(design), dtype=np.float64)

    # -- fitting ---------------------------------------------------------

    def _thread_limit(self):
        from contextlib import nullcontext

        if self.n_threads is None:
            return nullcontext()
        from threadpoolctl import threadpool_limits

        return threadpool_limits(limits=self.n_threads)

    def fit(self, kg_index: KGIndex, train_literals: pd.DataFrame) -> "PerAttributeLiteralPredictor":
        """Build the table from the training graph and fit one model per attribute.

        Only ``train_literals`` reach the featuriser; validation and test values
        are never read.
        """
        self.entity_to_idx = dict(kg_index.entity_to_idx)
        self.train_means_ = train_literals.groupby("attribute")["value"].mean().to_dict()
        self._global_mean = float(train_literals["value"].mean())
        self.regressors_, self.fallback_attributes_, self.usable_columns_ = {}, [], {}

        ctx = FeaturisationContext(kg_index=kg_index, literals=train_literals)
        self.features_ = self.featuriser.fit_transform(ctx)
        self.feature_names_ = self.featuriser.feature_names(ctx)
        self._resolve_own_columns()
        if self.verbose:
            print(
                f"[{self.backbone_name}] table {self.features_.shape} "
                f"({np.isnan(self.features_).mean():.1%} missing)"
            )

        rng = np.random.default_rng(self.random_seed)
        for attribute in sorted(self.train_means_):
            if self.fit_attributes is not None and attribute not in self.fit_attributes:
                continue
            rows = train_literals[train_literals["attribute"] == attribute]
            entity_rows, targets = self._rows_and_targets(rows)
            if len(entity_rows) < self.min_rows_per_attribute:
                self.fallback_attributes_.append(attribute)
                continue
            if self.max_train_samples is not None and len(entity_rows) > self.max_train_samples:
                keep = rng.choice(len(entity_rows), self.max_train_samples, replace=False)
                entity_rows, targets = entity_rows[keep], targets[keep]

            design = self._design_matrix(entity_rows, attribute)
            usable = self._usable_columns(design)
            if not usable.any():
                self.fallback_attributes_.append(attribute)
                continue
            self.usable_columns_[attribute] = usable

            model = self._make_model(attribute, len(targets))
            try:
                with self._thread_limit():
                    model.fit(design[:, usable], targets)
            except Exception as error:
                self.fallback_attributes_.append(attribute)
                self.usable_columns_.pop(attribute, None)
                print(
                    f"[{self.backbone_name}] {attribute}: fit failed "
                    f"({type(error).__name__}: {error}); using the training mean."
                )
                continue
            self.regressors_[attribute] = model
            if self.verbose:
                print(
                    f"[{self.backbone_name}] {attribute}: {len(targets)} context rows x "
                    f"{int(usable.sum())} columns"
                )
        return self

    @staticmethod
    def _usable_columns(design: np.ndarray) -> np.ndarray:
        """Columns that are observed somewhere and not constant in this context."""
        observed = ~np.isnan(design)
        has_values = observed.any(axis=0)
        spread = np.zeros(design.shape[1], dtype=np.float64)
        if has_values.any():
            present = design[:, has_values]
            spread[has_values] = np.nanmax(present, axis=0) - np.nanmin(present, axis=0)
        return has_values & (np.nan_to_num(spread, nan=0.0) > 0)

    def _rows_and_targets(self, rows: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        mapped = rows["head"].map(self.entity_to_idx)
        known = mapped.notna()
        return (
            mapped[known].to_numpy(dtype=np.int64),
            rows.loc[known, "value"].to_numpy(dtype=np.float64),
        )

    def _resolve_own_columns(self) -> None:
        """Find the columns masked when predicting each attribute ``a``.

        ``own_literal::a`` is the target itself. Every ``llm::b`` with ``b != a``
        is masked too: the answer values depend only on the entity's name, but
        which cells hold an answer reveals which attributes the entity has.
        """
        reduced = getattr(self.featuriser, "reduced_groups_", {})
        for group in ("own_literal", "llm_answer"):
            if isinstance(self.featuriser, CompositeFeaturiser) and group in reduced:
                raise ValueError(
                    f"The {group} group was PCA-reduced, which makes per-target masking "
                    "impossible and would leak the target into its own row."
                )
        self._own_columns = {}
        llm_columns: Dict[str, int] = {}
        for column, name in enumerate(self.feature_names_):
            if name.startswith("own_literal::"):
                self._own_columns.setdefault(name.split("::", 1)[1], []).append(column)
            elif name.startswith("llm::"):
                llm_columns[name.split("::", 1)[1]] = column
        for attribute in llm_columns:
            self._own_columns.setdefault(attribute, []).extend(
                column for other, column in llm_columns.items() if other != attribute
            )

    def _design_matrix(self, entity_rows: np.ndarray, attribute: str) -> np.ndarray:
        design = self.features_[entity_rows].astype(np.float64, copy=True)
        for column in self._own_columns.get(attribute, []):
            design[:, column] = np.nan
        return design

    # -- prediction ------------------------------------------------------

    def predict_literals(self, entity, attribute, **_) -> np.ndarray:
        """Predict values for zipped ``(entity, attribute)`` lists."""
        entities, attributes = list(entity), list(attribute)
        known = np.array([e in self.entity_to_idx for e in entities])
        rows = np.array([self.entity_to_idx.get(e, 0) for e in entities], dtype=np.int64)
        predictions = np.empty(len(entities), dtype=np.float64)

        for target in set(attributes):
            selection = np.array([i for i, a in enumerate(attributes) if a == target])
            fallback = self.train_means_.get(target, self._global_mean)
            usable, unknown = selection[known[selection]], selection[~known[selection]]
            predictions[unknown] = fallback
            regressor = self.regressors_.get(target)
            if regressor is None or len(usable) == 0:
                predictions[usable] = fallback
                continue
            design = self._design_matrix(rows[usable], target)[:, self.usable_columns_[target]]
            with self._thread_limit():
                predictions[usable] = self._point_predict(regressor, design)
        return predictions


# ----------------------------------------------------------------------
# Backbones
# ----------------------------------------------------------------------


class TabPFNLiteralPredictor(PerAttributeLiteralPredictor):
    """TabPFN-3.5 (``tabpfn`` 9.x). Needs ``TABPFN_TOKEN`` for unattended runs."""

    backbone_name = "TabPFN"

    def __init__(self, featuriser, device: str = "cpu", n_estimators: int = 4, **kwargs):
        super().__init__(featuriser, **kwargs)
        self.device = device
        self.n_estimators = n_estimators

    def _make_model(self, attribute, num_rows):
        from tabpfn import TabPFNRegressor

        return TabPFNRegressor(
            device=self.device, n_estimators=self.n_estimators, ignore_pretraining_limits=True,
            random_state=self.random_seed, n_preprocessing_jobs=1,
        )

    def _point_predict(self, model, design):
        return np.asarray(model.predict(design, output_type="median"), dtype=np.float64)


class TabICLLiteralPredictor(PerAttributeLiteralPredictor):
    """TabICLv2 (``tabicl`` 2.2). Weights download from Hugging Face, no token."""

    backbone_name = "TabICL"

    def __init__(self, featuriser, device: Optional[str] = None, n_estimators: int = 8, **kwargs):
        super().__init__(featuriser, **kwargs)
        self.device = device
        self.n_estimators = n_estimators

    def _make_model(self, attribute, num_rows):
        from tabicl import TabICLRegressor

        return TabICLRegressor(
            device=self.device, n_estimators=self.n_estimators, random_state=self.random_seed
        )

    def _point_predict(self, model, design):
        return np.asarray(model.predict(design, output_type="median"), dtype=np.float64)


#: One loaded Kumo-Tabular per (size, device). The large checkpoint takes about
#: 14 s to load and would otherwise sit on the GPU once per attribute.
_KUMO_MODELS: Dict[tuple, object] = {}


class _KumoAdapter:
    """``fit``/``predict`` over Kumo-Tabular's tensor interface.

    The shared model holds one fitted context at a time, so the training rows
    are kept here and the context is refitted at predict time. Kumo-Tabular
    regresses onto 999 quantiles; the median is quantile 500.
    """

    def __init__(self, size: str, device: str):
        self.size, self.device = size, device

    def _model(self):
        import torch
        from sdm.models.kumo.tabular.model import KumoTabular

        key = (self.size, self.device)
        if key not in _KUMO_MODELS:
            _KUMO_MODELS[key] = KumoTabular(
                task="regression", size=self.size, pretrained=True, device=torch.device(self.device),
            ).eval()
        return _KUMO_MODELS[key]

    def fit(self, X, y):
        self._x = np.asarray(X, dtype=np.float32)
        self._y = np.asarray(y, dtype=np.float32).reshape(-1, 1)
        return self

    def median(self, X) -> np.ndarray:
        import torch

        model = self._model()
        as_tensor = lambda a: torch.tensor(a, dtype=torch.float32, device=self.device)  # noqa: E731
        model.fit(as_tensor(self._x), as_tensor(self._y))
        bank = model.predict(as_tensor(np.asarray(X, dtype=np.float32))).numerical
        return bank.float().cpu().numpy().reshape(len(X), -1)[:, 499].astype(np.float64)


class KumoTabularLiteralPredictor(PerAttributeLiteralPredictor):
    """NVIDIA Kumo-Tabular, large (``structured-data-models`` 0.2.1).

    Not deterministic across runs on the GPU, so repeated runs differ slightly.
    """

    backbone_name = "KumoTabular"

    def __init__(self, featuriser, device: Optional[str] = None, size: str = "large", **kwargs):
        super().__init__(featuriser, **kwargs)
        self.device = str(device or _default_device())
        self.size = size

    def _make_model(self, attribute, num_rows):
        return _KumoAdapter(self.size, self.device)

    def _point_predict(self, model, design):
        return model.median(design)


class _NoriAdapter:
    """Keeps the training rows; builds a ``NoriRegressor`` only to predict.

    A fitted ``NoriRegressor`` keeps its network, so one per attribute would
    hold one model copy per attribute on the GPU. Nori-1.5 also holds the whole
    context on the device, so the largest YAGO15k attributes (about 6,500 rows
    by 190 columns) exceed 24 GB; those are refitted on the CPU with the same
    weights and context, which is slow but gives the same model.
    """

    def __init__(self, model: str, device: str):
        self.model, self.device = model, device

    def fit(self, X, y):
        self._x = np.asarray(X, dtype=np.float64)
        self._y = np.asarray(y, dtype=np.float64)
        return self

    def median(self, X) -> np.ndarray:
        import torch

        try:
            return self._median_on(self.device, X)
        except torch.OutOfMemoryError:
            if self.device == "cpu":
                raise
            print(f"[Nori] {self._x.shape[0]} x {self._x.shape[1]} context exceeds GPU memory; using the CPU")
            _free_gpu()
            return self._median_on("cpu", X)

    def _median_on(self, device: str, X) -> np.ndarray:
        from synthefy_nori import NoriRegressor

        regressor = NoriRegressor(model=self.model, device=device)
        try:
            regressor.fit(self._x, self._y)
            out = regressor.predict(np.asarray(X, dtype=np.float64), output_type="median")
            return np.asarray(out, dtype=np.float64).ravel()
        finally:
            del regressor
            _free_gpu()


class NoriLiteralPredictor(PerAttributeLiteralPredictor):
    """Synthefy Nori-1.5 (``synthefy-nori`` 0.22.0), the local estimator only."""

    backbone_name = "Nori"

    def __init__(self, featuriser, device: Optional[str] = None, model: str = "nori-1-5", **kwargs):
        super().__init__(featuriser, **kwargs)
        self.device = str(device or _default_device())
        self.model = model

    def _make_model(self, attribute, num_rows):
        return _NoriAdapter(self.model, self.device)

    def _point_predict(self, model, design):
        return model.median(design)


class GradientBoostingLiteralPredictor(PerAttributeLiteralPredictor):
    """``HistGradientBoostingRegressor`` with absolute-error loss.

    Needs no weights and no GPU, so it is the package's smoke test. No reported
    table includes it.
    """

    backbone_name = "GBT"

    def _make_model(self, attribute, num_rows):
        from sklearn.ensemble import HistGradientBoostingRegressor

        return HistGradientBoostingRegressor(
            loss="absolute_error", max_iter=200, learning_rate=0.1,
            random_state=self.random_seed, early_stopping=num_rows >= 200,
        )


BACKBONES = {
    "tabpfn": TabPFNLiteralPredictor,
    "tabicl": TabICLLiteralPredictor,
    "kumo": KumoTabularLiteralPredictor,
    "nori": NoriLiteralPredictor,
    "gbt": GradientBoostingLiteralPredictor,
}

#: Column labels, in table order.
BACKBONE_LABELS = {
    "tabpfn": "TabPFN-3.5", "tabicl": "TabICLv2", "kumo": "Kumo-Tabular", "nori": "Nori-1.5",
    "gbt": "GBT",
}


def build_predictor(
    backbone: str, featuriser: EntityFeaturiser, seed: int = 1, n_threads: Optional[int] = 4,
    device: Optional[str] = None, max_train_samples: Optional[int] = 10000, verbose: bool = True,
) -> PerAttributeLiteralPredictor:
    if backbone not in BACKBONES:
        raise ValueError(f"Unknown backbone '{backbone}'. Available: {sorted(BACKBONES)}")
    kwargs = dict(
        max_train_samples=max_train_samples, random_seed=seed, n_threads=n_threads,
        verbose=verbose,
    )
    if backbone != "gbt":
        kwargs["device"] = device or _default_device()
    return BACKBONES[backbone](featuriser, **kwargs)


# ----------------------------------------------------------------------
# The experiment loop
# ----------------------------------------------------------------------


def _package_versions() -> Dict[str, str]:
    from importlib import metadata

    versions = {}
    for name in ("tabpfn", "tabicl", "structured-data-models", "synthefy-nori", "torch",
                 "scikit-learn", "numpy", "pandas", "scipy"):
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = "not installed"
    return versions


def run_experiment(
    dataset_dir: str, backbone: str, conversion: str, seeds: Sequence[int] = (1, 2, 3),
    n_threads: Optional[int] = 4, device: Optional[str] = None,
) -> Tuple[pd.DataFrame, dict]:
    """Fit one conversion under one backbone for every seed; score test and valid.

    The dataset's shipped literal splits are the same for every seed, so the
    seeds vary only the backbone, the context subsample and the cross-fitting.

    Returns the per-attribute frame and a manifest of what produced it.
    """
    groups = CONVERSIONS[conversion]
    dataset = os.path.basename(os.path.normpath(dataset_dir))
    frames, timings = [], {}
    for seed in seeds:
        start = time.time()
        kg_index = KGIndex.from_dataset_dir(dataset_dir)
        split = load_literal_splits(os.path.join(dataset_dir, "literals"), kg_index=kg_index)
        featuriser = build_featuriser(groups, seed=seed, llm_answers=LLM_ANSWERS.format(dataset=dataset))
        predictor = build_predictor(backbone, featuriser, seed=seed, n_threads=n_threads, device=device)
        predictor.fit_attributes = frozenset(split.valid["attribute"]) | frozenset(split.test["attribute"])
        predictor.fit(kg_index, split.train)
        for name, frame in (("test", split.test), ("valid", split.valid)):
            if frame is None or frame.empty:
                continue
            detail = evaluate_predictor(predictor, frame, split.train)["per_attribute"]
            frames.append(detail.assign(
                dataset=dataset, conversion=conversion, backbone=backbone, seed=seed, split=name,
            ))
        timings[seed] = round(time.time() - start, 1)
        print(f"[{backbone}/{conversion}] seed {seed} done in {timings[seed]:.0f}s")
        del predictor
        _free_gpu()

    manifest = {
        "dataset": dataset, "backbone": backbone, "conversion": conversion,
        "groups": list(groups), "seeds": list(seeds),
        "seconds_per_seed": timings, "package_versions": _package_versions(),
    }
    return pd.concat(frames, ignore_index=True), manifest
