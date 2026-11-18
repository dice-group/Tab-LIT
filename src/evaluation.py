"""Metrics, LitEm's own runs, and the result tables.

Errors are never averaged over attributes: they carry different units (years,
degrees, meters), so a mean over them is dominated by the widest-spread
attribute. Comparisons across attributes are counts of attributes.
"""

from __future__ import annotations

import glob
import os
from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------


def per_attribute_metrics(
    frame: pd.DataFrame, train_literals: Optional[pd.DataFrame] = None,
    value_column: str = "value", prediction_column: str = "predictions",
) -> pd.DataFrame:
    """Per-attribute MAE, RMSE, normalized MAE, Spearman rho and R^2.

    ``NMAE`` divides by the training standard deviation, which one outlier
    inflates and every subsequent NMAE it touches deflates; ``RNMAE`` divides
    by the training median absolute deviation instead and does not have that
    failure mode. ``MASE`` divides by the per-attribute median baseline's own
    MAE on the same rows, so ``1.0`` means exactly "no better than the
    constant a practitioner would otherwise use" -- the cleanest of the
    scale-free metrics to quote. ``R2`` is computed against the training
    mean.
    """
    from scipy import stats as scipy_stats

    scale: Dict[str, float] = {}
    centre: Dict[str, float] = {}
    centres: Dict[str, float] = {}
    deviations: Dict[str, float] = {}
    if train_literals is not None:
        grouped = train_literals.groupby("attribute")["value"]
        scale = grouped.std(ddof=0).to_dict()
        centre = grouped.mean().to_dict()
        centres = grouped.median().to_dict()
        deviations = grouped.apply(
            lambda v: 1.4826 * float(np.median(np.abs(v - v.median())))
        ).to_dict()

    records = []
    for attribute, group in frame.groupby("attribute", sort=True):
        truth = group[value_column].to_numpy(dtype=np.float64)
        predicted = group[prediction_column].to_numpy(dtype=np.float64)
        errors = predicted - truth

        std = scale.get(attribute)
        if std is None or not np.isfinite(std) or std == 0:
            std = float(np.std(truth)) or 1.0
        mean = centre.get(attribute, float(np.mean(truth)))
        median = centres.get(attribute, float(np.median(truth)))
        deviation = deviations.get(attribute)
        if deviation is None or not np.isfinite(deviation) or deviation == 0:
            deviation = float(np.mean(np.abs(truth - np.median(truth)))) or std

        mae = float(np.mean(np.abs(errors)))
        rmse = float(np.sqrt(np.mean(errors**2)))
        baseline_mae = float(np.mean(np.abs(truth - median))) or np.nan

        if len(truth) > 1 and np.ptp(truth) > 0 and np.ptp(predicted) > 0:
            spearman = float(scipy_stats.spearmanr(truth, predicted).statistic)
        else:
            spearman = np.nan

        baseline = float(np.sum((truth - mean) ** 2))
        r2 = 1.0 - float(np.sum(errors**2)) / baseline if baseline > 0 else np.nan

        records.append(
            {
                "attribute": attribute, "n": len(truth), "MAE": mae, "RMSE": rmse,
                "NMAE": mae / std, "RNMAE": mae / deviation, "MASE": mae / baseline_mae,
                "spearman": spearman, "R2": r2,
            }
        )
    return pd.DataFrame.from_records(records)


def evaluate_predictor(
    predictor, test_literals: pd.DataFrame, train_literals: Optional[pd.DataFrame] = None,
) -> Dict[str, object]:
    """Score a literal predictor on a held-out frame.

    Works with anything exposing ``predict_literals(entity=..., attribute=...)``.
    """
    frame = test_literals.copy()
    entities = frame["head"].to_list()
    attributes = frame["attribute"].to_list()
    frame["predictions"] = predictor.predict_literals(entity=entities, attribute=attributes)
    per_attribute = per_attribute_metrics(frame, train_literals=train_literals)
    return {"per_attribute": per_attribute, "predictions": frame}


# ----------------------------------------------------------------------
# LitEm's own runs
# ----------------------------------------------------------------------

#: The LitEm run read: TransE embeddings, d = 128, MLP regression head.
LITEM_RUN = "TransE_128_mlp"


def litem_scores(literal_results: str, dataset_dir: str) -> pd.DataFrame:
    """Per-run test MAE, RMSE and R^2 of LitEm, from its own result files.

    ``<literal_results>/<dataset>/TransE_128_mlp/lit_eval_results_n_run.csv``
    holds MAE and RMSE for each of LitEm's five runs. LitEm saved no predictions, so R^2 is recovered
    from the RMSE on LitEm's own test rows (the shipped ``literals/test.txt``,
    unfiltered), against the training mean, as for the backbones.
    """
    name = os.path.basename(os.path.normpath(dataset_dir))
    runs = pd.read_csv(os.path.join(literal_results, name, LITEM_RUN, "lit_eval_results_n_run.csv"))
    read = lambda part: pd.read_csv(os.path.join(dataset_dir, "literals", f"{part}.txt"), sep="\t",
                                    header=None, names=["head", "attribute", "value"])
    test, train = read("test"), read("train")
    mean = test["attribute"].map(train.groupby("attribute")["value"].mean())
    spread = ((test["value"] - mean) ** 2).groupby(test["attribute"]).sum()
    n = test.groupby("attribute").size()
    rows = []
    for run in range(1, 1 + sum(c.startswith("MAE_run_") for c in runs)):
        for _, r in runs.iterrows():
            a = r["relation"]
            rmse = r[f"RMSE_run_{run}"]
            rows.append({"attribute": a, "n": int(n[a]), "MAE": r[f"MAE_run_{run}"], "RMSE": rmse,
                         "R2": 1.0 - n[a] * rmse ** 2 / spread[a], "seed": run})
    return pd.DataFrame(rows).assign(dataset=name, backbone="litem", split="test")


#: Metric names as printed in the main table.
METRIC_LABELS = {"MAE": "MAE", "RMSE": "RMSE", "R2": "R²"}

#: Datasets in table order.
DATASETS = ("FB15k-237", "YAGO15k", "DB15K")

#: Attributes with this many test rows or fewer are not evaluated: below it,
#: seed-to-seed noise swamps the differences between methods.
MIN_TEST_ROWS = 40

#: Short attribute labels for the table rows.
SHORT = {
    "birthDate": "birthDate", "deathDate": "deathDate", "foundingDate": "foundingDate",
    "releaseDate": "releaseDate", "wgs84_pos#lat": "latitude", "wgs84_pos#long": "longitude",
    "wasBornOnDate": "BornOnDate", "diedOnDate": "diedOnDate", "hasLatitude": "Latitude",
    "hasLongitude": "Longitude", "wasCreatedOnDate": "CreatedOnDate",
    "wasDestroyedOnDate": "DestroyedOnDate",
    "people.person.date_of_birth": "date_of_birth",
    "people.deceased_person.date_of_death": "date_of_death",
    "location.geocode.latitude": "latitude", "location.geocode.longitude": "longitude",
    "location.location.area": "loc.area (10^6)",
    "topic_server.population_number": "pop._number (10^6)",
    "organization.organization.date_founded": "org.date_founded",
    "location.dated_location.date_founded": "loc.date_founded",
    "film.film.initial_release_date": "release_date",
    "people.person.height_meters": "height_meters",
}

#: Printed in millions.
MILLIONS = {"location.location.area", "topic_server.population_number"}


# ----------------------------------------------------------------------
# Result files and the tables
# ----------------------------------------------------------------------


def result_path(results_dir: str, dataset: str, backbone: str, conversion: str) -> str:
    """``<dataset>_<backbone>_<conversion>.csv``."""
    return os.path.join(results_dir, f"{dataset}_{backbone}_{conversion}.csv")


def mrap_path(results_dir: str, dataset: str) -> str:
    return os.path.join(results_dir, f"{dataset}_mrap.csv")


def litem_path(results_dir: str, dataset: str) -> str:
    return os.path.join(results_dir, f"{dataset}_litem.csv")


def llm_direct_path(results_dir: str, dataset: str) -> str:
    return os.path.join(results_dir, f"{dataset}_llm_direct.csv")


def load_results(results_dir: str, conversions: Sequence[str] = ("full",),
                 mrap: bool = True) -> pd.DataFrame:
    """Test rows of the given conversions, plus MrAP.

    Raises when two files disagree on an attribute's test-row count at the same
    seed: they were scored on different rows and cannot share a table.
    """
    paths = [p for c in conversions for p in glob.glob(os.path.join(results_dir, f"*_{c}.csv"))]
    if mrap:
        paths += glob.glob(mrap_path(results_dir, "*"))
    if not paths:
        raise FileNotFoundError(f"No result files for {list(conversions)} under {results_dir}.")
    frame = pd.concat([pd.read_csv(p) for p in sorted(paths)], ignore_index=True)
    frame = frame[(frame["split"] == "test") & frame["conversion"].fillna("mrap").isin([*conversions, "mrap"])]
    counts = frame.groupby(["dataset", "attribute", "seed"])["n"].nunique()
    mixed = counts[counts > 1]
    if len(mixed):
        raise ValueError(f"Result files disagree on test rows for {mixed.index.tolist()[:5]}.")
    return frame


def _fmt(x: Optional[float]) -> str:
    if x is None or not np.isfinite(x):
        return "--"
    if x >= 100:
        return f"{x:.1f}"
    if x < 0.1:
        return f"{x:.4f}"
    return f"{x:.3f}"


def _mark(values: Dict[str, Optional[float]], cells: Dict[str, str], second: bool,
          higher: bool = False) -> Dict[str, str]:
    """Bold the best value, and italicize the second best when ``second``."""
    ranked = sorted({v for v in values.values() if v is not None and np.isfinite(v)}, reverse=higher)
    out = {}
    for key, value in values.items():
        cell = cells[key]
        if ranked and value == ranked[0]:
            cell = f"**{cell}**"
        elif second and len(ranked) > 1 and value == ranked[1]:
            cell = f"*{cell}*"
        out[key] = cell
    return out


def _complete_means(frame: pd.DataFrame, metric: str, seeds: int) -> pd.Series:
    """Mean over seeds per (dataset, attribute, backbone), only where every seed ran."""
    grouped = frame.groupby(["dataset", "attribute", "backbone"])[metric]
    means = grouped.mean()
    return means[grouped.count() >= seeds]


def shipped_table(results_dir: str, backbones: Sequence[str], labels: Dict[str, str],
                  seeds: int = 3) -> str:
    """Main table: test MAE, RMSE and R^2 on the shipped splits against LitEm and MrAP.

    LitEm is read from its own runs (:func:`litem_scores`) and kept out of the
    test-row check: on DB15K ``birthDate`` it scores two rows that
    :func:`load_literal_splits` drops.
    """
    frame = load_results(results_dir)
    litem = pd.concat([pd.read_csv(p) for p in sorted(glob.glob(litem_path(results_dir, "*")))],
                      ignore_index=True).groupby(["dataset", "attribute"])
    ours = frame[frame["backbone"] != "mrap"]
    mrap = frame[frame["backbone"] == "mrap"].groupby(["dataset", "attribute"])
    n = frame.groupby(["dataset", "attribute"])["n"].first()
    metrics = ("MAE", "RMSE", "R2")
    means = {m: _complete_means(ours, m, seeds) for m in metrics}
    mrap_means = {m: mrap[m].mean() for m in metrics}
    litem_means = {m: litem[m].mean() for m in metrics}

    columns = ["LitEm", "MrAP", *backbones]
    lines, wins, total = [], {b: 0 for b in backbones}, 0
    lines.append("| Attribute | n | Metric | " + " | ".join(["LitEm", "MrAP"] + [labels[b] for b in backbones]) + " |")
    lines.append("|---|---:|---|" + "---:|" * len(columns))
    for dataset in DATASETS:
        rows = [(a, int(k)) for (d, a), k in n.items() if d == dataset and k > MIN_TEST_ROWS]
        if not rows:
            continue
        lines.append(f"| **{dataset}** |" + " |" * (len(columns) + 2))
        for attribute, count in sorted(rows, key=lambda r: -r[1]):
            for metric in metrics:
                values = {"LitEm": litem_means[metric].get((dataset, attribute)),
                          "MrAP": mrap_means[metric].get((dataset, attribute))}
                for b in backbones:
                    values[b] = means[metric].get((dataset, attribute, b))
                if metric == "MAE":
                    total += 1
                    bar = min(v for v in (values["LitEm"], values["MrAP"]) if v is not None)
                    for b in backbones:
                        wins[b] += values[b] is not None and values[b] < bar
                if metric == "R2":
                    text = {k: "--" if v is None or not np.isfinite(v) else f"{v:.3f}" for k, v in values.items()}
                else:
                    scale = 1e6 if attribute in MILLIONS else 1.0
                    values = {k: None if v is None else v / scale for k, v in values.items()}
                    text = {k: _fmt(v) for k, v in values.items()}
                cells = _mark(values, text, second=True, higher=metric == "R2")
                name = SHORT.get(attribute, attribute) if metric == "MAE" else ""
                lines.append(f"| {name} | {count if metric == 'MAE' else ''} | {METRIC_LABELS[metric]} | "
                             + " | ".join(cells[c] for c in columns) + " |")
    summary = "; ".join(f"{labels[b]} {wins[b]}/{total}" for b in backbones)
    return "\n".join(lines) + f"\n\nLower MAE than both LitEm and MrAP: {summary}."


def llm_table(results_dir: str, backbones: Sequence[str], labels: Dict[str, str],
              seeds: int = 3) -> str:
    """Qwen3-8B answering directly and as a column: median test MASE over the attributes.

    The direct answers come from one greedy run. The learners' MASE is a mean
    over seeds per attribute, kept only where every seed ran.
    """
    frame = load_results(results_dir, ("full", "full_llm"), mrap=False)
    frame = frame[frame["n"] > MIN_TEST_ROWS]
    direct = pd.concat([pd.read_csv(p) for p in sorted(glob.glob(llm_direct_path(results_dir, "*")))],
                       ignore_index=True)
    direct = direct[direct["n"] > MIN_TEST_ROWS]
    expected = frame.groupby(["dataset", "attribute"]).ngroups

    lines = ["| Table or prompt | Learner | Median MASE | MASE < 1 | Attributes |",
             "|---|---|---:|---:|---:|"]

    def row(name: str, learner: str, values: pd.Series) -> None:
        median = _fmt(values.median()) if len(values) == expected else "--"
        lines.append(f"| {name} | {learner} | {median} | {int((values < 1).sum())} | {len(values)} |")

    for arm, label in (("name", "name only"), ("graph", "one-hop triples"),
                       ("graph_values", "triples, neighbor values")):
        row(label, "Qwen3-8B", direct[direct["arm"] == arm].set_index(["dataset", "attribute"])["MASE"])
    for conversion, label in (("full", "conversion"), ("full_llm", "+ answer column")):
        means = _complete_means(frame[frame["conversion"] == conversion], "MASE", seeds)
        for i, b in enumerate(backbones):
            row(label if i == 0 else "", labels[b], means.xs(b, level="backbone") if b in
                means.index.get_level_values("backbone") else pd.Series(dtype=float))

    full = _complete_means(frame[frame["conversion"] == "full"], "MASE", seeds)
    with_llm = _complete_means(frame[frame["conversion"] == "full_llm"], "MASE", seeds)
    shared = full.index.intersection(with_llm.index)
    better = (with_llm[shared] < full[shared]).groupby(level="backbone").agg(["sum", "size"])
    summary = "; ".join(f"{labels[b]} {int(better.loc[b, 'sum'])}/{int(better.loc[b, 'size'])}"
                        for b in backbones if b in better.index)
    return "\n".join(lines) + f"\n\nAttributes where the answer column lowers MASE: {summary}."
