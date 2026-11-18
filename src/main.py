"""Command line for Tab-LIT. Run from the repository root.

    python -m src.main run --dataset DB15K --backbone gbt
    python -m src.main experiment --backbones tabpfn tabicl kumo nori gbt
    python -m src.main experiment --conversion full_llm --backbones kumo tabpfn gbt
    python -m src.main mrap --mrap_dir ../literal-embeddings/MrAP
    python -m src.main llm
    python -m src.main report --readme

``--dataset`` takes a dataset directory or a name (``DB15K``, ``FB15k-237``,
``YAGO15k``) resolved against the ``KGs/`` folder next to this package, or
``$TABLIT_KG_ROOT`` when set.
"""

from __future__ import annotations

import argparse
import json
import os
import re

from .evaluation import (DATASETS, evaluate_predictor, llm_direct_path, llm_table, mrap_path,
                         result_path, shipped_table)
from .featurizer import CONVERSIONS, KGIndex, build_featuriser, load_literal_splits
from .runner import BACKBONE_LABELS, BACKBONES, LLM_ANSWERS, build_predictor, run_experiment

PACKAGE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(PACKAGE, "results")
REPORTED_BACKBONES = ["tabpfn", "tabicl", "kumo", "nori"]
LLM_BACKBONES = ["kumo", "tabpfn", "gbt"]


def _dataset_dir(dataset: str) -> str:
    if os.path.isdir(dataset):
        return dataset
    root = os.environ.get("TABLIT_KG_ROOT", os.path.join(PACKAGE, "..", "KGs"))
    candidate = os.path.join(root, dataset)
    if os.path.isdir(candidate):
        return candidate
    raise FileNotFoundError(f"Could not resolve dataset '{dataset}' as a path or a name under {root}.")


def cmd_run(args: argparse.Namespace) -> None:
    dataset_dir = _dataset_dir(args.dataset)
    dataset = os.path.basename(os.path.normpath(dataset_dir))
    kg_index = KGIndex.from_dataset_dir(dataset_dir)
    splits = load_literal_splits(os.path.join(dataset_dir, "literals"), kg_index=kg_index)
    featuriser = build_featuriser(CONVERSIONS[args.conversion], seed=args.seed,
                                  llm_answers=LLM_ANSWERS.format(dataset=dataset))
    predictor = build_predictor(args.backbone, featuriser, seed=args.seed)
    predictor.fit(kg_index, splits.train)
    metrics = evaluate_predictor(predictor, splits.test, splits.train)
    print(f"\n{args.backbone} / {args.conversion} on {dataset} (seed {args.seed})")
    print(metrics["per_attribute"].to_string(index=False))


def cmd_experiment(args: argparse.Namespace) -> None:
    os.makedirs(RESULTS, exist_ok=True)
    for backbone in args.backbones:
        for dataset in args.datasets:
            out = result_path(RESULTS, dataset, backbone, args.conversion)
            if os.path.exists(out) and not args.overwrite:
                print(f"skip {out} (exists; pass --overwrite to rerun)")
                continue
            frame, manifest = run_experiment(
                _dataset_dir(dataset), backbone, args.conversion, seeds=args.seeds,
                n_threads=args.n_threads,
            )
            frame.to_csv(out, index=False)
            with open(out.replace(".csv", "_manifest.json"), "w") as handle:
                json.dump(manifest, handle, indent=2)
            print(f"wrote {out}")


def cmd_mrap(args: argparse.Namespace) -> None:
    from .mrap import run_mrap

    mrap_dir = args.mrap_dir or os.environ.get("MRAP_DIR")
    if not mrap_dir:
        raise SystemExit("Pass --mrap_dir or set MRAP_DIR to a literal-embeddings/MrAP checkout.")
    os.makedirs(RESULTS, exist_ok=True)
    for dataset in args.datasets:
        out = mrap_path(RESULTS, dataset)
        run_mrap(mrap_dir, _dataset_dir(dataset)).to_csv(out, index=False)
        print(f"wrote {out}")


def cmd_llm(args: argparse.Namespace) -> None:
    from . import llm

    generate = llm.Generator(args.model, args.cache)
    os.makedirs(RESULTS, exist_ok=True)
    for dataset in args.datasets:
        dataset_dir = _dataset_dir(dataset)
        kg = KGIndex.from_dataset_dir(dataset_dir)
        splits = load_literal_splits(os.path.join(dataset_dir, "literals"), kg_index=kg)
        labels = llm.load_labels(os.path.join(dataset_dir, "entity2text.txt"))
        answers = llm.answer_table(dataset, kg, splits, labels, generate)
        answers.to_csv(LLM_ANSWERS.format(dataset=dataset), index=False)
        direct = llm.direct_answers(dataset, kg, splits, labels, generate)
        direct.to_csv(LLM_ANSWERS.format(dataset=dataset).replace("_answers", "_direct"), index=False)
        scores = llm.score_direct(direct, splits.train).assign(dataset=dataset, model=args.model)
        scores.to_csv(llm_direct_path(RESULTS, dataset), index=False)
        print(f"[llm] {dataset}: {len(answers)} cells, {answers['llm_value'].notna().mean():.1%} parsed; "
              f"{len(direct)} direct rows")


def cmd_report(args: argparse.Namespace) -> None:
    labels = dict(BACKBONE_LABELS)
    sections = [
        "### Shipped splits: MAE\n\n" + shipped_table(RESULTS, args.backbones, labels),
        "### A language model as baseline and as a column: MASE\n\n"
        + llm_table(RESULTS, LLM_BACKBONES, labels),
    ]
    text = "\n\n".join(sections)
    print(text)
    if args.readme:
        path = os.path.join(PACKAGE, "README.md")
        readme = open(path).read()
        block = f"<!-- BEGIN GENERATED TABLES -->\n{text}\n<!-- END GENERATED TABLES -->"
        readme, count = re.subn(
            r"<!-- BEGIN GENERATED TABLES -->.*<!-- END GENERATED TABLES -->",
            lambda _: block, readme, flags=re.S,
        )
        if count != 1:
            raise SystemExit("README.md has no generated-tables block to replace.")
        open(path, "w").write(readme)
        print(f"\nupdated {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="Fit one backbone on one dataset and print per-attribute metrics.")
    run.add_argument("--dataset", default="DB15K")
    run.add_argument("--backbone", choices=sorted(BACKBONES), default="gbt")
    run.add_argument("--conversion", choices=sorted(CONVERSIONS), default="full")
    run.add_argument("--seed", type=int, default=1)
    run.set_defaults(func=cmd_run)

    exp = sub.add_parser("experiment", help="Run one conversion x backbones x seeds and write results/.")
    exp.add_argument("--datasets", nargs="+", default=list(DATASETS))
    exp.add_argument("--backbones", nargs="+", choices=sorted(BACKBONES), default=REPORTED_BACKBONES)
    exp.add_argument("--conversion", choices=sorted(CONVERSIONS), default="full")
    exp.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    exp.add_argument("--n_threads", type=int, default=4)
    exp.add_argument("--overwrite", action="store_true")
    exp.set_defaults(func=cmd_experiment)

    mrap = sub.add_parser("mrap", help="Rerun MrAP on the same test rows.")
    mrap.add_argument("--mrap_dir", default=None)
    mrap.add_argument("--datasets", nargs="+", default=list(DATASETS))
    mrap.set_defaults(func=cmd_mrap)

    llm = sub.add_parser("llm", help="Ask the language model; write llm/ and the direct scores.")
    llm.add_argument("--datasets", nargs="+", default=list(DATASETS))
    llm.add_argument("--model", default="Qwen/Qwen3-8B")
    llm.add_argument("--cache", default=os.path.join(PACKAGE, "llm", "cache", "qwen3-8b.jsonl"))
    llm.set_defaults(func=cmd_llm)

    report = sub.add_parser("report", help="Print the result tables from results/.")
    report.add_argument("--backbones", nargs="+", default=REPORTED_BACKBONES)
    report.add_argument("--readme", action="store_true", help="also rewrite the tables in README.md")
    report.set_defaults(func=cmd_report)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
