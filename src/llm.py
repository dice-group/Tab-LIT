"""A language model as a direct predictor of numeric literals, and as a source of a table column.

Every prompt names one entity and one attribute and asks for the value only.
Three prompt types differ in what they show the model:

``name``
    the entity's name.
``graph``
    the entity anonymized as ``X`` with up to 30 of its one-hop training
    triples, written with readable entity and relation names.
``graph_values``
    the same triples, each with up to three known training values of the
    neighbor in brackets.

Decoding is greedy with thinking disabled. A reply without a parsable number
is asked once more with the reply prefilled as ``My best estimate is``. Dates
are asked as ``YYYY-MM-DD`` and re-encoded in each dataset's own float
convention.

:func:`answer_table` asks the ``name`` prompt for every (entity, attribute)
cell of the shipped splits; :class:`src.featurizer.LLMAnswerFeaturiser` turns
that table into the ``llm::<a>`` columns. :func:`direct_answers` asks all three
prompts for the test rows, and :func:`score_direct` scores them like any other
predictor, falling back to the training median where no reply parsed.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import urllib.parse
from datetime import date
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .featurizer import KGIndex, LiteralSplits

ARMS = ("name", "graph", "graph_values")
RETRY_PREFIX = "My best estimate is "
MAX_FACTS = 30
MAX_VALUES_PER_NEIGHBOR = 3

DATE = "date"
#: attribute -> (what to ask for, unit or format hint)
ATTRIBUTES = {
    "birthDate": ("date of birth", DATE),
    "deathDate": ("date of death", DATE),
    "foundingDate": ("founding date", DATE),
    "releaseDate": ("release date", DATE),
    "completionDate": ("completion date", DATE),
    "formationDate": ("formation date", DATE),
    "wgs84_pos#lat": ("latitude", "decimal degrees, negative for south"),
    "wgs84_pos#long": ("longitude", "decimal degrees, negative for west"),
    "height": ("height", "metres"),
    "people.person.date_of_birth": ("date of birth", DATE),
    "people.deceased_person.date_of_death": ("date of death", DATE),
    "film.film.initial_release_date": ("initial release date", DATE),
    "organization.organization.date_founded": ("founding date", DATE),
    "location.dated_location.date_founded": ("founding date", DATE),
    "location.geocode.latitude": ("latitude", "decimal degrees, negative for south"),
    "location.geocode.longitude": ("longitude", "decimal degrees, negative for west"),
    "location.location.area": ("area", "square kilometres"),
    "topic_server.population_number": ("population", "number of people"),
    "people.person.height_meters": ("height", "metres"),
    "people.person.weight_kg": ("weight", "kilograms"),
    "wasBornOnDate": ("date of birth", DATE),
    "diedOnDate": ("date of death", DATE),
    "wasCreatedOnDate": ("creation or founding date", DATE),
    "wasDestroyedOnDate": ("destruction or dissolution date", DATE),
    "happenedOnDate": ("date it happened", DATE),
    "hasLatitude": ("latitude", "decimal degrees, negative for south"),
    "hasLongitude": ("longitude", "decimal degrees, negative for west"),
}


# ----------------------------------------------------------------------
# Dates and answers
# ----------------------------------------------------------------------


def encode_date(dataset: str, year: int, month, day) -> float:
    """A date as the dataset stores it: one float, in the dataset's own convention."""
    if dataset == "DB15K":  # year + (day of year - 1) / 365
        if month is None:
            return float(year)
        try:
            day_of_year = date(2001, month, day or 1).timetuple().tm_yday
        except ValueError:
            day_of_year = 1
        return year + (day_of_year - 1) / 365
    if dataset == "FB15k-237":  # year + month / 12
        return year + (month / 12 if month else 0.0)
    return year + (month or 0) / 100 + (day or 0) / 10000  # YAGO15k: yyyy.mmdd


def decode_date(dataset: str, value: float):
    """``(year, month or None)`` of a stored date.

    FB15k-237 stores December as the next whole year, which cannot be told
    apart from a year-only date, so a whole year is shown without a month.
    """
    year = math.floor(value)
    fraction = value - year
    if dataset == "DB15K":
        month = min(12, int(fraction * 365 // 30.42) + 1) if fraction > 0 else None
    elif dataset == "FB15k-237":
        month = int(round(fraction * 12)) or None
    else:
        month = int(round(fraction * 100)) or None
    if month is not None and not 1 <= month <= 12:
        month = None
    return year, month


#: Tried in order, so that a verbose reply ("founded on August 1, 1984.
#: 1984-08-01") yields the ISO date or the four-digit year, never the day.
DATE_PATTERNS = [
    re.compile(r"(?<!\d)(-?\d{3,4})-(\d{1,2})-(\d{1,2})(?!\d)"),
    re.compile(r"(?<!\d)(-?\d{3,4})-(\d{1,2})(?![\d-])"),
    re.compile(r"(?<![\d.])(-?\d{3,4})(?![\d.])"),
    re.compile(r"(?<![\d.])(-?\d{1,2})(?![\d.])"),
]
DECIMAL_RE = re.compile(r"-?\d[\d,]*\.\d+(?:[eE][-+]?\d+)?")
NUMBER_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?(?:[eE][-+]?\d+)?")


def parse_answer(text: str, dataset: str, attribute: str) -> float:
    """The number in a reply, in the dataset's encoding, or ``NaN``."""
    kind = ATTRIBUTES[attribute][1]
    text = str(text).strip()
    if kind == DATE:
        for pattern in DATE_PATTERNS:
            match = pattern.search(text)
            if match:
                break
        else:
            return np.nan
        year, month, day = ([int(g) for g in match.groups()] + [None, None])[:3]
        if month is not None and not 1 <= month <= 12:
            month, day = None, None
        return encode_date(dataset, year, month, day)
    # Coordinates come back as decimals; prefer one over a stray integer in prose.
    match = (DECIMAL_RE.search(text) if "degrees" in kind else None) or NUMBER_RE.search(text)
    return float(match.group().replace(",", "")) if match else np.nan


# ----------------------------------------------------------------------
# Prompts
# ----------------------------------------------------------------------


def readable(name: str, labels: Dict[str, str]) -> str:
    """A label when one is given, else the last segment of the IRI with underscores as spaces."""
    if name in labels:
        return labels[name]
    name = name.strip("<>").rsplit("/", 1)[-1].rsplit("#", 1)[-1]
    return urllib.parse.unquote(name).replace("_", " ")


def show(dataset: str, attribute: str, value: float) -> str:
    what, hint = ATTRIBUTES.get(attribute, (attribute, ""))
    if hint == DATE:
        year, month = decode_date(dataset, value)
        return f"{what} = {year}" + (f"-{month:02d}" if month else "")
    unit = {"metres": " m", "kilograms": " kg", "square kilometres": " km2"}.get(hint, "")
    return f"{what} = {value:g}{unit}"


class Evidence:
    """Readable one-hop facts about an entity, from the training graph and training values."""

    def __init__(self, dataset: str, kg: KGIndex, train: pd.DataFrame, labels: Dict[str, str]):
        self.dataset, self.kg, self.labels = dataset, kg, labels
        self.values = {entity: dict(zip(group["attribute"], group["value"]))
                       for entity, group in train.groupby("head")}
        self.by_entity: Dict[int, list] = {}
        for h, r, t in kg.triples:
            self.by_entity.setdefault(int(h), []).append((int(h), int(r), int(t)))
            if h != t:
                self.by_entity.setdefault(int(t), []).append((int(h), int(r), int(t)))

    def name(self, entity: str) -> str:
        return readable(entity, self.labels)

    def _neighbor_values(self, neighbor: str, attribute: str) -> str:
        known = self.values.get(neighbor, {})
        order = sorted(known, key=lambda a: (a != attribute, a))[:MAX_VALUES_PER_NEIGHBOR]
        return "; ".join(show(self.dataset, a, known[a]) for a in order if a in ATTRIBUTES)

    def facts(self, entity: str, attribute: str, with_values: bool, rng) -> str:
        idx = self.kg.entity_to_idx.get(entity)
        rows = self.by_entity.get(idx, []) if idx is not None else []
        lines = []
        for h, r, t in dict.fromkeys(rows):  # the graph files repeat some triples
            relation = readable(self.kg.idx_to_relation[r], {})
            other = self.kg.idx_to_entity[t if h == idx else h]
            text = (f"(X, {relation}, {self.name(other)})" if h == idx
                    else f"({self.name(other)}, {relation}, X)")
            values = self._neighbor_values(other, attribute) if with_values else ""
            has_target = with_values and attribute in self.values.get(other, {})
            lines.append((not has_target, not values, text + (f"  [{values}]" if values else "")))
        if len(lines) > MAX_FACTS:
            # Neighbors carrying the target attribute first, then any value, then the rest.
            rng.shuffle(lines)
            lines = sorted(lines, key=lambda x: (x[0], x[1]))[:MAX_FACTS]
        return "\n".join(x[2] for x in lines) if lines else "(no known facts)"


def build_prompt(arm: str, name: str, facts: str, attribute: str) -> str:
    what, hint = ATTRIBUTES[attribute]
    form = ("YYYY-MM-DD (YYYY alone if only the year is known; a minus sign for BCE)"
            if hint == DATE else hint)
    subject = name if arm == "name" else "X"
    parts = []
    if arm != "name":
        parts.append("An entity X is described by these facts (known values of related "
                     f"entities in brackets):\n{facts}")
    parts.append(f"What is the {what} of {subject}?")
    parts.append(f"Answer with the value only, as {form}. If unsure, give your best estimate.")
    return "\n\n".join(parts)


# ----------------------------------------------------------------------
# Generation
# ----------------------------------------------------------------------


class Generator:
    """Greedy chat completion with Hugging Face ``transformers``, cached by prompt.

    The cache is a JSONL file keyed by (model, prefix, prompt), so an interrupted
    run resumes where it stopped. The model is loaded only when a prompt is not
    in the cache.
    """

    def __init__(self, model_id: str, cache_path: str, batch_size: int = 32,
                 max_new_tokens: int = 24, char_budget: int = 40_000):
        self.model_id, self.batch_size = model_id, batch_size
        self.max_new_tokens, self.char_budget = max_new_tokens, char_budget
        self.model = None
        self.cache_path = cache_path
        self.cache: Dict[str, str] = {}
        if os.path.exists(cache_path):
            with open(cache_path) as handle:
                for line in handle:
                    record = json.loads(line)
                    self.cache[record["key"]] = record["answer"]

    def _key(self, prompt: str, prefix: str) -> str:
        return hashlib.sha1(f"{self.model_id}\0{prefix}\0{prompt}".encode()).hexdigest()

    def __call__(self, prompts: List[str], prefix: str = "") -> List[str]:
        import torch

        keys = [self._key(p, prefix) for p in prompts]
        todo = sorted({k: p for k, p in zip(keys, prompts) if k not in self.cache}.items(),
                      key=lambda kv: len(kv[1]))
        if not todo:
            return [self.cache[k] for k in keys]
        if self.model is None:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, padding_side="left")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id, dtype=torch.bfloat16).to("cuda").eval()
        os.makedirs(os.path.dirname(self.cache_path) or ".", exist_ok=True)
        with open(self.cache_path, "a") as handle:
            start = 0
            while start < len(todo):
                # Sorted by length: shrink the batch until batch size x longest
                # prompt fits the character budget.
                size = self.batch_size
                while size > 1 and size * len(todo[min(start + size, len(todo)) - 1][1]) > self.char_budget:
                    size //= 2
                batch, start = todo[start:start + size], start + size
                chats = [self.tokenizer.apply_chat_template(
                    [{"role": "user", "content": p}], tokenize=False,
                    add_generation_prompt=True, enable_thinking=False) + prefix for _, p in batch]
                encoded = self.tokenizer(chats, return_tensors="pt", padding=True).to(self.model.device)
                with torch.no_grad():
                    out = self.model.generate(**encoded, max_new_tokens=self.max_new_tokens, do_sample=False)
                answers = self.tokenizer.batch_decode(out[:, encoded["input_ids"].shape[1]:],
                                                      skip_special_tokens=True)
                for (key, _), answer in zip(batch, answers):
                    self.cache[key] = answer
                    handle.write(json.dumps({"key": key, "answer": answer}) + "\n")
                handle.flush()
        return [self.cache[k] for k in keys]


def _ask(generate: Generator, prompts: List[str], dataset: str, attributes: List[str]) -> pd.DataFrame:
    """Replies, parsed values and the retry flag, asking unparsable prompts once more."""
    answers = generate(prompts)
    parsed = [parse_answer(t, dataset, a) for t, a in zip(answers, attributes)]
    retried = [np.isnan(p) for p in parsed]
    again = [i for i, r in enumerate(retried) if r]
    if again:
        replies = generate([prompts[i] for i in again], prefix=RETRY_PREFIX)
        for i, reply in zip(again, replies):
            answers[i] = RETRY_PREFIX + reply
            parsed[i] = parse_answer(reply, dataset, attributes[i])
    return pd.DataFrame({"answer": answers, "parsed": parsed, "retried": retried})


def answer_table(dataset: str, kg: KGIndex, splits: LiteralSplits, labels: Dict[str, str],
                 generate: Generator) -> pd.DataFrame:
    """The ``name`` answer for every (entity, attribute) cell of the three splits."""
    cells = pd.concat([splits.train, splits.valid, splits.test])[["head", "attribute"]]
    cells = cells.drop_duplicates()
    cells = cells[cells["attribute"].isin(ATTRIBUTES)].reset_index(drop=True)
    prompts = [build_prompt("name", readable(e, labels), "", a)
               for e, a in zip(cells["head"], cells["attribute"])]
    replies = _ask(generate, prompts, dataset, cells["attribute"].tolist())
    return cells.assign(answer=replies["answer"], llm_value=replies["parsed"])


def direct_answers(dataset: str, kg: KGIndex, splits: LiteralSplits, labels: Dict[str, str],
                   generate: Generator, seed: int = 1) -> pd.DataFrame:
    """All three prompt types for every test row."""
    test = splits.test[splits.test["attribute"].isin(ATTRIBUTES)].reset_index(drop=True)
    evidence = Evidence(dataset, kg, splits.train, labels)
    rng = np.random.default_rng(seed)
    names = [evidence.name(e) for e in test["head"]]
    plain = [evidence.facts(e, a, False, rng) for e, a in zip(test["head"], test["attribute"])]
    valued = [evidence.facts(e, a, True, rng) for e, a in zip(test["head"], test["attribute"])]
    frames = []
    for arm in ARMS:
        facts = plain if arm == "graph" else valued
        prompts = [build_prompt(arm, n, f, a) for n, f, a in zip(names, facts, test["attribute"])]
        replies = _ask(generate, prompts, dataset, test["attribute"].tolist())
        frames.append(pd.concat([test.assign(arm=arm), replies], axis=1))
    return pd.concat(frames, ignore_index=True)


def score_direct(direct: pd.DataFrame, train: pd.DataFrame) -> pd.DataFrame:
    """Per-attribute MAE and MASE of each prompt type; no parse falls back to the training median."""
    medians = train.groupby("attribute")["value"].median()
    median = direct["attribute"].map(medians)
    rows = direct.assign(error=(direct["value"] - direct["parsed"].fillna(median)).abs(),
                         baseline=(direct["value"] - median).abs())
    out = rows.groupby(["arm", "attribute"]).agg(
        n=("error", "size"), MAE=("error", "mean"), baseline_MAE=("baseline", "mean"),
        parse_rate=("parsed", lambda s: s.notna().mean())).reset_index()
    out["MASE"] = out["MAE"] / out["baseline_MAE"]
    return out


def load_labels(path: Optional[str]) -> Dict[str, str]:
    """``entity \\t label`` lines; FB15k-237 needs them, the IRIs of the others are readable."""
    if not path or not os.path.exists(path):
        return {}
    return dict(pd.read_csv(path, sep="\t", header=None, quoting=3).values)
