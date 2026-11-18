"""Answer parsing, date encoding and scoring of the language model's replies."""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.llm import build_prompt, decode_date, parse_answer, readable, score_direct  # noqa: E402


class TestParseAnswer:
    @pytest.mark.parametrize("dataset, expected", [
        ("DB15K", 1984 + 212 / 365), ("FB15k-237", 1984 + 8 / 12), ("YAGO15k", 1984.0801),
    ])
    def test_iso_date_in_each_encoding(self, dataset, expected):
        assert parse_answer("1984-08-01", dataset, "birthDate" if dataset == "DB15K" else
                            {"FB15k-237": "people.person.date_of_birth", "YAGO15k": "wasBornOnDate"}[dataset]
                            ) == pytest.approx(expected)

    def test_verbose_date_yields_the_iso_date_not_the_day(self):
        assert parse_answer("Founded on August 1, 1984. 1984-08-01", "FB15k-237",
                            "organization.organization.date_founded") == pytest.approx(1984 + 8 / 12)

    def test_coordinates_prefer_the_decimal(self):
        assert parse_answer("About 2 km away: 48.8566", "DB15K", "wgs84_pos#lat") == pytest.approx(48.8566)

    def test_no_number_is_nan(self):
        assert np.isnan(parse_answer("I do not know.", "DB15K", "height"))

    def test_decode_inverts_yago_encoding(self):
        assert decode_date("YAGO15k", 1984.0801) == (1984, 8)


class TestPrompts:
    def test_name_prompt(self):
        prompt = build_prompt("name", "Patton Oswalt", "", "birthDate")
        assert prompt.startswith("What is the date of birth of Patton Oswalt?")

    def test_graph_prompt_hides_the_name(self):
        prompt = build_prompt("graph", "Patton Oswalt", "(X, starring, Y)", "birthDate")
        assert "Patton Oswalt" not in prompt and "(X, starring, Y)" in prompt

    def test_readable_iri(self):
        assert readable("<http://dbpedia.org/resource/Patton_Oswalt>", {}) == "Patton Oswalt"


def test_unparsed_reply_falls_back_to_the_training_median():
    train = pd.DataFrame({"attribute": ["x"] * 3, "value": [1.0, 2.0, 9.0]})
    direct = pd.DataFrame({"arm": ["name"] * 2, "attribute": ["x"] * 2, "value": [2.0, 4.0],
                           "parsed": [np.nan, 4.0]})
    scores = score_direct(direct, train)
    assert scores.loc[0, "MAE"] == pytest.approx(0.0)
    assert scores.loc[0, "parse_rate"] == pytest.approx(0.5)
