"""Tests for standalone quality scoring module."""

import json
from uuid import uuid4

import pytest
from src.compilation.quality import (
    QualityScores,
    QualityReport,
    compute_quality_scores,
    score_all_pairs,
    export_quality_report,
)
from src.data.models import PreferencePair, ReactionType


UUID_1 = uuid4()
UUID_2 = uuid4()


class TestQualityScores:
    def test_default_scores_are_zero(self):
        scores = QualityScores(pair_id="test-1")
        assert scores.composite == 0.0
        assert scores.structural_validity == 0.0

    def test_to_dict(self):
        scores = QualityScores(pair_id="abc", composite=0.75, structural_validity=0.9)
        d = scores.to_dict()
        assert d["pair_id"] == "abc"
        assert d["composite"] == 0.75
        assert d["structural_validity"] == 0.9

    def test_rounding_in_dict(self):
        scores = QualityScores(pair_id="x", composite=0.123456789)
        d = scores.to_dict()
        assert d["composite"] == 0.1235


class TestComputeQualityScores:
    def make_pair(self, chosen_text: str, rejected_text: str, **kwargs) -> PreferencePair:
        metadata = {
            "chosen_yield": kwargs.get("yield_val", 85.0),
            "reflection_confidence": kwargs.get("confidence", 0.7),
            "failure_categories": kwargs.get("categories", ["steric_hindrance"]),
            "rejected_errors": kwargs.get("errors", ["valence error"]),
        }
        rtype_str = kwargs.get("rtype", "other")
        try:
            rtype = ReactionType(rtype_str)
        except ValueError:
            rtype = ReactionType.OTHER
        return PreferencePair(
            id=uuid4(),
            prompt="Test prompt",
            chosen=chosen_text,
            rejected=rejected_text,
            chosen_hypothesis_id=UUID_1,
            rejected_hypothesis_id=UUID_2,
            reaction_type=rtype,
            quality_score=0.5,
            metadata=metadata,
        )

    def test_valid_pair_scores(self):
        chosen = json.dumps({"Reactants": "CCO", "Products": "CCOC(C)=O"})
        rejected = json.dumps({"Reactants": "XXXX", "Products": "YYYY"})
        pair = self.make_pair(chosen, rejected, rtype="condensation")
        scores = compute_quality_scores(pair)
        assert scores.composite > 0.2
        assert scores.reaction_specificity > 0.5

    def test_named_reaction_scores_higher_than_other(self):
        chosen = json.dumps({"Reactants": "CCO", "Products": "CCBr"})
        rejected = json.dumps({"Reactants": "C", "Products": "C"})
        named = self.make_pair(chosen, rejected, rtype="condensation")
        other = self.make_pair(chosen, rejected, rtype="other")
        named_scores = compute_quality_scores(named)
        other_scores = compute_quality_scores(other)
        assert named_scores.reaction_specificity > other_scores.reaction_specificity

    def test_high_yield_scores_higher(self):
        chosen = json.dumps({"Reactants": "CCO", "Products": "CCC"})
        rejected = json.dumps({"Reactants": "XX", "Products": "XX"})
        pair = self.make_pair(chosen, rejected, yield_val=95.0)
        scores = compute_quality_scores(pair)
        assert scores.yield_differential > 0.9

    def test_low_yield_scores_low(self):
        chosen = json.dumps({"Reactants": "CCO", "Products": "CCC"})
        rejected = json.dumps({"Reactants": "XX", "Products": "XX"})
        pair = self.make_pair(chosen, rejected, yield_val=15.0)
        scores = compute_quality_scores(pair)
        assert scores.yield_differential < 0.3


class TestScoreAllPairs:
    def make_pair(self, i: int) -> PreferencePair:
        return PreferencePair(
            id=uuid4(),
            prompt=f"Test {i}",
            chosen=json.dumps({"Reactants": "CCO", "Products": "CCBr"}),
            rejected=json.dumps({"Reactants": "C", "Products": "C"}),
            chosen_hypothesis_id=UUID_1,
            rejected_hypothesis_id=UUID_2,
            reaction_type=ReactionType.OTHER,
            quality_score=0.5,
            metadata={"chosen_yield": 85.0, "reflection_confidence": 0.7},
        )

    def test_empty_list(self):
        report = score_all_pairs([])
        assert report.total_pairs == 0
        assert report.mean_composite == 0.0

    def test_single_pair_report(self):
        pairs = [self.make_pair(1)]
        report = score_all_pairs(pairs)
        assert report.total_pairs == 1
        assert 0.0 <= report.mean_composite <= 1.0

    def test_multiple_pairs_statistics(self):
        pairs = [self.make_pair(i) for i in range(3)]
        report = score_all_pairs(pairs)
        assert report.total_pairs == 3
        assert report.min_composite <= report.max_composite

    def test_summary_table(self):
        pairs = [self.make_pair(1)]
        report = score_all_pairs(pairs)
        table = report.summary_table()
        assert "Structural Validity" in table
        assert "Composite" in table
        assert "Range" in table


class TestExportQualityReport:
    def test_export_to_json(self, tmp_path):
        report = QualityReport(
            total_pairs=1,
            mean_composite=0.5,
            mean_validity=0.8,
            mean_druglikeness=0.4,
            mean_reflection=0.6,
            mean_yield=0.7,
            mean_diversity=0.3,
            mean_specificity=0.2,
            min_composite=0.5,
            max_composite=0.5,
        )
        path = tmp_path / "quality.json"
        export_quality_report(report, path)
        assert path.exists()
        data = json.loads(path.read_text())
        assert data["total_pairs"] == 1
        assert data["mean_composite"] == 0.5
