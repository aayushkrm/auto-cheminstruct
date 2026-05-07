"""Tests for core data models."""

import pytest
from uuid import uuid4

from src.data.models import (
    ChemicalEntity,
    ReactionHypothesis,
    ReactionType,
    VerificationResult,
    VerificationStatus,
    ReflectionTrace,
    FailureCategory,
    PreferencePair,
    ReactionConditions,
    Solvent,
    ComputedProperties,
    AgentMessage,
    SessionState,
    PipelineStatus,
)


class TestChemicalEntity:
    def test_valid_smiles(self):
        entity = ChemicalEntity(smiles="c1ccccc1")
        assert entity.smiles == "c1ccccc1"

    def test_empty_smiles_raises(self):
        with pytest.raises(ValueError, match="SMILES string cannot be empty"):
            ChemicalEntity(smiles="")

    def test_whitespace_smiles_raises(self):
        with pytest.raises(ValueError, match="SMILES string cannot be empty"):
            ChemicalEntity(smiles="   ")

    def test_all_fields(self):
        entity = ChemicalEntity(
            smiles="CCO",
            inchi="InChI=1S/C2H6O/c1-2-3/h3H,2H2,1H3",
            selfies="[C][C][O]",
            name="ethanol",
            molecular_weight=46.07,
            formula="C2H6O",
        )
        assert entity.name == "ethanol"
        assert entity.molecular_weight == 46.07


class TestReactionHypothesis:
    def test_create_minimal(self):
        hyp = ReactionHypothesis(
            session_id=uuid4(),
            reactants=[ChemicalEntity(smiles="CCO")],
            products=[ChemicalEntity(smiles="CC=O")],
        )
        assert hyp.reaction_type == ReactionType.OTHER
        assert len(hyp.reactants) == 1

    def test_empty_reactants_raises(self):
        with pytest.raises(ValueError):
            ReactionHypothesis(
                session_id=uuid4(),
                reactants=[],
                products=[ChemicalEntity(smiles="CCO")],
            )

    def test_yield_estimate_validation(self):
        hyp = ReactionHypothesis(
            session_id=uuid4(),
            reactants=[ChemicalEntity(smiles="CCO")],
            products=[ChemicalEntity(smiles="CC=O")],
            yield_estimate=85.0,
        )
        assert hyp.yield_estimate == 85.0

    def test_yield_estimate_out_of_range(self):
        with pytest.raises(ValueError):
            ReactionHypothesis(
                session_id=uuid4(),
                reactants=[ChemicalEntity(smiles="CCO")],
                products=[ChemicalEntity(smiles="CC=O")],
                yield_estimate=150.0,
            )

    def test_full_conditions(self):
        hyp = ReactionHypothesis(
            session_id=uuid4(),
            reactants=[ChemicalEntity(smiles="CCO")],
            products=[ChemicalEntity(smiles="CC=O")],
            conditions=ReactionConditions(
                temperature_celsius=80.0,
                solvent=Solvent.THF,
                catalyst="Pd/C",
                time_hours=24.0,
            ),
            reaction_type=ReactionType.OXIDATION,
        )
        assert hyp.conditions.solvent == Solvent.THF
        assert hyp.conditions.temperature_celsius == 80.0


class TestVerificationResult:
    def test_default_pending(self, sample_hypothesis):
        result = VerificationResult(hypothesis_id=sample_hypothesis.id)
        assert result.status == VerificationStatus.PENDING
        assert not result.smiles_valid

    def test_passed_status(self, sample_hypothesis):
        result = VerificationResult(
            hypothesis_id=sample_hypothesis.id,
            status=VerificationStatus.PASSED,
            smiles_valid=True,
            valence_valid=True,
            steric_valid=True,
            energy_valid=True,
        )
        assert result.status == VerificationStatus.PASSED

    def test_computed_properties(self, sample_hypothesis):
        props = ComputedProperties(
            sa_score=3.5,
            logp=2.1,
            total_energy_hartree=-100.5,
        )
        result = VerificationResult(
            hypothesis_id=sample_hypothesis.id,
            computed_properties=props,
        )
        assert result.computed_properties.sa_score == 3.5
        assert result.computed_properties.total_energy_hartree == -100.5


class TestReflectionTrace:
    def test_create_with_confidence(self, sample_hypothesis):
        trace = ReflectionTrace(
            hypothesis_id=sample_hypothesis.id,
            verification_result_id=uuid4(),
            failure_categories=[FailureCategory.STERIC_HINDRANCE],
            causal_explanation="Steric effects prevent reaction.",
            confidence=0.8,
        )
        assert trace.failure_categories == [FailureCategory.STERIC_HINDRANCE]
        assert trace.confidence == 0.8

    def test_confidence_bounds(self, sample_hypothesis):
        with pytest.raises(ValueError):
            ReflectionTrace(
                hypothesis_id=sample_hypothesis.id,
                verification_result_id=uuid4(),
                causal_explanation="test",
                confidence=1.5,
            )


class TestPreferencePair:
    def test_create_pair(self, sample_hypothesis):
        pair = PreferencePair(
            prompt="Test prompt",
            chosen="Chosen response",
            rejected="Rejected response",
            chosen_hypothesis_id=sample_hypothesis.id,
            rejected_hypothesis_id=sample_hypothesis.id,
            reaction_type=ReactionType.CROSS_COUPLING,
        )
        assert pair.prompt == "Test prompt"
        assert pair.quality_score == 0.5


class TestAgentMessage:
    def test_create_message(self):
        msg = AgentMessage(
            source="hypothesis",
            target="verification",
            msg_type="hypothesis_generated",
            payload={"data": "test"},
            session_id=uuid4(),
        )
        assert msg.source == "hypothesis"
        assert msg.priority == 3

    def test_priority_bounds(self):
        with pytest.raises(ValueError):
            AgentMessage(
                source="test",
                target="test",
                msg_type="test",
                payload={},
                session_id=uuid4(),
                priority=6,
            )


class TestSessionState:
    def test_create_session(self):
        session = SessionState(config_hash="abc123")
        assert session.status == PipelineStatus.IDLE
        assert session.hypotheses_generated == 0

    def test_track_progress(self):
        session = SessionState(config_hash="abc123")
        session.hypotheses_generated = 50
        session.hypotheses_passed = 30
        assert session.hypotheses_generated == 50
