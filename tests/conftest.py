"""Shared pytest fixtures for Auto-ChemInstruct tests."""

from unittest.mock import MagicMock
from uuid import uuid4

import pytest

from src.data.models import (
    ChemicalEntity,
    ComputedProperties,
    ReactionConditions,
    ReactionHypothesis,
    ReactionType,
    ReflectionTrace,
    FailureCategory,
    VerificationResult,
    VerificationStatus,
    PreferencePair,
    SessionState,
    PipelineStatus,
)


@pytest.fixture
def session_id():
    return uuid4()


@pytest.fixture
def sample_smiles_reactants():
    return ["c1ccccc1Br", "B1OC(C)(C)C(C)(C)O1"]


@pytest.fixture
def sample_smiles_products():
    return ["c1ccccc1-c1ccccc1"]


@pytest.fixture
def sample_hypothesis(session_id, sample_smiles_reactants, sample_smiles_products):
    return ReactionHypothesis(
        session_id=session_id,
        reactants=[ChemicalEntity(smiles=s) for s in sample_smiles_reactants],
        products=[ChemicalEntity(smiles=s) for s in sample_smiles_products],
        reaction_type=ReactionType.CROSS_COUPLING,
        conditions=ReactionConditions(
            temperature_celsius=80.0,
            catalyst="Pd(PPh3)4",
        ),
        yield_estimate=85.0,
        rationale="Suzuki coupling between bromobenzene and phenylboronic acid pinacol ester",
    )


@pytest.fixture
def passed_verification(sample_hypothesis):
    return VerificationResult(
        hypothesis_id=sample_hypothesis.id,
        status=VerificationStatus.PASSED,
        smiles_valid=True,
        valence_valid=True,
        steric_valid=True,
        energy_valid=True,
        computed_properties=ComputedProperties(
            sa_score=3.5,
            qed=0.6,
            logp=2.1,
        ),
    )


@pytest.fixture
def failed_verification(sample_hypothesis):
    return VerificationResult(
        hypothesis_id=sample_hypothesis.id,
        status=VerificationStatus.FAILED,
        smiles_valid=True,
        valence_valid=True,
        steric_valid=False,
        energy_valid=False,
        errors=["Steric clash between atoms", "SA score outside range"],
    )


@pytest.fixture
def sample_reflection(sample_hypothesis, failed_verification):
    return ReflectionTrace(
        hypothesis_id=sample_hypothesis.id,
        verification_result_id=failed_verification.id,
        failure_categories=[FailureCategory.STERIC_HINDRANCE],
        primary_cause="Bulky tert-butyl group blocks palladium approach",
        causal_explanation="The tert-butyl substituent on the boronic ester creates significant steric hindrance...",
        chemical_principles=["Steric effects", "Oxidative addition"],
        fix_suggestion="Replace tert-butyl ester with pinacol ester",
        confidence=0.85,
    )


@pytest.fixture
def mock_llm():
    """Mock LLM that returns valid JSON responses."""
    llm = MagicMock()
    llm.invoke.return_value = MagicMock()
    llm.invoke.return_value.content = '{"reaction_type": "cross_coupling", "reactant_smiles": ["c1ccccc1Br", "c1ccccc1B(O)O"], "product_smiles": ["c1ccccc1-c1ccccc1"], "solvent": "THF", "catalyst": "Pd(PPh3)4", "temperature_celsius": 80.0, "yield_estimate": 85.0, "rationale": "Suzuki coupling"}'
    return llm


@pytest.fixture
def mock_llm_reflection():
    """Mock LLM that returns valid reflection JSON."""
    llm = MagicMock()
    llm.invoke.return_value = MagicMock()
    llm.invoke.return_value.content = '''{
        "failure_categories": ["steric_hindrance"],
        "primary_cause": "Bulky group blocks reactive site",
        "causal_explanation": "The large substituent prevents the nucleophile from approaching the electrophilic carbon. The transition state would require an impossible geometry.",
        "chemical_principles": ["Steric hindrance", "SN2 mechanism"],
        "fix_suggestion": "Use a less bulky electrophile or switch to SN1 conditions",
        "confidence": 0.85
    }'''
    return llm


@pytest.fixture
def sample_session():
    return SessionState(
        config_hash="abc123",
        status=PipelineStatus.IDLE,
    )


@pytest.fixture
def sample_preference_pair(sample_hypothesis, sample_reflection):
    return PreferencePair(
        prompt="Propose a cross_coupling reaction...",
        chosen="SUCCESSFUL REACTION: Reactants: c1ccccc1Br...",
        rejected="FAILED REACTION: Reactants: c1ccccc1Br... FAILURE ANALYSIS: ...",
        chosen_hypothesis_id=sample_hypothesis.id,
        rejected_hypothesis_id=sample_hypothesis.id,
        reflection_trace_id=sample_reflection.id,
        reaction_type=ReactionType.CROSS_COUPLING,
        quality_score=0.75,
    )
