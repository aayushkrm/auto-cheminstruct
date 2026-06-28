"""Tests for CARL reasoning chain: 4-step parallel steric/electronic/thermo analysis + synthesis."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.carl.chain import (
    CARLChain,
    CARLReflectionAgent,
    CARLResult,
    CausalSynthesis,
    ElectronicAnalysis,
    StericAnalysis,
    ThermodynamicAnalysis,
    _format_conditions,
    _format_products,
    _format_reactants,
    make_electronic_fn,
    make_steric_fn,
    make_synthesis_fn,
    make_thermo_fn,
)

# ── Formatting helpers ──


class TestFormatting:
    def test_format_reactants_from_dict(self):
        result = _format_reactants({"reactants": ["CC", "CC=O"]})
        assert "CC" in result
        assert "CC=O" in result

    def test_format_products_from_dict(self):
        result = _format_products({"products": ["CCC"]})
        assert "CCC" in result

    def test_format_conditions_from_dict(self):
        result = _format_conditions({"conditions": {"temperature_c": 80, "solvent": "THF"}})
        assert "80°C" in result
        assert "THF" in result

    def test_format_empty_conditions(self):
        assert _format_conditions({}) == "not specified"


# ── Step function tests ──


class TestStepFunctions:
    def test_steric_fn_returns_valid(self):
        result = make_steric_fn()({})
        assert isinstance(result, StericAnalysis)
        assert result.confidence > 0
        assert isinstance(result.steric_clashes, list)
        assert len(result.steric_clashes) > 0

    def test_electronic_fn_returns_valid(self):
        result = make_electronic_fn()({})
        assert isinstance(result, ElectronicAnalysis)
        assert result.confidence > 0

    def test_thermo_fn_returns_valid(self):
        result = make_thermo_fn()({})
        assert isinstance(result, ThermodynamicAnalysis)
        assert result.confidence > 0

    def test_synthesis_fn_integrates_steps(self):
        func = make_synthesis_fn()
        inp = {
            "steric": StericAnalysis(steric_clashes=["clash"], accessible_transition_state=False),
            "electronic": ElectronicAnalysis(homo_lumo_compatible=False),
            "thermodynamic": ThermodynamicAnalysis(enthalpy_favorable=False),
            "reactants": "CC",
            "products": "CCC",
            "reaction_type": "esterification",
        }
        result = func(inp)
        assert isinstance(result, CausalSynthesis)
        assert result.primary_cause != "Unknown"
        assert len(result.chemical_principles) > 0


# ── CARL Chain integration tests ──


class TestCARLChain:
    @pytest.fixture
    def chain(self):
        return CARLChain()

    @pytest.fixture
    def hypothesis(self):
        return {
            "reactants": ["CC", "CC=O"],
            "products": ["CCC(O)C"],
            "reaction_type": "aldol_condensation",
            "conditions": {"temperature_c": 25, "solvent": "EtOH"},
        }

    def test_chain_runs_all_steps(self, chain, hypothesis):
        result = chain.run(hypothesis, errors=["steric clash at C2"])
        assert isinstance(result, CARLResult)
        assert result.steric is not None
        assert result.electronic is not None
        assert result.thermodynamic is not None
        assert result.synthesis is not None
        assert result.overall_confidence > 0
        assert result.synthesis.primary_cause != "Unknown"

    def test_chain_produces_explanation(self, chain, hypothesis):
        result = chain.run(hypothesis, errors=["frontier orbital mismatch"])
        assert len(result.synthesis.causal_explanation) > 50
        assert len(result.synthesis.fix_suggestion) > 10

    def test_chain_errors_captured(self):
        chain = CARLChain(
            steric_fn=lambda _: (_ for _ in ()).__next__(),  # raises StopIteration
        )
        result = chain.run({"reactants": [], "products": [], "conditions": {}})
        assert len(result.errors) >= 1
        assert "steric" in str(result.errors[0]).lower()

    def test_chain_deterministic(self):
        hyp = {"reactants": ["CC"], "products": ["CCC"], "conditions": {}}
        chain = CARLChain()
        r1 = chain.run(hyp)
        r2 = chain.run(hyp)
        assert r1.synthesis.primary_cause == r2.synthesis.primary_cause

    def test_chain_with_reactionhypothesis(self):
        from uuid import uuid4

        from src.data.models import (
            ChemicalEntity,
            ReactionConditions,
            ReactionHypothesis,
            ReactionType,
        )

        hyp = ReactionHypothesis(
            session_id=uuid4(),
            reactants=[ChemicalEntity(smiles="CC")],
            products=[ChemicalEntity(smiles="CCC")],
            reaction_type=ReactionType.ESTERIFICATION,
            conditions=ReactionConditions(),
        )
        chain = CARLChain()
        result = chain.run(hyp, errors=["test error"])
        assert result.synthesis is not None
        assert "esterification" in str(hyp.reaction_type).lower()


# ── CARL Reflection Agent tests ──


class TestCARLReflectionAgent:
    @pytest.fixture
    def agent(self):
        return CARLReflectionAgent(enabled=True)

    def test_reflect_returns_result(self, agent):
        hyp = {"reactants": ["CC"], "products": ["CCC"], "conditions": {}}
        vresult = MagicMock(errors=["test error"], status=MagicMock(value="FAILED"))
        result = agent.reflect(hyp, vresult)
        assert result is not None
        assert isinstance(result, CARLResult)

    def test_reflect_disabled(self):
        agent = CARLReflectionAgent(enabled=False)
        result = agent.reflect({}, MagicMock(errors=["e"]))
        assert result is None

    def test_reflect_batch_filters_passed(self, agent):
        hyp1 = {"reactants": ["CC"], "products": ["CCC"], "conditions": {}}
        hyp2 = {"reactants": ["NN"], "products": ["NNN"], "conditions": {}}

        vfail = MagicMock(errors=["e"], status=MagicMock(value="FAILED"))
        vpass = MagicMock(errors=[], status=MagicMock(value="PASSED"))

        results = agent.reflect_batch(
            [hyp1, hyp2],
            [vfail, vpass],
        )
        assert len(results) == 1

    def test_reflect_batch_empty(self, agent):
        results = agent.reflect_batch([], [])
        assert results == []
