"""Tests for agent modules (hypothesis, verification, reflection, compilation)."""

import pytest
from unittest.mock import MagicMock
from uuid import uuid4

from src.agents.hypothesis_agent import HypothesisGenerationAgent
from src.agents.verification_agent import VerificationAgent
from src.agents.reflection_agent import ReflectionAgent
from src.agents.compilation_agent import CompilationAgent
from src.data.models import (
    ReactionHypothesis,
    VerificationStatus,
    ReactionType,
    ChemicalEntity,
    ReactionConditions,
)


class TestHypothesisAgent:
    def test_agent_creation(self, mock_llm):
        agent = HypothesisGenerationAgent(
            llm=mock_llm,
            temperature=0.9,
            num_generations_per_prompt=3,
        )
        assert agent.temperature == 0.9
        assert agent.num_generations_per_prompt == 3

    def test_generate_hypotheses(self, mock_llm, session_id):
        agent = HypothesisGenerationAgent(
            llm=mock_llm,
            num_generations_per_prompt=3,
        )
        hypotheses = agent.generate(
            session_id=session_id,
            num_hypotheses=3,
        )
        assert len(hypotheses) > 0
        assert all(isinstance(h, ReactionHypothesis) for h in hypotheses)
        assert all(h.session_id == session_id for h in hypotheses)

    def test_parse_response_single(self, mock_llm, session_id):
        agent = HypothesisGenerationAgent(llm=mock_llm)
        hypotheses = agent._parse_response(
            response='{"reaction_type": "oxidation", "reactant_smiles": ["CCO"], "product_smiles": ["CC=O"], "rationale": "Oxidation of ethanol"}',
            session_id=session_id,
            prompt_used="test",
        )
        assert len(hypotheses) == 1
        assert hypotheses[0].reaction_type == ReactionType.OXIDATION

    def test_parse_response_json_block(self, mock_llm, session_id):
        agent = HypothesisGenerationAgent(llm=mock_llm)
        response = '''```json
{"reaction_type": "cross_coupling", "reactant_smiles": ["c1ccccc1Br"], "product_smiles": ["c1ccccc1-c1ccccc1"], "rationale": "Suzuki"}
```'''
        hypotheses = agent._parse_response(
            response=response,
            session_id=session_id,
            prompt_used="test",
        )
        assert len(hypotheses) == 1

    def test_invalid_response_empty(self, mock_llm, session_id):
        agent = HypothesisGenerationAgent(llm=mock_llm)
        hypotheses = agent._parse_response(
            response="not json at all",
            session_id=session_id,
            prompt_used="test",
        )
        assert len(hypotheses) == 0


class TestVerificationAgent:
    def test_verify_valid_molecule(self, sample_hypothesis):
        agent = VerificationAgent(enable_xtb=False)
        result = agent.verify(sample_hypothesis)
        assert result.smiles_valid
        assert result.valence_valid
        assert result.status in (VerificationStatus.PASSED, VerificationStatus.FAILED)

    def test_verify_invalid_smiles(self):
        agent = VerificationAgent(enable_xtb=False)
        hypothesis = ReactionHypothesis(
            session_id=uuid4(),
            reactants=[ChemicalEntity(smiles="INVALID")],
            products=[ChemicalEntity(smiles="CCO")],
        )
        result = agent.verify(hypothesis)
        assert result.status == VerificationStatus.FAILED
        assert not result.smiles_valid

    def test_verify_batch(self, sample_hypothesis):
        agent = VerificationAgent(enable_xtb=False)
        hypotheses = [sample_hypothesis, sample_hypothesis]
        results = agent.verify_batch(hypotheses)
        assert len(results) == 2
        assert all(r.smiles_valid for r in results)

    def test_xtb_disabled(self, sample_hypothesis):
        agent = VerificationAgent(enable_xtb=False)
        result = agent.verify(sample_hypothesis)
        assert result.computed_properties.total_energy_hartree is None


class TestReflectionAgent:
    def test_agent_creation(self, mock_llm_reflection):
        agent = ReflectionAgent(llm=mock_llm_reflection, temperature=0.3)
        assert agent.temperature == 0.3

    def test_reflect_on_failure(self, mock_llm_reflection, sample_hypothesis, failed_verification):
        agent = ReflectionAgent(llm=mock_llm_reflection)
        trace = agent.reflect(
            hypothesis=sample_hypothesis,
            verification=failed_verification,
        )
        assert trace is not None
        assert trace.hypothesis_id == sample_hypothesis.id
        assert len(trace.failure_categories) > 0

    def test_no_reflect_on_pass(self, mock_llm_reflection, sample_hypothesis, passed_verification):
        agent = ReflectionAgent(llm=mock_llm_reflection)
        trace = agent.reflect(
            hypothesis=sample_hypothesis,
            verification=passed_verification,
        )
        assert trace is None


class TestCompilationAgent:
    def test_compile_pairs(
        self,
        sample_hypothesis,
        passed_verification,
        failed_verification,
        sample_reflection,
    ):
        agent = CompilationAgent(output_format="dpo", train_split=0.8, val_split=0.1, test_split=0.1)

        result = agent.compile(
            hypotheses=[sample_hypothesis],
            verification_results=[passed_verification, failed_verification],
            reflection_traces=[sample_reflection],
        )

        assert "train" in result
        assert "val" in result
        assert "test" in result
        assert "metadata" in result
        metadata = result["metadata"]
        assert metadata["total_hypotheses"] >= 1

    def test_deduplicate(self):
        agent = CompilationAgent(deduplicate=True)
        from src.data.models import PreferencePair, ReactionType

        pairs = [
            PreferencePair(
                prompt="test",
                chosen="chosen",
                rejected="rejected",
                chosen_hypothesis_id=uuid4(),
                rejected_hypothesis_id=uuid4(),
                reaction_type=ReactionType.OXIDATION,
            ),
            PreferencePair(
                prompt="test",
                chosen="chosen",
                rejected="rejected",
                chosen_hypothesis_id=uuid4(),
                rejected_hypothesis_id=uuid4(),
                reaction_type=ReactionType.OXIDATION,
            ),
        ]
        deduped = agent._deduplicate(pairs)
        assert len(deduped) == 1

    def test_build_metadata(self):
        agent = CompilationAgent()
        splits = {"train": [], "val": [], "test": []}
        metadata = agent._build_metadata(
            splits=splits,
            hypotheses=[],
            verification_results=[],
            reflection_traces=[],
        )
        assert metadata["total_hypotheses"] == 0
        assert metadata["total_pairs"] == 0

    def test_empty_compile(self):
        agent = CompilationAgent()
        result = agent.compile(
            hypotheses=[],
            verification_results=[],
            reflection_traces=[],
        )
        assert result["metadata"]["total_pairs"] == 0
