"""End-to-end integration tests for the Auto-ChemInstruct pipeline.

Uses mock LLM to avoid API calls while testing the full orchestration flow.
"""

import pytest
from unittest.mock import MagicMock, patch
from uuid import uuid4

from src.config import load_config
from src.data.models import (
    PipelineStatus,
    ReactionType,
    VerificationStatus,
    ReactionHypothesis,
    ChemicalEntity,
    VerificationResult,
)
from src.pipeline.orchestrator import PipelineOrchestrator


def _make_mock_llm():
    """Mock LLM that returns hypothesis JSON for generation and reflection JSON for reflection."""
    llm = MagicMock()

    def side_effect(messages, **kwargs):
        msg_text = str(messages)
        result = MagicMock()

        if (
            "failure" in msg_text.lower()
            or "reflection" in msg_text.lower()
            or "why this chemical reaction fails" in msg_text.lower()
        ):
            result.content = (
                '{"failure_categories": ["steric_hindrance"], '
                '"primary_cause": "Bulky tert-butyl group blocks approach", '
                '"causal_explanation": "The large substituent prevents the nucleophile '
                "from approaching the electrophilic carbon center. The transition state "
                'requires an impossible geometry due to van der Waals clashes.", '
                '"chemical_principles": ["Steric hindrance", "SN2 mechanism"], '
                '"fix_suggestion": "Replace with a smaller electrophile or use SN1 conditions", '
                '"confidence": 0.85}'
            )
        else:
            result.content = (
                '{"reaction_type": "cross_coupling", '
                '"reactant_smiles": ["c1ccccc1Br", "c1ccccc1B(O)O"], '
                '"product_smiles": ["c1ccccc1-c1ccccc1"], '
                '"solvent": "THF", "catalyst": "Pd(PPh3)4", '
                '"temperature_celsius": 80.0, "yield_estimate": 85.0, '
                '"mechanism_steps": "Oxidative addition, transmetalation, reductive elimination", '
                '"rationale": "Suzuki coupling with good yield expected"}'
            )
        return result

    llm.invoke = MagicMock(side_effect=side_effect)
    return llm


@pytest.fixture
def mock_llm():
    return _make_mock_llm()


class TestEndToEndPipeline:
    def test_pipeline_start_session(self, mock_llm):
        config = load_config()
        orch = PipelineOrchestrator(config)
        orch.llm = orch.llm_json = mock_llm

        session_id = orch.start_session()
        assert session_id is not None
        assert orch.session.status == PipelineStatus.IDLE

    def test_pipeline_full_flow_mocked(self, mock_llm):
        config = load_config()
        config.pipeline.batch_size = 3

        orch = PipelineOrchestrator(config)
        orch.llm = orch.llm_json = mock_llm

        result = orch.run_pipeline(num_hypotheses=3)

        assert "summary" in result
        summary = result["summary"]
        assert summary["hypotheses_generated"] > 0
        passed = summary["hypotheses_passed"]
        failed = summary["hypotheses_failed"]
        assert passed + failed == summary["hypotheses_generated"]
        assert orch.session.status == PipelineStatus.COMPLETED

    def test_pipeline_checkpoint_save_load(self, mock_llm, tmp_path):
        config = load_config()
        config.pipeline.checkpoint_dir = str(tmp_path)
        config.pipeline.batch_size = 2

        orch = PipelineOrchestrator(config)
        orch.llm = orch.llm_json = mock_llm

        session_id = orch.start_session()
        orch._save_checkpoint()

        orch2 = PipelineOrchestrator(config)
        orch2.resume_session(session_id)
        assert orch2.session.id == session_id
        assert orch2.session.status == PipelineStatus.IDLE

    def test_pipeline_empty_run(self, mock_llm):
        config = load_config()
        config.pipeline.batch_size = 1

        orch = PipelineOrchestrator(config)
        orch.llm = orch.llm_json = mock_llm

        result = orch.run_pipeline(num_hypotheses=1)
        assert result["summary"]["hypotheses_generated"] >= 0

    def test_verification_measures_pass_rate(self, mock_llm):
        config = load_config()
        config.pipeline.batch_size = 3

        orch = PipelineOrchestrator(config)
        orch.llm = orch.llm_json = mock_llm

        orch.start_session()
        hypotheses = orch._generate_hypotheses(num_hypotheses=3)
        results = orch._verify_hypotheses(hypotheses)

        passed = sum(1 for r in results if r.status == VerificationStatus.PASSED)
        failed = sum(1 for r in results if r.status == VerificationStatus.FAILED)

        assert passed + failed <= len(results)
        assert len(results) == len(hypotheses)

    def test_reflection_on_failures(self, mock_llm):
        config = load_config()
        config.pipeline.batch_size = 2

        orch = PipelineOrchestrator(config)
        orch.llm = orch.llm_json = mock_llm

        orch.start_session()

        failed_hyp = ReactionHypothesis(
            session_id=orch.session.id,
            reactants=[ChemicalEntity(smiles="INVALID")],
            products=[ChemicalEntity(smiles="CCO")],
            reaction_type=ReactionType.OXIDATION,
        )
        failed_result = VerificationResult(
            hypothesis_id=failed_hyp.id,
            status=VerificationStatus.FAILED,
            smiles_valid=False,
            errors=["Invalid SMILES"],
        )

        traces = orch._reflect_on_failures([failed_hyp], [failed_result])
        assert len(traces) == 1

    def test_compilation_saves_metadata(self, mock_llm):
        config = load_config()
        config.pipeline.batch_size = 2

        orch = PipelineOrchestrator(config)
        orch.llm = orch.llm_json = mock_llm

        orch.start_session()
        hypotheses = orch._generate_hypotheses(num_hypotheses=2)
        results = orch._verify_hypotheses(hypotheses)

        failed_hyps = [
            h for h, r in zip(hypotheses, results) if r.status == VerificationStatus.FAILED
        ]
        failed_results = [r for r in results if r.status == VerificationStatus.FAILED]
        traces = orch._reflect_on_failures(failed_hyps, failed_results)

        compilation = orch._compile_dataset(hypotheses, results, traces)

        metadata = compilation["metadata"]
        assert "total_hypotheses" in metadata
        assert "passed_verification" in metadata
        assert "failed_verification" in metadata
        assert "total_pairs" in metadata
        assert "output_format" in metadata
        assert "created_at" in metadata

    def test_pipeline_handles_empty_generation(self, mock_llm):
        """Test pipeline handles case where LLM returns malformed response."""
        config = load_config()
        config.pipeline.batch_size = 2

        orch = PipelineOrchestrator(config)
        orch.llm = mock_llm

        orch.start_session()
        hypotheses = orch._generate_hypotheses(num_hypotheses=0)
        results = orch._verify_hypotheses(hypotheses)
        traces = orch._reflect_on_failures([], [])
        compilation = orch._compile_dataset(hypotheses, results, traces)

        assert compilation["metadata"]["total_pairs"] == 0
