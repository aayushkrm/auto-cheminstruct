"""Orchestrator Agent — central coordination hub for the Auto-ChemInstruct pipeline.

Manages session lifecycle, agent lifecycle, task queuing, inter-agent message routing,
state persistence, and the overall pipeline state machine.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from datetime import datetime
from pathlib import Path
from typing import Optional
from uuid import UUID, uuid4

from loguru import logger

from src.config import AutoChemConfig
from src.data.models import (
    AgentMessage,
    PipelineStatus,
    PreferencePair,
    ReactionHypothesis,
    ReflectionTrace,
    SessionState,
    VerificationResult,
    VerificationStatus,
)
from src.exceptions import CheckpointError, PipelineError
from src.utils.llm_factory import create_llm


def _compute_iteration_temperature(
    iteration: int,
    total_iterations: int,
    config: AutoChemConfig,
) -> float:
    """Compute temperature for a bootstrap iteration using configured schedule."""
    from src.utils.temperature_scheduler import compute_temperature

    return compute_temperature(
        iteration=iteration,
        total_iterations=total_iterations,
        schedule=config.pipeline.temperature_schedule,
        max_temp=config.pipeline.temperature_max,
        min_temp=config.pipeline.temperature_min,
    )


class PipelineOrchestrator:
    """Central orchestrator managing the full Auto-ChemInstruct pipeline.

    Coordinates:
        HypothesisAgent → VerificationAgent → [ReflectionAgent] → CompilationAgent
    """

    def __init__(self, config: AutoChemConfig):
        self.config = config
        self.session: Optional[SessionState] = None
        self._db_path: Optional[Path] = None
        self.rag = None

        self._init_llm()
        self._init_rag()

    def _init_llm(self) -> None:
        """Initialize LLM client from config."""
        self.llm = create_llm(self.config)
        logger.info(
            "LLM initialized: provider={}, model={}",
            self.config.llm.provider,
            self.config.llm.model,
        )

    def _init_rag(self) -> None:
        """Initialize RAG system (vector store + chemical knowledge graph)."""
        if not self.config.rag.enabled:
            return

        try:
            from src.rag.chemical_rag import ChemicalRAG

            self.rag = ChemicalRAG(self.config)
            if self.rag.initialize():
                logger.info("RAG system initialized for pipeline enrichment")
            else:
                logger.warning("RAG initialization failed — continuing without retrieval")
                self.rag = None
        except Exception as e:
            logger.warning("RAG not available: {}", e)
            self.rag = None

    def start_session(self) -> UUID:
        """Start a new pipeline session.

        Returns:
            Session UUID.
        """
        session_id = uuid4()

        config_hash = hashlib.md5(
            json.dumps(self.config.model_dump(), sort_keys=True, default=str).encode()
        ).hexdigest()[:12]

        self.session = SessionState(
            id=session_id,
            config_hash=config_hash,
            status=PipelineStatus.IDLE,
            started_at=datetime.now(),
        )

        self._init_checkpoint()
        self._save_checkpoint()

        logger.info("Session started: id={}, config_hash={}", session_id, config_hash)
        return session_id

    def resume_session(self, session_id: UUID) -> bool:
        """Resume a previous session from checkpoint.

        Args:
            session_id: UUID of the session to resume.

        Returns:
            True if session was found and resumed.

        Raises:
            PipelineError: If session not found or checkpoint corrupted.
        """
        self.session = SessionState(id=session_id, config_hash="")
        self._init_checkpoint()
        self._load_checkpoint()

        if self.session.id != session_id:
            raise PipelineError(f"Checkpoint mismatch: expected {session_id}, got {self.session.id}")

        logger.info(
            "Session resumed: id={}, status={}, hypotheses_generated={}",
            session_id,
            self.session.status.value,
            self.session.hypotheses_generated,
        )
        return True

    def run_pipeline(
        self,
        num_hypotheses: int = 100,
        seed_prompts: list[str] | None = None,
        bootstrap_iterations: int = 1,
    ) -> dict:
        """Execute the full Auto-ChemInstruct pipeline with self-bootstrapping.

        The self-bootstrapping loop: generate → verify → reflect →
        accumulate learning → repeat with learned constraints.

        Args:
            num_hypotheses: Number of hypotheses to generate.
            seed_prompts: Optional custom prompts for generation.
            bootstrap_iterations: Number of self-bootstrapping iterations.
                Set >1 to enable the reflection feedback loop.
                Each iteration generates ~num_hypotheses/batch_size hypotheses
                per batch, using accumulated failure knowledge.

        Returns:
            Dict with compiled dataset splits and session summary.
        """
        if self.session is None:
            self.start_session()

        total_batches = max(1, num_hypotheses // self.config.pipeline.batch_size)
        self.session.total_batches = total_batches
        self.session.current_batch = 0
        self._save_checkpoint()

        logger.info(
            "Pipeline started: hypotheses={}, batch_size={}, batches={}, "
            "bootstrap_iterations={}",
            num_hypotheses,
            self.config.pipeline.batch_size,
            total_batches,
            bootstrap_iterations,
        )

        all_hypotheses: list[ReactionHypothesis] = []
        all_results: list[VerificationResult] = []
        all_traces: list[ReflectionTrace] = []
        learning_context = None
        context_prompt = None

        from src.data.models import LearningContext

        for iteration in range(bootstrap_iterations):
            iter_hypotheses = num_hypotheses if iteration == 0 else num_hypotheses

            iter_temperature = _compute_iteration_temperature(
                iteration=iteration,
                total_iterations=bootstrap_iterations,
                config=self.config,
            )

            logger.info(
                "Bootstrap iteration {}/{}: learning_context={}, temperature={:.3f}",
                iteration + 1,
                bootstrap_iterations,
                learning_context is not None,
                iter_temperature,
            )

            self.session.status = PipelineStatus.GENERATING
            self._save_checkpoint()

            new_hypotheses = self._generate_hypotheses(
                num_hypotheses=iter_hypotheses,
                seed_prompts=seed_prompts,
                learning_context=context_prompt,
                temperature=iter_temperature,
            )

            if not new_hypotheses:
                logger.warning("No hypotheses generated in iteration {}", iteration + 1)
                continue

            self.session.hypotheses_generated += len(new_hypotheses)
            self._save_checkpoint()

            self.session.status = PipelineStatus.VERIFYING
            self._save_checkpoint()

            new_results = self._verify_hypotheses(new_hypotheses)
            self.session.hypotheses_verified += len(new_results)
            self.session.hypotheses_passed += sum(
                1 for r in new_results if r.status == VerificationStatus.PASSED
            )
            self.session.hypotheses_failed += sum(
                1 for r in new_results if r.status == VerificationStatus.FAILED
            )
            self._save_checkpoint()

            all_hypotheses.extend(new_hypotheses)
            all_results.extend(new_results)

            # Index verified reactions into RAG for multi-hop retrieval
            self._index_to_rag(new_hypotheses, new_results)

            failed_hypotheses = [
                h for h, r in zip(new_hypotheses, new_results)
                if r.status == VerificationStatus.FAILED
            ]
            failed_results = [
                r for r in new_results
                if r.status == VerificationStatus.FAILED
            ]

            if failed_hypotheses:
                self.session.status = PipelineStatus.REFLECTING
                self._save_checkpoint()

                new_traces = self._reflect_on_failures(failed_hypotheses, failed_results)
                all_traces.extend(new_traces)
                self.session.reflections_generated += len(new_traces)
                self._save_checkpoint()

                if bootstrap_iterations > 1:
                    learning_context = self._accumulate_learning(
                        traces=new_traces,
                        results=new_results,
                        learning_context=learning_context,
                    )
                    if learning_context:
                        context_prompt = learning_context.build_context_prompt()
                        logger.debug(
                            "Learning context generated: {} chars, {} categories",
                            len(context_prompt),
                            len(learning_context.failure_categories),
                        )

        self.session.status = PipelineStatus.COMPILING
        self._save_checkpoint()

        compilation = self._compile_dataset(all_hypotheses, all_results, all_traces)
        self.session.pairs_compiled = compilation["metadata"]["total_pairs"]
        self._save_checkpoint()

        self.session.status = PipelineStatus.COMPLETED
        self.session.completed_at = datetime.now()
        self._save_checkpoint()

        logger.info(
            "Pipeline complete: {} hypotheses → {} passed, {} failed, {} reflections, {} pairs",
            len(all_hypotheses),
            self.session.hypotheses_passed,
            self.session.hypotheses_failed,
            len(all_traces),
            self.session.pairs_compiled,
        )

        return {
            "session_id": self.session.id,
            "hypotheses": all_hypotheses,
            "verification_results": all_results,
            "reflection_traces": all_traces,
            "compilation": compilation,
            "summary": {
                "hypotheses_generated": self.session.hypotheses_generated,
                "hypotheses_passed": self.session.hypotheses_passed,
                "hypotheses_failed": self.session.hypotheses_failed,
                "reflections_generated": self.session.reflections_generated,
                "pairs_compiled": self.session.pairs_compiled,
                "total_batches": self.session.total_batches,
                "completed_at": self.session.completed_at.isoformat() if self.session.completed_at else None,
            },
        }

    def _generate_hypotheses(
        self,
        num_hypotheses: int,
        seed_prompts: list[str] | None = None,
        learning_context: str | None = None,
        temperature: float | None = None,
    ) -> list[ReactionHypothesis]:
        """Run hypothesis generation agent.

        Args:
            learning_context: Optional prompt snippet with accumulated
                failure patterns from prior bootstrap iterations.
            temperature: Optional temperature override (for scheduling).
        """
        from src.agents.hypothesis_agent import HypothesisGenerationAgent

        agent = HypothesisGenerationAgent(
            llm=self.llm,
            temperature=temperature if temperature is not None else self.config.hypothesis_agent.temperature,
            top_p=self.config.hypothesis_agent.top_p,
            max_tokens=self.config.hypothesis_agent.max_tokens,
            num_generations_per_prompt=self.config.hypothesis_agent.num_generations_per_prompt,
        )

        all_hypotheses: list[ReactionHypothesis] = []
        batch_size = self.config.pipeline.batch_size

        for batch_start in range(0, num_hypotheses, batch_size):
            batch_n = min(batch_size, num_hypotheses - batch_start)

            # Enrich seed prompt with RAG context
            seed = seed_prompts[self.session.current_batch] if seed_prompts and self.session.current_batch < len(seed_prompts) else None
            if self.rag and seed:
                enriched = self.rag.enrich_prompt(
                    base_prompt=seed,
                    query=seed,
                    use_multi_hop=True,
                )
                if enriched != seed:
                    logger.debug("RAG-enriched prompt with multi-hop context")
                    seed = enriched

            batch_hypotheses = agent.generate(
                session_id=self.session.id,
                num_hypotheses=batch_n,
                seed_prompt=seed,
                learning_context=learning_context,
            )
            all_hypotheses.extend(batch_hypotheses)
            self.session.current_batch += 1
            self._save_checkpoint()

            logger.info(
                "Generation batch {}/{}: {} hypotheses",
                self.session.current_batch,
                self.session.total_batches,
                len(batch_hypotheses),
            )

            if not batch_hypotheses:
                logger.warning("Empty batch at iteration {}/{}", self.session.current_batch, self.session.total_batches)

        return all_hypotheses

    def _index_to_rag(
        self,
        hypotheses: list[ReactionHypothesis],
        results: list[VerificationResult],
    ) -> None:
        """Index verified hypothesis data into the RAG system."""
        if self.rag is None:
            return

        passed = [(h, r) for h, r in zip(hypotheses, results) if r.status == VerificationStatus.PASSED]
        if not passed:
            return

        try:
            for h, r in passed:
                reactants = [e.smiles for e in h.reactants]
                products = [e.smiles for e in h.products]

                # Index reaction template into vector store
                template = (
                    f"Reaction: {h.reaction_name} ({h.reaction_type}). "
                    f"Reactants: {', '.join(reactants)}. "
                    f"Products: {', '.join(products)}. "
                    f"Conditions: {h.conditions}. "
                    f"Rationale: {h.rationale}"
                )
                self.rag.index_reaction_templates([template])

                # Build knowledge graph edges
                self.rag.index_reaction_graph(reactants, products, h.reaction_type)

                # Index scaffold relations
                all_smiles = reactants + products
                self.rag.index_scaffold_relations(all_smiles)
                self.rag.index_functional_groups(all_smiles)

            logger.debug("Indexed {} passed hypotheses into RAG", len(passed))
        except Exception as e:
            logger.warning("RAG indexing failed: {}", e)

    def _verify_hypotheses(
        self, hypotheses: list[ReactionHypothesis]
    ) -> list[VerificationResult]:
        """Run verification agent on all hypotheses."""
        from src.agents.verification_agent import VerificationAgent

        agent = VerificationAgent(
            enable_xtb=self.config.verification_agent.enable_xtb,
            xtb_method=self.config.verification_agent.xtb_method,
            xtb_timeout=self.config.verification_agent.xtb_timeout,
            xtb_max_atoms=self.config.verification_agent.xtb_max_atoms,
            energy_barrier_threshold_kcal=self.config.verification_agent.energy_barrier_threshold,
            sa_score_min=self.config.verification_agent.sa_score_min,
            sa_score_max=self.config.verification_agent.sa_score_max,
            qed_min=self.config.verification_agent.qed_min,
        )

        results: list[VerificationResult] = []
        batch_size = self.config.pipeline.batch_size

        for i in range(0, len(hypotheses), batch_size):
            batch = hypotheses[i : i + batch_size]
            batch_results = agent.verify_batch(batch)
            results.extend(batch_results)

            passed = sum(1 for r in batch_results if r.status == VerificationStatus.PASSED)
            logger.info(
                "Verification batch {}/{}: {} verified, {} passed",
                i // batch_size + 1,
                max(1, len(hypotheses) // batch_size),
                len(batch_results),
                passed,
            )

        return results

    def _reflect_on_failures(
        self,
        failed_hypotheses: list[ReactionHypothesis],
        failed_results: list[VerificationResult],
    ) -> list[ReflectionTrace]:
        """Run reflection agent on failed verifications."""
        from src.agents.reflection_agent import ReflectionAgent

        agent = ReflectionAgent(
            llm=self.llm,
            temperature=self.config.reflection_agent.temperature,
            max_tokens=self.config.reflection_agent.max_tokens,
        )

        traces = agent.reflect_batch(failed_hypotheses, failed_results)
        return traces

    def _accumulate_learning(
        self,
        traces: list[ReflectionTrace],
        results: list[VerificationResult],
        learning_context,
    ):
        """Accumulate reflection traces into a self-bootstrapping learning context.

        Args:
            traces: New reflection traces from the current iteration.
            results: Verification results (passed + failed).
            learning_context: Existing LearningContext or None.

        Returns:
            Updated LearningContext.
        """
        from src.agents.reflection_agent import ReflectionAgent

        agent = ReflectionAgent(
            llm=self.llm,
            temperature=self.config.reflection_agent.temperature,
            max_tokens=self.config.reflection_agent.max_tokens,
        )

        return agent.accumulate_learning(traces, results, learning_context)

    def _compile_dataset(
        self,
        hypotheses: list[ReactionHypothesis],
        results: list[VerificationResult],
        traces: list[ReflectionTrace],
    ) -> dict:
        """Run compilation agent and save dataset."""
        from src.agents.compilation_agent import CompilationAgent

        agent = CompilationAgent(
            output_format=self.config.compilation_agent.output_format,
            train_split=self.config.compilation_agent.train_split,
            val_split=self.config.compilation_agent.val_split,
            test_split=self.config.compilation_agent.test_split,
            min_pairs_per_reaction_type=self.config.compilation_agent.min_pairs_per_reaction_type,
            deduplicate=self.config.compilation_agent.deduplicate,
            random_seed=self.config.pipeline.seed,
        )

        compilation = agent.compile(hypotheses, results, traces)

        output_dir = f"datasets/autochem-{self.session.id}"
        agent.save_dataset(compilation, output_dir)

        return compilation

    def _init_checkpoint(self) -> None:
        """Initialize the checkpoint database for this session."""
        checkpoint_dir = Path(self.config.pipeline.checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self._db_path = checkpoint_dir / f"{self.session.id}.db"

    def _save_checkpoint(self) -> None:
        """Persist current session state to SQLite checkpoint."""
        if self._db_path is None or self.session is None:
            return

        try:
            conn = sqlite3.connect(str(self._db_path))
            conn.execute(
                """CREATE TABLE IF NOT EXISTS checkpoint (
                    key TEXT PRIMARY KEY,
                    value TEXT,
                    updated_at TEXT
                )"""
            )

            self.session.updated_at = datetime.now()
            state_json = self.session.model_dump_json()

            conn.execute(
                "INSERT OR REPLACE INTO checkpoint (key, value, updated_at) VALUES (?, ?, ?)",
                ("session_state", state_json, datetime.now().isoformat()),
            )
            conn.commit()
            conn.close()

            logger.debug("Checkpoint saved: {}", self._db_path)
        except Exception as e:
            logger.error("Failed to save checkpoint: {}", e)
            raise CheckpointError(f"Checkpoint save failed: {e}") from e

    def _load_checkpoint(self) -> None:
        """Load session state from SQLite checkpoint."""
        if self._db_path is None or not self._db_path.exists():
            raise CheckpointError(f"Checkpoint not found: {self._db_path}")

        try:
            conn = sqlite3.connect(str(self._db_path))
            cursor = conn.execute(
                "SELECT value FROM checkpoint WHERE key = ?", ("session_state",)
            )
            row = cursor.fetchone()
            conn.close()

            if row is None:
                raise CheckpointError("No session_state in checkpoint")

            self.session = SessionState.model_validate_json(row[0])

        except Exception as e:
            raise CheckpointError(f"Failed to load checkpoint: {e}") from e
