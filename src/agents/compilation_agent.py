"""Compilation Agent — builds DPO/RLHF preference pairs from verified reactions and reflection traces.

Transforms the pipeline outputs (successful reactions, failed reactions with causal
reflections) into structured instruction datasets suitable for preference optimization
training of chemistry DSLMs.
"""

from __future__ import annotations

import json
import hashlib
from collections import defaultdict
from datetime import datetime
from typing import Optional
from uuid import UUID, uuid4

import numpy as np
from loguru import logger

from src.chemistry.diversity import reaction_type_counts
from src.data.models import (
    PreferencePair,
    ReactionHypothesis,
    ReactionType,
    ReflectionTrace,
    VerificationResult,
    VerificationStatus,
)


class CompilationAgent:
    """Compiles verified reaction data into DPO/RLHF preference pair datasets."""

    def __init__(
        self,
        output_format: str = "dpo",
        train_split: float = 0.8,
        val_split: float = 0.1,
        test_split: float = 0.1,
        min_pairs_per_reaction_type: int = 10,
        deduplicate: bool = True,
        random_seed: int = 42,
    ):
        self.output_format = output_format
        self.train_split = train_split
        self.val_split = val_split
        self.test_split = test_split
        self.min_pairs_per_reaction_type = min_pairs_per_reaction_type
        self.deduplicate = deduplicate
        self.random_seed = random_seed
        self._rng = np.random.RandomState(random_seed)

    def compile(
        self,
        hypotheses: list[ReactionHypothesis],
        verification_results: list[VerificationResult],
        reflection_traces: list[ReflectionTrace],
    ) -> dict:
        """Compile all pipeline outputs into a structured dataset.

        Args:
            hypotheses: All generated reaction hypotheses.
            verification_results: Verification results (one per hypothesis).
            reflection_traces: Reflection traces for failed reactions.

        Returns:
            Dict with train, val, test splits, each containing PreferencePair lists.
        """
        result_by_id = {r.hypothesis_id: r for r in verification_results}
        trace_by_id = {t.hypothesis_id: t for t in reflection_traces}

        pairs = self._build_pairs(hypotheses, result_by_id, trace_by_id)

        if self.deduplicate:
            pairs = self._deduplicate(pairs)

        logger.info(
            "Compiled {} preference pairs across {} reaction types",
            len(pairs),
            len(set(p.reaction_type for p in pairs)),
        )

        splits = self._split_pairs(pairs)

        return {
            "train": splits["train"],
            "val": splits["val"],
            "test": splits["test"],
            "metadata": self._build_metadata(splits, hypotheses, verification_results, reflection_traces),
        }

    def _build_pairs(
        self,
        hypotheses: list[ReactionHypothesis],
        result_by_id: dict[UUID, VerificationResult],
        trace_by_id: dict[UUID, ReflectionTrace],
    ) -> list[PreferencePair]:
        """Construct preference pairs by matching passed reactions with failed+reflected ones.

        Strategy:
            - Group hypotheses by reaction type
            - Within each type, match passed reactions (chosen) with failed+reflected (rejected)
            - Each passed reaction can pair with up to 1-3 failed reactions
        """
        passed: dict[ReactionType, list[ReactionHypothesis]] = defaultdict(list)
        failed: dict[ReactionType, list[tuple[ReactionHypothesis, VerificationResult, ReflectionTrace]]] = defaultdict(list)

        for hyp in hypotheses:
            result = result_by_id.get(hyp.id)
            if result is None:
                continue

            if result.status == VerificationStatus.PASSED:
                passed[hyp.reaction_type].append(hyp)
            elif result.status == VerificationStatus.FAILED:
                trace = trace_by_id.get(hyp.id)
                if trace is not None:
                    failed[hyp.reaction_type].append((hyp, result, trace))

        pairs: list[PreferencePair] = []
        total_failed_with_traces = sum(len(v) for v in failed.values())

        for rtype in ReactionType:
            type_passed = passed.get(rtype, [])
            type_failed = failed.get(rtype, [])

            if not type_passed:
                continue

            if not type_failed:
                # Fallback: pair passed reactions with synthetic rejected entries
                for hyp in type_passed:
                    pair = self._build_solo_pair(hyp)
                    if pair:
                        pairs.append(pair)
                continue

            self._rng.shuffle(type_failed)

            for i, chosen_hyp in enumerate(type_passed):
                rejected_idx = i % len(type_failed)
                rej_hyp, rej_result, rej_trace = type_failed[rejected_idx]

                chosen_text = self._format_chosen(chosen_hyp)
                rejected_text = self._format_rejected(rej_hyp, rej_result, rej_trace)

                prompt = self._generate_prompt(chosen_hyp)

                pair = PreferencePair(
                    id=uuid4(),
                    prompt=prompt,
                    chosen=chosen_text,
                    rejected=rejected_text,
                    chosen_hypothesis_id=chosen_hyp.id,
                    rejected_hypothesis_id=rej_hyp.id,
                    reflection_trace_id=rej_trace.id,
                    reaction_type=rtype,
                    quality_score=self._score_pair(chosen_hyp, rej_hyp, rej_trace),
                    metadata={
                        "chosen_yield": chosen_hyp.yield_estimate,
                        "rejected_errors": rej_result.errors,
                        "failure_categories": [c.value for c in rej_trace.failure_categories],
                        "reflection_confidence": rej_trace.confidence,
                    },
                    created_at=datetime.now(),
                )
                pairs.append(pair)

        return pairs

    def _format_chosen(self, hyp: ReactionHypothesis) -> str:
        """Format a successful reaction as the chosen (preferred) response."""
        reactants = ", ".join(r.smiles for r in hyp.reactants)
        products = ", ".join(p.smiles for p in hyp.products)
        parts = [
            f"SUCCESSFUL REACTION:",
            f"Reactants: {reactants}",
            f"Products: {products}",
            f"Type: {hyp.reaction_type.value}",
        ]
        if hyp.conditions.temperature_celsius:
            parts.append(f"Temperature: {hyp.conditions.temperature_celsius}°C")
        if hyp.conditions.solvent:
            parts.append(f"Solvent: {hyp.conditions.solvent.value}")
        if hyp.conditions.catalyst:
            parts.append(f"Catalyst: {hyp.conditions.catalyst}")
        if hyp.yield_estimate:
            parts.append(f"Expected Yield: {hyp.yield_estimate}%")
        if hyp.mechanism_steps:
            parts.append(f"Mechanism: {hyp.mechanism_steps}")
        if hyp.rationale:
            parts.append(f"Rationale: {hyp.rationale}")
        return "\n".join(parts)

    def _format_rejected(
        self,
        hyp: ReactionHypothesis,
        result: VerificationResult,
        trace: ReflectionTrace,
    ) -> str:
        """Format a failed reaction with reflection as the rejected response."""
        reactants = ", ".join(r.smiles for r in hyp.reactants)
        products = ", ".join(p.smiles for p in hyp.products)
        parts = [
            f"FAILED REACTION:",
            f"Reactants: {reactants}",
            f"Products: {products}",
            f"Type: {hyp.reaction_type.value}",
            "",
            f"FAILURE ANALYSIS:",
            f"Primary Cause: {trace.primary_cause or 'Analysis unavailable'}",
            f"Causal Explanation: {trace.causal_explanation}",
        ]
        if trace.failure_categories:
            cats = ", ".join(c.value for c in trace.failure_categories)
            parts.append(f"Failure Categories: {cats}")
        if trace.chemical_principles:
            parts.append(f"Relevant Principles: {', '.join(trace.chemical_principles)}")
        if trace.fix_suggestion:
            parts.append(f"Suggested Fix: {trace.fix_suggestion}")
        if result.errors:
            parts.append(f"Validation Errors: {'; '.join(result.errors)}")
        return "\n".join(parts)

    def _generate_prompt(self, hyp: ReactionHypothesis) -> str:
        """Generate an instruction prompt for the preference pair."""
        reactants = ", ".join(r.smiles for r in hyp.reactants)
        rtype = hyp.reaction_type.value.replace("_", " ")
        templates = [
            f"Propose a {rtype} reaction converting {reactants} to the expected products. Include reaction conditions and mechanism.",
            f"Design a synthetic route for {rtype} using {reactants} as starting materials. What products form and under what conditions?",
            f"Given the reactants {reactants}, predict the {rtype} products and propose optimal reaction conditions.",
            f"Analyze the {rtype} reaction of {reactants}. What are the expected products, mechanism, and suitable conditions?",
        ]
        idx = hash(hyp.id) % len(templates)
        return templates[idx]

    def _build_solo_pair(self, hyp: ReactionHypothesis) -> PreferencePair | None:
        """Build a pair with a synthetic rejected entry when no failures exist."""
        try:
            chosen_text = self._format_chosen(hyp)
            rejected_text = (
                "NO REJECTED REACTION AVAILABLE\n"
                "This variant produced no failed hypotheses for comparison.\n"
                f"All {hyp.reaction_type.value} reactions passed verification."
            )
            prompt = self._generate_prompt(hyp)
            return PreferencePair(
                id=uuid4(),
                prompt=prompt,
                chosen=chosen_text,
                rejected=rejected_text,
                chosen_hypothesis_id=hyp.id,
                rejected_hypothesis_id=hyp.id,
                reflection_trace_id=None,
                reaction_type=hyp.reaction_type,
                quality_score=self._score_pair(hyp, hyp, None),
                metadata={"solo_pair": True},
                created_at=datetime.now(),
            )
        except Exception:
            return None

    def _score_pair(
        self,
        chosen: ReactionHypothesis,
        rejected: ReactionHypothesis,
        trace: ReflectionTrace | None,
    ) -> float:
        """Score a preference pair for quality (0-1) using chemistry-aware metrics.

        Factors:
            - Structural validity (chosen SMILES must be chemically valid)
            - Chemical feasibility (QED drug-likeness)
            - Reflection depth (causal chain, specific chemical principles)
            - Yield differential (passed should have reasonable yield)
            - Scaffold diversity bonus (non-trivial structures preferred)
            - Reaction type specificity (named/classified reactions preferred)
        """
        from rdkit import Chem
        from rdkit.Chem import Descriptors

        score = 0.0
        weights_applied = 0.0

        # 1. Structural validity (20% weight)
        chosen_valid = all(
            Chem.MolFromSmiles(e.smiles) is not None
            for e in chosen.reactants + chosen.products
        )
        rejected_invalid = any(
            Chem.MolFromSmiles(e.smiles) is None
            for e in rejected.reactants + rejected.products
        )
        if chosen_valid:
            score += 0.20
        elif not rejected_invalid:
            score += 0.05
        weights_applied += 0.20

        # 2. Chemical feasibility via QED (15% weight)
        chosen_mols = []
        for e in chosen.reactants + chosen.products:
            mol = Chem.MolFromSmiles(e.smiles)
            if mol:
                chosen_mols.append(mol)
        if chosen_mols:
            try:
                qed_vals = [Descriptors.qed(m) for m in chosen_mols]
                avg_qed = sum(qed_vals) / len(qed_vals)
                score += avg_qed * 0.15
            except Exception:
                score += 0.05
        weights_applied += 0.15

        # 3. Reflection quality (25% weight)
        if trace:
            score += trace.confidence * 0.10
            if len(trace.causal_explanation) > 100:
                score += 0.05
            if len(trace.causal_explanation) > 300:
                score += 0.05
            if trace.chemical_principles:
                score += min(len(trace.chemical_principles) * 0.025, 0.05)
        else:
            score += 0.05  # minimal score for no reflection
        weights_applied += 0.25

        # 4. Yield differential (10% weight)
        if chosen.yield_estimate:
            if chosen.yield_estimate >= 80:
                score += 0.10
            elif chosen.yield_estimate >= 60:
                score += 0.08
            elif chosen.yield_estimate >= 40:
                score += 0.05
            elif chosen.yield_estimate >= 20:
                score += 0.02
        weights_applied += 0.10

        # 5. Scaffold diversity bonus (15% weight)
        if chosen_mols:
            from rdkit.Chem.Scaffolds import MurckoScaffold
            try:
                scaffolds = set()
                heavy_count = 0
                for m in chosen_mols:
                    heavy_count += m.GetNumHeavyAtoms()
                    scaff = MurckoScaffold.GetScaffoldForMol(m)
                    if scaff and scaff.GetNumAtoms() > 0:
                        scaffolds.add(Chem.MolToSmiles(scaff))
                scaffold_ratio = len(scaffolds) / max(1, len(chosen_mols))
                score += min(scaffold_ratio, 1.0) * 0.08
                if heavy_count / len(chosen_mols) > 5:
                    score += 0.07
                else:
                    score += 0.03
            except Exception:
                score += 0.05
        weights_applied += 0.15

        # 6. Reaction type specificity (15% weight)
        specific_types = {
            "substitution", "addition", "elimination", "cycloaddition",
            "rearrangement", "oxidation", "reduction", "coupling",
            "condensation", "hydrolysis", "grignard", "alkylation",
            "acylation", "diels_alder", "wittig", "suzuki", "heck",
        }
        rtype = chosen.reaction_type.value.lower() if hasattr(chosen.reaction_type, 'value') else str(chosen.reaction_type).lower()
        if rtype in specific_types or any(st in rtype for st in specific_types):
            score += 0.15
        elif rtype != "other" and rtype != "unknown":
            score += 0.10
        else:
            score += 0.03
        weights_applied += 0.15

        return round(max(0.0, min(1.0, score)), 4)

    def _deduplicate(self, pairs: list[PreferencePair]) -> list[PreferencePair]:
        """Remove duplicate preference pairs (same chosen+rejected content)."""
        seen: set[str] = set()
        unique: list[PreferencePair] = []

        for pair in pairs:
            content_hash = hashlib.md5(
                (pair.chosen + pair.rejected).encode()
            ).hexdigest()
            if content_hash not in seen:
                seen.add(content_hash)
                unique.append(pair)

        removed = len(pairs) - len(unique)
        if removed > 0:
            logger.info("Removed {} duplicate preference pairs", removed)

        return unique

    def _split_pairs(
        self, pairs: list[PreferencePair]
    ) -> dict[str, list[PreferencePair]]:
        """Split pairs into train/val/test with stratified sampling by reaction type."""
        by_type: dict[ReactionType, list[PreferencePair]] = defaultdict(list)
        for pair in pairs:
            by_type[pair.reaction_type].append(pair)

        train, val, test = [], [], []

        for rtype, type_pairs in by_type.items():
            indices = self._rng.permutation(len(type_pairs))
            type_pairs = [type_pairs[i] for i in indices]

            n = len(type_pairs)
            n_train = max(1, int(n * self.train_split))
            n_val = max(0, int(n * self.val_split))

            train.extend(type_pairs[:n_train])
            if n_val > 0:
                val.extend(type_pairs[n_train : n_train + n_val])
            test.extend(type_pairs[n_train + n_val :])

        logger.info(
            "Split: train={}, val={}, test={}",
            len(train),
            len(val),
            len(test),
        )

        return {"train": train, "val": val, "test": test}

    def _build_metadata(
        self,
        splits: dict[str, list[PreferencePair]],
        hypotheses: list[ReactionHypothesis],
        verification_results: list[VerificationResult],
        reflection_traces: list[ReflectionTrace],
    ) -> dict:
        """Build dataset metadata."""
        passed = sum(
            1 for r in verification_results if r.status == VerificationStatus.PASSED
        )
        failed = sum(
            1 for r in verification_results if r.status == VerificationStatus.FAILED
        )

        return {
            "total_hypotheses": len(hypotheses),
            "passed_verification": passed,
            "failed_verification": failed,
            "reflections_generated": len(reflection_traces),
            "total_pairs": sum(len(splits[s]) for s in ["train", "val", "test"]),
            "train_pairs": len(splits["train"]),
            "val_pairs": len(splits["val"]),
            "test_pairs": len(splits["test"]),
            "reaction_types": {
                rtype: count
                for rtype, count in reaction_type_counts(
                    splits["train"] + splits["val"] + splits["test"]
                ).items()
            },
            "output_format": self.output_format,
            "created_at": datetime.now().isoformat(),
        }

    def export_to_hf_dataset(self, splits: dict) -> dict:
        """Convert preference pairs to HuggingFace datasets format.

        Args:
            splits: Dict with train/val/test lists of PreferencePair.

        Returns:
            Dict of HuggingFace Dataset objects.
        """
        try:
            from datasets import Dataset, DatasetDict
        except ImportError:
            logger.error("HuggingFace datasets not installed. Install with: pip install datasets")
            return {}

        dataset_dict = {}
        for split_name, pairs in splits.items():
            if split_name == "metadata" or not pairs:
                continue

            records = []
            for pair in pairs:
                records.append({
                    "id": str(pair.id),
                    "prompt": pair.prompt,
                    "chosen": pair.chosen,
                    "rejected": pair.rejected,
                    "reaction_type": pair.reaction_type.value,
                    "quality_score": pair.quality_score,
                    "metadata": json.dumps(pair.metadata),
                })

            dataset_dict[split_name] = Dataset.from_list(records)

        return dataset_dict

    def save_dataset(
        self,
        splits: dict,
        output_dir: str = "datasets/auto-cheminstruct-v1",
    ) -> str:
        """Save compiled dataset to disk in HuggingFace format.

        Args:
            splits: Dict with train/val/test lists of PreferencePair.
            output_dir: Directory to save the dataset.

        Returns:
            Path to saved dataset directory.
        """
        from pathlib import Path
        import json as json_module

        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        for split_name, pairs in splits.items():
            if split_name == "metadata" or not pairs:
                continue

            records = []
            for pair in pairs:
                records.append({
                    "id": str(pair.id),
                    "prompt": pair.prompt,
                    "chosen": pair.chosen,
                    "rejected": pair.rejected,
                    "reaction_type": pair.reaction_type.value,
                    "quality_score": pair.quality_score,
                    "metadata": pair.metadata,
                })

            split_file = out_path / f"{split_name}.jsonl"
            with open(split_file, "w") as f:
                for record in records:
                    f.write(json_module.dumps(record) + "\n")

            logger.info("Saved {} {} pairs to {}", len(records), split_name, split_file)

        try:
            hf_dataset = self.export_to_hf_dataset(splits)
            if hf_dataset:
                from datasets import DatasetDict, concatenate_datasets
                dd = DatasetDict(hf_dataset)
                dd.save_to_disk(str(out_path / "hf_dataset"))
                logger.info("Saved HuggingFace dataset to {}", out_path / "hf_dataset")
        except Exception as e:
            logger.warning("Failed to save HuggingFace dataset: {}", e)

        return str(out_path)
