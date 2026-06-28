"""Auto-ChemInstruct problem validator — wraps existing verification for GigaEvo.

This module bridges the existing Auto-ChemInstruct verification system
into GigaEvo's problem.validate(output) interface.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.agents.verification_agent import VerificationAgent
from src.config import load_config


def validate(output: dict, helper_module: object | None = None) -> dict:  # type: ignore[type-arg]
    config = load_config()
    vcfg = config.verification_agent
    verifier = VerificationAgent(
        enable_xtb=vcfg.enable_xtb,
        xtb_method=vcfg.xtb_method,
        xtb_timeout=vcfg.xtb_timeout,
        xtb_max_atoms=vcfg.xtb_max_atoms,
        energy_barrier_threshold_kcal=vcfg.energy_barrier_threshold,
        sa_score_min=vcfg.sa_score_min,
        sa_score_max=vcfg.sa_score_max,
        qed_min=vcfg.qed_min,
    )

    try:
        hypothesis = _build_hypothesis(output)
    except Exception as e:
        return {
            "fitness": 0.0,
            "structural_validity": 0.0,
            "energetic_feasibility": 0.0,
            "chemical_diversity": 0.0,
            "pass_rate": 0.0,
            "errors": [f"Hypothesis construction failed: {e}"],
        }

    result = verifier.verify(hypothesis)

    structural = 1.0 if result.smiles_valid else 0.0
    energetic = 1.0 if result.energy_valid else 0.0
    pass_rate = 1.0 if result.status.value == "PASSED" else 0.0

    fitness = _compute_composite_score(result, structural, energetic)
    diversity = _estimate_diversity(hypothesis)

    errors = [str(e) for e in (result.errors or [])]

    return {
        "fitness": round(fitness, 4),
        "structural_validity": structural,
        "energetic_feasibility": energetic,
        "chemical_diversity": round(diversity, 4),
        "pass_rate": pass_rate,
        "errors": errors,
    }


def _build_hypothesis(output: dict):  # type: ignore[type-arg]
    """Build ReactionHypothesis from GigaEvo output dict."""
    from uuid import uuid4

    from src.data.models import (
        ChemicalEntity,
        ReactionConditions,
        ReactionHypothesis,
        ReactionType,
    )

    reactants = []
    for smi in output.get("reactants", []):
        try:
            from rdkit import Chem

            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                raise ValueError(f"Invalid SMILES: {smi}")
            formula = Chem.rdMolDescriptors.CalcMolFormula(mol)
            mw = Chem.Descriptors.MolWt(mol)
            reactants.append(
                ChemicalEntity(
                    smiles=smi,
                    molecular_weight=mw,
                    formula=formula,
                )
            )
        except Exception:
            reactants.append(ChemicalEntity(smiles=smi))

    products = []
    for smi in output.get("products", []):
        try:
            from rdkit import Chem

            mol = Chem.MolFromSmiles(smi)
            formula = Chem.rdMolDescriptors.CalcMolFormula(mol) if mol else None
            mw = Chem.Descriptors.MolWt(mol) if mol else None
            products.append(
                ChemicalEntity(
                    smiles=smi,
                    molecular_weight=mw,
                    formula=formula,
                )
            )
        except Exception:
            products.append(ChemicalEntity(smiles=smi))

    try:
        reaction_type = ReactionType(output.get("reaction_type", "other"))
    except ValueError:
        reaction_type = ReactionType.OTHER

    cond = output.get("conditions", {})
    conditions = ReactionConditions(
        temperature_celsius=cond.get("temperature_c", 25.0),
        pressure_atm=cond.get("pressure_atm", 1.0),
        solvent=cond.get("solvent", None),
        catalyst=cond.get("catalyst", None),
        time_hours=cond.get("time_h", None),
        ph=cond.get("pH", None),
        atmosphere=cond.get("atmosphere", None),
    )

    steps_raw = output.get("mechanism_steps", [])
    steps_str = (
        "\n".join(steps_raw)
        if isinstance(steps_raw, list)
        else str(steps_raw)
        if steps_raw
        else None
    )

    return ReactionHypothesis(
        session_id=uuid4(),
        reactants=reactants,
        products=products,
        reaction_type=reaction_type,
        conditions=conditions,
        yield_estimate=output.get("yield_estimate"),
        mechanism_steps=steps_str,
        rationale=output.get("rationale", ""),
    )


def _compute_composite_score(result, structural: float, energetic: float) -> float:  # type: ignore[no-untyped-def]
    score = 0.0
    if structural > 0:
        score += 0.30
        props = result.properties
        if props:
            if props.qed and props.qed > 0.3:
                score += 0.15
            if props.sa_score and props.sa_score < 4.0:
                score += 0.10
    if energetic > 0:
        score += 0.25
    if result.steric_valid:
        score += 0.10
    if result.chemically_feasible:
        score += 0.10
    return score


def _estimate_diversity(hypothesis) -> float:  # type: ignore[no-untyped-def]
    try:
        from rdkit import Chem

        from src.chemistry.diversity import compute_diversity_score

        all_smiles: list[str] = [e.smiles for e in hypothesis.reactants] + [
            e.smiles for e in hypothesis.products
        ]
        mols = [Chem.MolFromSmiles(s) for s in all_smiles]
        mols = [m for m in mols if m is not None]
        if len(mols) < 2:
            return 0.5
        score = compute_diversity_score(mols)
        return float(score)
    except Exception:
        return 0.5
