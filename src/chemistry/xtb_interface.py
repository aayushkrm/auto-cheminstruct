"""xTB semi-empirical QM interface for reaction energy validation."""

from __future__ import annotations

import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Optional

from loguru import logger

from src.exceptions import SimulationError, SimulationTimeoutError, XTBNotFoundError


def _find_xtb_binary() -> str:
    """Locate xTB binary on the system."""
    xtb_path = shutil.which("xtb")
    if xtb_path is None:
        raise XTBNotFoundError(
            "xTB binary not found. Install from https://github.com/grimme-lab/xtb"
        )
    return xtb_path


def run_xtb_single_point(
    xyz_content: str,
    charge: int = 0,
    multiplicity: int = 1,
    method: str = "GFN2-xTB",
    timeout: int = 300,
    work_dir: str | None = None,
) -> dict:
    """Run a single-point xTB calculation on a molecule.

    Args:
        xyz_content: XYZ format molecular geometry.
        charge: Molecular charge.
        multiplicity: Spin multiplicity (2S+1).
        method: xTB method (GFN2-xTB or GFN1-xTB or GFN-FF).
        timeout: Maximum runtime in seconds.
        work_dir: Working directory for calculation. Uses temp dir if None.

    Returns:
        Dict with: success, total_energy, homo, lumo, gap, dipole,
        wall_time, output.

    Raises:
        XTBNotFoundError: If xTB binary not found.
        SimulationTimeoutError: If calculation times out.
        SimulationError: If calculation fails.
    """
    xtb_bin = _find_xtb_binary()
    cleanup = False

    if work_dir is None:
        work_dir_path = Path(tempfile.mkdtemp(prefix="xtb_"))
        cleanup = True
    else:
        work_dir_path = Path(work_dir)
        work_dir_path.mkdir(parents=True, exist_ok=True)

    xyz_file = work_dir_path / "molecule.xyz"
    xyz_file.write_text(xyz_content)

    try:
        cmd = [
            xtb_bin,
            str(xyz_file),
            "--gfn",
            "2" if "GFN2" in method else "1" if "GFN1" in method else "ff",
            "--chrg",
            str(charge),
            "--uhf",
            str(multiplicity - 1),
        ]

        logger.debug("Running xTB: {}", " ".join(cmd))

        start = time.monotonic()
        process = subprocess.run(
            cmd,
            cwd=str(work_dir_path),
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        elapsed = time.monotonic() - start

        stdout = process.stdout
        stderr = process.stderr

        if process.returncode != 0:
            raise SimulationError(
                f"xTB failed with code {process.returncode}: {stderr[:500]}"
            )

        result = {
            "success": True,
            "wall_time": elapsed,
            "output": stdout,
            "errors": stderr,
        }

        total_energy = _parse_total_energy(stdout)
        if total_energy is not None:
            result["total_energy"] = total_energy

        homo_lumo = _parse_homo_lumo(stdout)
        if homo_lumo:
            result.update(homo_lumo)

        dipole = _parse_dipole(stdout)
        if dipole is not None:
            result["dipole"] = dipole

        return result

    except subprocess.TimeoutExpired:
        process.kill()
        raise SimulationTimeoutError(
            f"xTB calculation timed out after {timeout}s"
        )
    finally:
        if cleanup:
            shutil.rmtree(work_dir_path, ignore_errors=True)


def run_xtb_optimization(
    xyz_content: str,
    charge: int = 0,
    multiplicity: int = 1,
    method: str = "GFN2-xTB",
    timeout: int = 600,
    work_dir: str | None = None,
) -> dict:
    """Run xTB geometry optimization.

    Returns same dict as run_xtb_single_point plus optimized XYZ.
    """
    xtb_bin = _find_xtb_binary()
    cleanup = False

    if work_dir is None:
        work_dir_path = Path(tempfile.mkdtemp(prefix="xtb_opt_"))
        cleanup = True
    else:
        work_dir_path = Path(work_dir)
        work_dir_path.mkdir(parents=True, exist_ok=True)

    xyz_file = work_dir_path / "molecule.xyz"
    xyz_file.write_text(xyz_content)

    try:
        cmd = [
            xtb_bin,
            str(xyz_file),
            "--gfn",
            "2" if "GFN2" in method else "1",
            "--chrg",
            str(charge),
            "--uhf",
            str(multiplicity - 1),
            "--opt",
        ]

        logger.debug("Running xTB optimization: {}", " ".join(cmd))

        start = time.monotonic()
        process = subprocess.run(
            cmd,
            cwd=str(work_dir_path),
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        elapsed = time.monotonic() - start

        if process.returncode != 0:
            raise SimulationError(
                f"xTB optimization failed: {process.stderr[:500]}"
            )

        result = {
            "success": True,
            "wall_time": elapsed,
            "output": process.stdout,
            "errors": process.stderr,
        }

        optimized_xyz_file = work_dir_path / "xtbopt.xyz"
        if optimized_xyz_file.exists():
            result["optimized_xyz"] = optimized_xyz_file.read_text()

        total_energy = _parse_total_energy(process.stdout)
        if total_energy is not None:
            result["total_energy"] = total_energy

        homo_lumo = _parse_homo_lumo(process.stdout)
        if homo_lumo:
            result.update(homo_lumo)

        return result

    except subprocess.TimeoutExpired:
        process.kill()
        raise SimulationTimeoutError(
            f"xTB optimization timed out after {timeout}s"
        )
    finally:
        if cleanup:
            shutil.rmtree(work_dir_path, ignore_errors=True)


def estimate_reaction_energy(
    reactant_energy: float,
    product_energy: float,
    unit: str = "hartree",
) -> float:
    """Compute reaction energy from reactant and product energies.

    ΔG_rxn = E_products - E_reactants

    Returns energy in kcal/mol.
    """
    hartree_to_kcal = 627.509
    if unit == "hartree":
        return (product_energy - reactant_energy) * hartree_to_kcal
    return product_energy - reactant_energy


def xyz_from_rdkit(mol) -> str:
    """Generate XYZ string from RDKit Mol (must have 3D conformer).

    Args:
        mol: RDKit Mol with conformer.

    Returns:
        XYZ format string.
    """
    from rdkit import Chem

    conf = mol.GetConformer()
    atoms = mol.GetAtoms()
    num_atoms = mol.GetNumAtoms()

    lines = [str(num_atoms), ""]
    for i, atom in enumerate(atoms):
        pos = conf.GetAtomPosition(i)
        lines.append(
            f"{atom.GetSymbol():<3} {pos.x:12.6f} {pos.y:12.6f} {pos.z:12.6f}"
        )

    return "\n".join(lines)


def _parse_total_energy(output: str) -> Optional[float]:
    """Parse total energy from xTB output."""
    for line in output.splitlines():
        if "TOTAL ENERGY" in line:
            parts = line.split()
            for i, part in enumerate(parts):
                if part == "ENERGY" and i + 1 < len(parts):
                    try:
                        return float(parts[i + 1])
                    except ValueError:
                        pass
        if "total energy" in line.lower():
            parts = line.split()
            try:
                return float(parts[-1])
            except (ValueError, IndexError):
                pass
    return None


def _parse_homo_lumo(output: str) -> dict:
    """Parse HOMO/LUMO energies from xTB output."""
    result = {}
    for line in output.splitlines():
        if "HOMO" in line and "LUMO" in line:
            parts = line.split()
            try:
                homo_idx = parts.index("HOMO")
                lumo_idx = parts.index("LUMO")
                result["homo"] = float(parts[homo_idx + 1])
                result["lumo"] = float(parts[lumo_idx + 1])
                result["gap"] = result["lumo"] - result["homo"]
            except (ValueError, IndexError):
                pass
            break
    return result


def _parse_dipole(output: str) -> Optional[float]:
    """Parse dipole moment from xTB output."""
    for line in output.splitlines():
        if "dipole moment" in line.lower() or "molecular dipole" in line.lower():
            parts = line.split()
            for i, part in enumerate(parts):
                if "D" in part and i > 0:
                    try:
                        return float(parts[i - 1])
                    except ValueError:
                        pass
    return None
