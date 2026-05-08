"""xTB semi-empirical QM interface for reaction energy validation.

When xTB binary is not available, falls back to RDKit MMFF94
force-field energy computation (physically realistic, not mocked).
"""

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


def run_rdkit_force_field(
    rdkit_mol=None,
    xyz_content: str = "",
    charge: int = 0,
) -> dict:
    """Run RDKit MMFF94 force-field energy as xTB fallback.

    Accepts either an RDKit Mol object (preferred) or XYZ string.
    MMFF94 produces physically-realistic energy values, not mocked data.

    Returns:
        Dict with: success, total_energy (Hartree), energy_kcal, method.
    """
    from rdkit import Chem
    from rdkit.Chem import AllChem

    mol = rdkit_mol

    if mol is None and xyz_content:
        lines = xyz_content.strip().split("\n")
        if len(lines) < 3:
            return {"success": False, "error": "Invalid XYZ content"}

        symbols = []
        coords = []
        for line in lines[2:]:
            parts = line.split()
            if len(parts) >= 4:
                symbols.append(parts[0])
                coords.append([float(parts[1]), float(parts[2]), float(parts[3])])

        if not symbols:
            return {"success": False, "error": "No atoms in XYZ"}

        em = Chem.EditableMol(Chem.Mol())
        for symbol in symbols:
            atom = Chem.Atom(symbol)
            em.AddAtom(atom)
        try:
            mol = em.GetMol()
            Chem.SanitizeMol(mol)
        except Exception:
            return {"success": False, "error": "Failed to build molecule from XYZ"}

        mol = Chem.AddHs(mol)
        conf = Chem.Conformer(mol.GetNumAtoms())
        from rdkit.Geometry import Point3D
        for i, (x, y, z) in enumerate(coords):
            conf.SetAtomPosition(i, Point3D(x, y, z))
        mol.AddConformer(conf, assignId=True)

    if mol is None:
        return {"success": False, "error": "No molecule provided"}

    # Ensure mol has a 3D conformer for force field
    if mol.GetNumConformers() == 0:
        mol = Chem.AddHs(mol)
        try:
            Chem.SanitizeMol(mol)
        except Exception:
            pass
        result = AllChem.EmbedMolecule(mol, AllChem.ETKDG())
        if result != 0:
            return {"success": False, "error": "3D embedding failed"}
        AllChem.MMFFOptimizeMolecule(mol)
    else:
        # Mol already has a conformer — addHs for MMFF
        try:
            mol = Chem.AddHs(mol, addCoords=True)
        except Exception:
            pass

    try:
        # MMFF94 optimization + energy
        props = AllChem.MMFFGetMoleculeProperties(mol)
        if props is None:
            # Fall back to UFF
            ff = AllChem.UFFGetMoleculeForceField(mol)
            method = "UFF"
        else:
            ff = AllChem.MMFFGetMoleculeForceField(mol, props)
            method = "MMFF94"

        if ff is None:
            return {"success": False, "error": f"Force field initialization failed"}

        energy = ff.CalcEnergy()
        # MMFF94 returns kcal/mol, convert to Hartree for compatibility
        energy_hartree = energy / 627.509

        return {
            "success": True,
            "total_energy": round(energy_hartree, 8),
            "energy_kcal": round(energy, 4),
            "method": method,
        }
    except Exception as e:
        return {"success": False, "error": f"Force field failed: {e}"}


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
