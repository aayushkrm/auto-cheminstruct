"""Custom exception hierarchy for Auto-ChemInstruct."""


class AutoChemError(Exception):
    """Base exception for all Auto-ChemInstruct errors."""


class ConfigurationError(AutoChemError):
    """Configuration validation or loading error."""


class ChemistryError(AutoChemError):
    """Chemical validation or processing error."""


class SMILESParseError(ChemistryError):
    """Failed to parse SMILES string."""


class MolecularValidationError(ChemistryError):
    """Molecular structure failed validation checks."""


class SimulationError(ChemistryError):
    """xTB or quantum chemical simulation error."""


class SimulationTimeoutError(SimulationError):
    """xTB simulation exceeded time limit."""


class XTBNotFoundError(SimulationError):
    """xTB binary not found on system."""


class AgentError(AutoChemError):
    """Agent execution or communication error."""


class LLMError(AgentError):
    """LLM API call or parsing error."""


class AgentCommunicationError(AgentError):
    """Inter-agent message routing failure."""


class PipelineError(AutoChemError):
    """Pipeline execution or state management error."""


class CheckpointError(PipelineError):
    """Checkpoint save/load failure."""


class DatasetError(AutoChemError):
    """Dataset compilation or export error."""


class ValidationError(AutoChemError):
    """Data model validation error."""
