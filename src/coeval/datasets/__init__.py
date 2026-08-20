"""CoEval dataset loaders."""

from coeval.core.registry import get_dataset, list_datasets, register_dataset
from coeval.datasets.attributionbench import (
    AttributionBenchAttributedQADataset,
    AttributionBenchExpertQADataset,
    AttributionBenchLFQADataset,
    AttributionBenchStanfordGenSearchDataset,
)
from coeval.datasets.careqa import CareQADataset
from coeval.datasets.conquer import (
    ConquerHealthTestDataset,
    ConquerHealthValDataset,
)
from coeval.datasets.headqa import HeadQADataset
from coeval.datasets.healthbench import (
    HealthBenchConsensusDataset,
    HealthBenchMainDataset,
)
from coeval.datasets.m_arc import MARCDataset
from coeval.datasets.medbullets import (
    Medbullets4OptionsDataset,
    Medbullets5OptionsDataset,
)
from coeval.datasets.medcalc import MedCalcDataset
from coeval.datasets.medhallu import MedHalluDataset
from coeval.datasets.medmcqa import MedMCQADataset
from coeval.datasets.medqa import MedQADataset
from coeval.datasets.medxpertqa import (
    MedXpertQAReasoningDataset,
    MedXpertQAUnderstandingDataset,
)
from coeval.datasets.metamedqa import MetaMedQADataset
from coeval.datasets.mmlu_pro_health import MMLUProHealthDataset
from coeval.datasets.pubmedqa import PubMedQADataset

__all__ = [
    "get_dataset",
    "list_datasets",
    "register_dataset",
    "AttributionBenchAttributedQADataset",
    "AttributionBenchExpertQADataset",
    "AttributionBenchLFQADataset",
    "AttributionBenchStanfordGenSearchDataset",
    "ConquerHealthTestDataset",
    "ConquerHealthValDataset",
    "CareQADataset",
    "HeadQADataset",
    "HealthBenchConsensusDataset",
    "HealthBenchMainDataset",
    "MARCDataset",
    "MedCalcDataset",
    "MedHalluDataset",
    "MedMCQADataset",
    "MedQADataset",
    "MedXpertQAReasoningDataset",
    "MedXpertQAUnderstandingDataset",
    "Medbullets4OptionsDataset",
    "Medbullets5OptionsDataset",
    "MetaMedQADataset",
    "MMLUProHealthDataset",
    "PubMedQADataset",
]
