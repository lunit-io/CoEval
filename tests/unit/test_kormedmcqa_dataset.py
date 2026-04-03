"""Tests for KorMedMCQA dataset loader."""

from unittest.mock import MagicMock, patch

from coeval.datasets.kormedmcqa import (
    KorMedMCQADentistDataset,
    KorMedMCQADoctorDataset,
    KorMedMCQANurseDataset,
    KorMedMCQAPharmDataset,
    _KorMedMCQABase,
)


def _make_row(
    question: str = "테스트 질문입니다.",
    a: str = "보기1",
    b: str = "보기2",
    c: str = "보기3",
    d: str = "보기4",
    e: str = "보기5",
    answer: int = 2,
    subject: str = "doctor",
    year: int = 2023,
) -> dict:
    return {
        "question": question,
        "A": a,
        "B": b,
        "C": c,
        "D": d,
        "E": e,
        "answer": answer,
        "subject": subject,
        "year": year,
        "period": 1,
        "q_number": 1,
        "cot": "",
    }


def _make_mock_dataset(rows: list[dict]) -> MagicMock:
    mock = MagicMock()
    mock.__iter__ = MagicMock(return_value=iter(rows))
    mock.__len__ = MagicMock(return_value=len(rows))
    return mock


class TestKorMedMCQASubsets:
    """Tests for individual exam-domain subset classes."""

    def _load_subset(self, cls, num_samples=None):
        mock_split = _make_mock_dataset([_make_row()])
        with patch(
            "coeval.datasets.kormedmcqa.load_dataset",
            return_value=mock_split,
        ) as mock_load:
            dataset = cls(num_samples=num_samples)
        return dataset, mock_load

    def test_doctor_subset_constant(self):
        assert KorMedMCQADoctorDataset.SUBSET == "doctor"

    def test_nurse_subset_constant(self):
        assert KorMedMCQANurseDataset.SUBSET == "nurse"

    def test_pharm_subset_constant(self):
        assert KorMedMCQAPharmDataset.SUBSET == "pharm"

    def test_dentist_subset_constant(self):
        assert KorMedMCQADentistDataset.SUBSET == "dentist"

    def test_doctor_loads_correct_subset(self):
        _, mock_load = self._load_subset(KorMedMCQADoctorDataset)
        mock_load.assert_called_once_with(
            _KorMedMCQABase.HUGGINGFACE_PATH, name="doctor", split="test"
        )

    def test_nurse_loads_correct_subset(self):
        _, mock_load = self._load_subset(KorMedMCQANurseDataset)
        mock_load.assert_called_once_with(
            _KorMedMCQABase.HUGGINGFACE_PATH, name="nurse", split="test"
        )

    def test_pharm_loads_correct_subset(self):
        _, mock_load = self._load_subset(KorMedMCQAPharmDataset)
        mock_load.assert_called_once_with(
            _KorMedMCQABase.HUGGINGFACE_PATH, name="pharm", split="test"
        )

    def test_dentist_loads_correct_subset(self):
        _, mock_load = self._load_subset(KorMedMCQADentistDataset)
        mock_load.assert_called_once_with(
            _KorMedMCQABase.HUGGINGFACE_PATH, name="dentist", split="test"
        )

    def test_doctor_dataset_name(self):
        dataset, _ = self._load_subset(KorMedMCQADoctorDataset)
        assert dataset.name == "KorMedMCQA (Doctor)"

    def test_nurse_dataset_name(self):
        dataset, _ = self._load_subset(KorMedMCQANurseDataset)
        assert dataset.name == "KorMedMCQA (Nurse)"

    def test_pharm_dataset_name(self):
        dataset, _ = self._load_subset(KorMedMCQAPharmDataset)
        assert dataset.name == "KorMedMCQA (Pharm)"

    def test_dentist_dataset_name(self):
        dataset, _ = self._load_subset(KorMedMCQADentistDataset)
        assert dataset.name == "KorMedMCQA (Dentist)"
