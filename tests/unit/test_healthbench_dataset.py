"""Tests for HealthBench dataset loaders."""

import json
from unittest.mock import MagicMock, patch

import pytest
from deepeval.dataset import ConversationalGolden
from deepeval.test_case import ConversationalTestCase

from coeval.datasets.healthbench import (
    HealthBenchConsensusDataset,
    HealthBenchMainDataset,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_sample(
    prompt_id: str = "test-001",
    prompt: list[dict] | None = None,
    rubrics: list[dict] | None = None,
    example_tags: list[str] | None = None,
) -> dict:
    """Create a mock HealthBench JSONL row."""
    if prompt is None:
        prompt = [
            {"role": "user", "content": "I have a headache. What should I do?"},
        ]
    if rubrics is None:
        rubrics = [
            {"criterion": "Recommends rest", "points": 1.0, "tags": ["advice"]},
            {"criterion": "Mentions medication", "points": 2.0, "tags": ["treatment"]},
        ]
    if example_tags is None:
        example_tags = ["general_health"]
    return {
        "prompt_id": prompt_id,
        "prompt": prompt,
        "rubrics": rubrics,
        "example_tags": example_tags,
    }


def _mock_jsonl_response(samples: list[dict]) -> bytes:
    """Create mock HTTP response bytes from sample dicts."""
    lines = [json.dumps(s) for s in samples]
    return "\n".join(lines).encode("utf-8")


# ---------------------------------------------------------------------------
# Dataset loading tests
# ---------------------------------------------------------------------------


class TestHealthBenchConsensusDataset:
    @pytest.fixture(autouse=True)
    def no_cache(self, tmp_path):
        """Point cache dir to an empty temp directory so no cached file is found."""
        with patch(
            "coeval.datasets.healthbench._CACHE_DIR",
            tmp_path / "coeval_cache",
        ):
            yield

    @patch("coeval.datasets.healthbench.urllib.request.urlopen")
    def test_loads_all_samples(self, mock_urlopen: MagicMock) -> None:
        """Should load all samples from JSONL."""
        samples = [_make_sample(f"id-{i}") for i in range(5)]
        mock_response = MagicMock()
        mock_response.read.return_value = _mock_jsonl_response(samples)
        mock_response.__enter__ = lambda s: s
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_response

        dataset = HealthBenchConsensusDataset(num_samples=None)
        assert len(dataset.goldens) == 5

    @patch("coeval.datasets.healthbench.urllib.request.urlopen")
    def test_goldens_are_conversational(self, mock_urlopen: MagicMock) -> None:
        """Goldens should be ConversationalGolden instances."""
        mock_response = MagicMock()
        mock_response.read.return_value = _mock_jsonl_response([_make_sample()])
        mock_response.__enter__ = lambda s: s
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_response

        dataset = HealthBenchConsensusDataset()
        assert isinstance(dataset.goldens[0], ConversationalGolden)

    @patch("coeval.datasets.healthbench.urllib.request.urlopen")
    def test_limits_num_samples(self, mock_urlopen: MagicMock) -> None:
        """num_samples should limit loaded goldens."""
        samples = [_make_sample(f"id-{i}") for i in range(10)]
        mock_response = MagicMock()
        mock_response.read.return_value = _mock_jsonl_response(samples)
        mock_response.__enter__ = lambda s: s
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_response

        dataset = HealthBenchConsensusDataset(num_samples=3)
        assert len(dataset.goldens) == 3

    @patch("coeval.datasets.healthbench.urllib.request.urlopen")
    def test_golden_has_correct_metadata(self, mock_urlopen: MagicMock) -> None:
        """Golden should carry rubrics, tags, and system_prompt in metadata."""
        sample = _make_sample(
            prompt_id="test-meta",
            prompt=[
                {"role": "system", "content": "You are a doctor."},
                {"role": "user", "content": "What is diabetes?"},
            ],
            rubrics=[
                {"criterion": "Explains diabetes", "points": 3.0, "tags": ["accuracy"]},
            ],
            example_tags=["endocrinology"],
        )
        mock_response = MagicMock()
        mock_response.read.return_value = _mock_jsonl_response([sample])
        mock_response.__enter__ = lambda s: s
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_response

        dataset = HealthBenchConsensusDataset()
        golden = dataset.goldens[0]

        assert golden.additional_metadata is not None
        assert golden.additional_metadata["prompt_id"] == "test-meta"
        assert len(golden.additional_metadata["rubrics"]) == 1
        assert golden.additional_metadata["rubrics"][0]["points"] == 3.0
        assert golden.additional_metadata["example_tags"] == ["endocrinology"]
        assert golden.additional_metadata["system_prompt"] == "You are a doctor."

    @patch("coeval.datasets.healthbench.urllib.request.urlopen")
    def test_golden_scenario_is_prompt_id(self, mock_urlopen: MagicMock) -> None:
        """ConversationalGolden.scenario should be the prompt_id."""
        mock_response = MagicMock()
        mock_response.read.return_value = _mock_jsonl_response(
            [_make_sample(prompt_id="my-prompt-123")]
        )
        mock_response.__enter__ = lambda s: s
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_response

        dataset = HealthBenchConsensusDataset()
        assert dataset.goldens[0].scenario == "my-prompt-123"

    @patch("coeval.datasets.healthbench.urllib.request.urlopen")
    def test_golden_turns_exclude_system(self, mock_urlopen: MagicMock) -> None:
        """Turn only supports user/assistant; system messages should be excluded."""
        sample = _make_sample(
            prompt=[
                {"role": "system", "content": "You are a doctor."},
                {"role": "user", "content": "Hello"},
            ],
        )
        mock_response = MagicMock()
        mock_response.read.return_value = _mock_jsonl_response([sample])
        mock_response.__enter__ = lambda s: s
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_response

        dataset = HealthBenchConsensusDataset()
        golden = dataset.goldens[0]

        assert golden.turns is not None
        assert len(golden.turns) == 1
        assert golden.turns[0].role == "user"
        assert golden.turns[0].content == "Hello"

    @patch("coeval.datasets.healthbench.urllib.request.urlopen")
    def test_skips_rows_without_prompt(self, mock_urlopen: MagicMock) -> None:
        """Rows with empty prompt should be skipped."""
        samples = [
            _make_sample(prompt_id="good", prompt=[{"role": "user", "content": "hi"}]),
            {"prompt_id": "bad", "prompt": [], "rubrics": [], "example_tags": []},
        ]
        mock_response = MagicMock()
        mock_response.read.return_value = _mock_jsonl_response(samples)
        mock_response.__enter__ = lambda s: s
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_response

        dataset = HealthBenchConsensusDataset()
        assert len(dataset.goldens) == 1

    @patch("coeval.datasets.healthbench.urllib.request.urlopen")
    def test_name_property(self, mock_urlopen: MagicMock) -> None:
        mock_response = MagicMock()
        mock_response.read.return_value = _mock_jsonl_response([_make_sample()])
        mock_response.__enter__ = lambda s: s
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_response

        dataset = HealthBenchConsensusDataset()
        assert dataset.name == "HealthBenchConsensus"

    @patch("coeval.datasets.healthbench.urllib.request.urlopen")
    def test_build_test_cases_returns_conversational(
        self, mock_urlopen: MagicMock
    ) -> None:
        """build_test_cases should return ConversationalTestCase with turns."""
        mock_response = MagicMock()
        mock_response.read.return_value = _mock_jsonl_response([_make_sample()])
        mock_response.__enter__ = lambda s: s
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_response

        dataset = HealthBenchConsensusDataset()
        test_cases = dataset.build_test_cases(["Test response"])

        assert len(test_cases) == 1
        tc = test_cases[0]
        assert isinstance(tc, ConversationalTestCase)
        # Last turn should be the assistant prediction
        assert tc.turns[-1].role == "assistant"
        assert tc.turns[-1].content == "Test response"
        # Metadata preserved
        assert tc.additional_metadata is not None
        assert "rubrics" in tc.additional_metadata
        assert "system_prompt" in tc.additional_metadata
        assert tc.additional_metadata["_sample_id"] == 0

    # -----------------------------------------------------------------------
    # Stratified sampling tests
    # -----------------------------------------------------------------------

    @patch("coeval.datasets.healthbench.urllib.request.urlopen")
    def test_sample_ratio_preserves_theme_distribution(
        self, mock_urlopen: MagicMock
    ) -> None:
        """sample_ratio keeps all themes with proportional counts."""
        # 3 themes: A=20, B=10, C=10 → ratio=0.5 → A=10, B=5, C=5
        samples = []
        for i in range(20):
            samples.append(
                _make_sample(f"a-{i}", example_tags=["theme:treatment_plan"])
            )
        for i in range(10):
            samples.append(_make_sample(f"b-{i}", example_tags=["theme:diagnosis"]))
        for i in range(10):
            samples.append(_make_sample(f"c-{i}", example_tags=["theme:safety"]))

        mock_response = MagicMock()
        mock_response.read.return_value = _mock_jsonl_response(samples)
        mock_response.__enter__ = lambda s: s
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_response

        dataset = HealthBenchConsensusDataset(sample_ratio=0.5)
        goldens = dataset.goldens

        # Total: round(20*0.5) + round(10*0.5) + round(10*0.5) = 10+5+5 = 20
        assert len(goldens) == 20

        # All themes must be present
        themes = {HealthBenchConsensusDataset._get_theme(g) for g in goldens}
        assert themes == {"theme:treatment_plan", "theme:diagnosis", "theme:safety"}

        # Check proportional counts
        from collections import Counter

        theme_counts = Counter(
            HealthBenchConsensusDataset._get_theme(g) for g in goldens
        )
        assert theme_counts["theme:treatment_plan"] == 10
        assert theme_counts["theme:diagnosis"] == 5
        assert theme_counts["theme:safety"] == 5

    @patch("coeval.datasets.healthbench.urllib.request.urlopen")
    def test_sample_ratio_is_reproducible(self, mock_urlopen: MagicMock) -> None:
        """Same seed produces identical samples across calls."""
        samples = [
            _make_sample(f"s-{i}", example_tags=["theme:diagnosis"]) for i in range(30)
        ]

        def _fresh_mock() -> MagicMock:
            resp = MagicMock()
            resp.read.return_value = _mock_jsonl_response(samples)
            resp.__enter__ = lambda s: s
            resp.__exit__ = MagicMock(return_value=False)
            return resp

        mock_urlopen.return_value = _fresh_mock()
        ds1 = HealthBenchConsensusDataset(sample_ratio=0.3, seed=99)

        mock_urlopen.return_value = _fresh_mock()
        ds2 = HealthBenchConsensusDataset(sample_ratio=0.3, seed=99)

        ids1 = [g.scenario for g in ds1.goldens]
        ids2 = [g.scenario for g in ds2.goldens]
        assert ids1 == ids2

    @patch("coeval.datasets.healthbench.urllib.request.urlopen")
    def test_sample_ratio_different_seed_gives_different_samples(
        self, mock_urlopen: MagicMock
    ) -> None:
        """Different seed produces different samples."""
        samples = [
            _make_sample(f"s-{i}", example_tags=["theme:diagnosis"]) for i in range(50)
        ]

        def _fresh_mock() -> MagicMock:
            resp = MagicMock()
            resp.read.return_value = _mock_jsonl_response(samples)
            resp.__enter__ = lambda s: s
            resp.__exit__ = MagicMock(return_value=False)
            return resp

        mock_urlopen.return_value = _fresh_mock()
        ds1 = HealthBenchConsensusDataset(sample_ratio=0.3, seed=1)

        mock_urlopen.return_value = _fresh_mock()
        ds2 = HealthBenchConsensusDataset(sample_ratio=0.3, seed=2)

        ids1 = {g.scenario for g in ds1.goldens}
        ids2 = {g.scenario for g in ds2.goldens}
        assert ids1 != ids2

    @patch("coeval.datasets.healthbench.urllib.request.urlopen")
    def test_num_samples_takes_priority_over_sample_ratio(
        self, mock_urlopen: MagicMock
    ) -> None:
        """When both num_samples and sample_ratio are set, num_samples wins."""
        samples = [
            _make_sample(f"id-{i}", example_tags=["theme:safety"]) for i in range(20)
        ]
        mock_response = MagicMock()
        mock_response.read.return_value = _mock_jsonl_response(samples)
        mock_response.__enter__ = lambda s: s
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_response

        # num_samples=3 → 3 samples; sample_ratio=0.5 would give 10
        dataset = HealthBenchConsensusDataset(num_samples=3, sample_ratio=0.5)
        assert len(dataset.goldens) == 3

    @patch("coeval.datasets.healthbench.urllib.request.urlopen")
    def test_sample_ratio_guarantees_at_least_one_per_theme(
        self, mock_urlopen: MagicMock
    ) -> None:
        """Even tiny ratio guarantees at least 1 sample per theme."""
        samples = [
            _make_sample("big-0", example_tags=["theme:treatment_plan"]),
            _make_sample("big-1", example_tags=["theme:treatment_plan"]),
            _make_sample("small-0", example_tags=["theme:rare_topic"]),
        ]
        mock_response = MagicMock()
        mock_response.read.return_value = _mock_jsonl_response(samples)
        mock_response.__enter__ = lambda s: s
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_response

        # ratio=0.01 → round(2*0.01)=0 → max(1,0)=1; round(1*0.01)=0 → max(1,0)=1
        dataset = HealthBenchConsensusDataset(sample_ratio=0.01)
        themes = {HealthBenchConsensusDataset._get_theme(g) for g in dataset.goldens}
        assert "theme:treatment_plan" in themes
        assert "theme:rare_topic" in themes
        assert len(dataset.goldens) == 2

    @patch("coeval.datasets.healthbench.urllib.request.urlopen")
    def test_sample_ratio_no_theme_uses_unknown(self, mock_urlopen: MagicMock) -> None:
        """Samples without theme:* tag are grouped under theme:unknown."""
        samples = [
            _make_sample(f"no-theme-{i}", example_tags=["general_health"])
            for i in range(10)
        ]
        mock_response = MagicMock()
        mock_response.read.return_value = _mock_jsonl_response(samples)
        mock_response.__enter__ = lambda s: s
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_response

        dataset = HealthBenchConsensusDataset(sample_ratio=0.5)
        assert len(dataset.goldens) == 5
        themes = {HealthBenchConsensusDataset._get_theme(g) for g in dataset.goldens}
        assert themes == {"theme:unknown"}

    @patch("coeval.datasets.healthbench.urllib.request.urlopen")
    def test_multi_turn_chat_prompt(self, mock_urlopen: MagicMock) -> None:
        """Multi-turn chat: system_prompt in metadata, turns have user/assistant only."""
        multi_turn_prompt = [
            {"role": "system", "content": "You are a medical assistant."},
            {"role": "user", "content": "I have diabetes. What should I eat?"},
            {"role": "assistant", "content": "You should eat a balanced diet."},
            {"role": "user", "content": "Can I eat fruit?"},
        ]
        sample = _make_sample(
            prompt_id="multi-turn-001",
            prompt=multi_turn_prompt,
        )
        mock_response = MagicMock()
        mock_response.read.return_value = _mock_jsonl_response([sample])
        mock_response.__enter__ = lambda s: s
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_response

        dataset = HealthBenchConsensusDataset()
        golden = dataset.goldens[0]

        # System prompt stored as string in metadata
        assert golden.additional_metadata is not None
        assert (
            golden.additional_metadata["system_prompt"]
            == "You are a medical assistant."
        )

        # Native turns exclude system role
        assert golden.turns is not None
        assert len(golden.turns) == 3
        assert golden.turns[0].role == "user"
        assert golden.turns[1].role == "assistant"
        assert golden.turns[2].role == "user"
        assert golden.turns[2].content == "Can I eat fruit?"


def _mock_urlopen(mock_urlopen: MagicMock, samples: list[dict]) -> None:
    """Wire a patched urlopen to return the given samples as a JSONL response."""
    mock_response = MagicMock()
    mock_response.read.return_value = _mock_jsonl_response(samples)
    mock_response.__enter__ = lambda s: s
    mock_response.__exit__ = MagicMock(return_value=False)
    mock_urlopen.return_value = mock_response


# ---------------------------------------------------------------------------
# HealthBench Main (oss_eval) tests
# ---------------------------------------------------------------------------


class TestHealthBenchMainDataset:
    @pytest.fixture(autouse=True)
    def no_cache(self, tmp_path):
        """Point cache dir at an empty temp directory so the real cache is untouched."""
        with patch(
            "coeval.datasets.healthbench._CACHE_DIR",
            tmp_path / "coeval_cache",
        ):
            yield

    def test_points_at_oss_eval_blob(self) -> None:
        assert HealthBenchMainDataset._URL.endswith(
            "2025-05-07-06-14-12_oss_eval.jsonl"
        )

    @patch("coeval.datasets.healthbench.urllib.request.urlopen")
    def test_loads_conversational_goldens(self, mock_urlopen: MagicMock) -> None:
        _mock_urlopen(mock_urlopen, [_make_sample(f"id-{i}") for i in range(4)])
        dataset = HealthBenchMainDataset()
        assert len(dataset.goldens) == 4
        assert isinstance(dataset.goldens[0], ConversationalGolden)
        assert dataset.name == "HealthBenchMain"

    @patch("coeval.datasets.healthbench.urllib.request.urlopen")
    def test_preserves_penalty_rubrics(self, mock_urlopen: MagicMock) -> None:
        """Negative-point criteria survive the load — the clip depends on them."""
        sample = _make_sample(
            prompt_id="penalty-001",
            rubrics=[
                {"criterion": "Accurate", "points": 5.0, "tags": ["accuracy"]},
                {
                    "criterion": "Overly verbose",
                    "points": -2.0,
                    "tags": ["cluster:concision"],
                },
            ],
        )
        _mock_urlopen(mock_urlopen, [sample])
        rubrics = HealthBenchMainDataset().goldens[0].additional_metadata["rubrics"]
        assert [r["points"] for r in rubrics] == [5.0, -2.0]


class TestHealthBenchSubsetWiring:
    def test_registry_names(self) -> None:
        assert HealthBenchMainDataset._registry_name == "healthbench_main"
        assert HealthBenchConsensusDataset._registry_name == "healthbench_consensus"

    def test_exported_from_package(self) -> None:
        import coeval.datasets as datasets_pkg

        for name in ("HealthBenchMainDataset", "HealthBenchConsensusDataset"):
            assert name in datasets_pkg.__all__
            assert hasattr(datasets_pkg, name)
