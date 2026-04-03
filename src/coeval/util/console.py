"""Rich console utilities for beautiful output."""

from typing import Any, Literal

from omegaconf import DictConfig, ListConfig
from rich import box
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.status import Status
from rich.table import Table
from rich.text import Text


class EvalConsole:
    """
    Rich console wrapper for evaluation output.

    Usage:
        from coeval.util import console
        console.banner()
        console.info("Loading...")
        console.success("Done!")
    """

    def __init__(self) -> None:
        self._console = Console()
        self._header_info: dict[str, Any] | None = None  # Store for redrawing

    def print(self, *args: Any, **kwargs: Any) -> None:
        """Proxy to rich console print."""
        self._console.print(*args, **kwargs)

    def rule(self, *args: Any, **kwargs: Any) -> None:
        """Proxy to rich console rule."""
        self._console.rule(*args, **kwargs)

    def banner(self) -> None:
        """Print the evaluation framework banner."""
        from rich.align import Align
        from rich.text import Text as RichText

        title = RichText(justify="center")
        title.append("🏥 ", style="")
        title.append("CoEval", style="bold cyan")
        title.append(" — Medical LLM Evaluation Framework", style="bold white")

        subtitle = RichText(justify="center")
        subtitle.append("📋 18 Datasets  ", style="bold magenta")
        subtitle.append("•  ", style="dim")
        subtitle.append("📊 9 Metrics  ", style="bold yellow")
        subtitle.append("•  ", style="dim")
        subtitle.append("⚡ Async Pipeline", style="bold green")

        tagline = RichText(
            "Benchmarking LLM-powered applications",
            style="dim italic",
            justify="center",
        )

        panel = Panel(
            Align.center(
                RichText.assemble(title, "\n", subtitle, "\n", tagline),
            ),
            box=box.DOUBLE,
            border_style="bright_blue",
            padding=(1, 2),
        )
        self._console.print(panel)

    def dataset_overview(
        self,
        dataset_info: list[tuple[str, int | None]],
        model_name: str,
    ) -> None:
        """
        Print dataset overview panel showing evaluation queue.

        Args:
            dataset_info: List of (dataset_name, num_samples) tuples
            model_name: Name of the model being evaluated
        """
        # Store for potential redraw
        self._header_info = {"datasets": dataset_info, "model": model_name}

        # Print model info
        self._console.print(f"[bold]🤖 Model:[/bold] [cyan]{model_name}[/cyan]")
        self._console.print()

        table = Table(
            title="[bold white]📋 Evaluation Queue[/bold white]",
            box=box.ROUNDED,
            show_header=True,
            header_style="bold cyan",
            title_justify="center",
        )
        table.add_column("#", style="dim", width=6, justify="right")
        table.add_column("DATASET", style="yellow", justify="left", max_width=20)
        table.add_column("SAMPLES", style="green", justify="right", width=14)

        total_samples = 0
        for i, (name, samples) in enumerate(dataset_info, 1):
            sample_str = str(samples) if samples else "all"
            if samples:
                total_samples += samples
            table.add_row(str(i), name, sample_str)

        # Add footer row with totals
        table.add_section()
        table.add_row(
            "",
            f"[bold]{len(dataset_info)} datasets[/bold]",
            f"[bold]{total_samples if total_samples else 'all'}[/bold]",
        )

        self._console.print(table)
        self._console.print()

    def dataset_header(
        self,
        dataset_name: str,
        dataset_idx: int,
        total_datasets: int,
        num_samples: int,
    ) -> None:
        """
        Print header for current dataset being processed.

        Args:
            dataset_name: Name of the dataset
            dataset_idx: Current dataset index (1-based)
            total_datasets: Total number of datasets
            num_samples: Number of samples in this dataset
        """
        self._console.print()
        header_text = (
            f"[bold white][{dataset_idx}/{total_datasets}][/bold white]    "
            f"[bold cyan]🧰 {dataset_name.upper()}[/bold cyan]"
        )
        panel = Panel(
            header_text,
            box=box.HEAVY,
            border_style="cyan",
            padding=(0, 2),
        )
        self._console.print(panel)
        self._console.print(
            f"[green]▶[/green] Starting evaluation on "
            f"[bold magenta]{num_samples}[/bold magenta] samples from "
            f"[cyan]{dataset_name}[/cyan]"
        )

    _LEADERBOARD_EXCLUDE_PREFIXES = (
        "theme:",
        "physician_agreed_category:",
        "cluster:",
    )

    def leaderboard(self, results: list[dict[str, Any]]) -> None:
        """
        Print final results summary after all datasets complete.

        Args:
            results: List of dicts with dataset, metric_scores, samples, time
        """
        self._console.print()
        self._console.rule("[bold white]🏆 FINAL RESULTS[/bold white]", style="white")
        self._console.print()

        all_metrics: list[str] = []
        for r in results:
            for metric_name in r.get("metric_scores", {}):
                if metric_name not in all_metrics and not metric_name.startswith(
                    self._LEADERBOARD_EXCLUDE_PREFIXES
                ):
                    all_metrics.append(metric_name)

        table = Table(
            box=box.ROUNDED,
            show_header=True,
            header_style="bold",
        )
        table.add_column("DATASET", style="white", justify="left", max_width=18)

        for metric_name in all_metrics:
            table.add_column(
                metric_name.upper().replace("_", " "),
                justify="right",
                width=12,
            )

        table.add_column("SAMPLES", style="dim", justify="right", width=10)
        table.add_column("TIME", style="dim", justify="right", width=10)

        total_samples = 0
        total_time = 0.0
        metric_totals: dict[str, list[float]] = {m: [] for m in all_metrics}

        for r in results:
            metric_scores = r.get("metric_scores", {})
            samples = r.get("samples", 0)
            time_s = r.get("time", 0)

            row_data = [r.get("dataset", "unknown")]

            for metric_name in all_metrics:
                import math

                score = metric_scores.get(metric_name)
                if score is not None and not math.isnan(score):
                    style = (
                        "green" if score >= 0.7 else "yellow" if score >= 0.5 else "red"
                    )
                    row_data.append(f"[{style}]{score:.3f}[/{style}]")
                    metric_totals[metric_name].append(score)
                else:
                    row_data.append("[dim]-[/dim]")

            row_data.extend([str(samples), f"{time_s:.1f}s"])
            table.add_row(*row_data)

            total_samples += samples
            total_time += time_s

        table.add_section()
        total_row = ["[bold]AVG[/bold]"]

        for metric_name in all_metrics:
            scores = metric_totals[metric_name]
            if scores:
                avg = sum(scores) / len(scores)
                style = "green" if avg >= 0.7 else "yellow" if avg >= 0.5 else "red"
                total_row.append(f"[bold {style}]{avg:.3f}[/bold {style}]")
            else:
                total_row.append("[dim]-[/dim]")

        total_row.extend(
            [f"[bold]{total_samples}[/bold]", f"[bold]{total_time:.1f}s[/bold]"]
        )
        table.add_row(*total_row)

        self._console.print(table)
        self._console.print()

    def clear(self) -> None:
        """Clear the console."""
        self._console.clear()

    def config(self, cfg: dict[str, Any], title: str = "Configuration") -> None:
        """Print configuration as a formatted table with uppercase keys."""
        table = Table(
            title=f"[bold]{title}[/bold]",
            box=box.ROUNDED,
            show_header=True,
            header_style="bold cyan",
            title_style="bold white",
        )
        table.add_column("PARAMETER", style="cyan", justify="right")
        table.add_column("VALUE", style="green")

        for key, value in cfg.items():
            display_key = key.upper().replace("_", " ")
            if isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    sub_display = f"{display_key}.{sub_key.upper()}"
                    table.add_row(sub_display, str(sub_value))
            elif isinstance(value, list):
                # Display each list item in its own row
                for i, item in enumerate(value):
                    row_key = display_key if i == 0 else ""
                    table.add_row(row_key, str(item))
            else:
                if key == "num_samples" and value == "all":
                    table.add_row(display_key, "[bold yellow]ALL[/bold yellow]")
                elif key == "structured_output" and value:
                    table.add_row(display_key, "[bold green]JSON[/bold green]")
                elif key == "temperature" and value == 0.0:
                    table.add_row(display_key, "[dim]0.0 (deterministic)[/dim]")
                else:
                    table.add_row(display_key, str(value))

        self._console.print(table)

    def set_config(self, cfg: Any) -> None:
        """Display evaluation configuration from Hydra DictConfig."""
        # Store config for redrawing later
        self._stored_cfg = cfg

        # Display model/client information
        self._print_model_info(cfg)

        # Build a single table for all datasets
        table = Table(
            title="[bold white]📋 Evaluation Queue[/bold white]",
            box=box.ROUNDED,
            show_header=True,
            header_style="bold cyan",
            title_justify="center",
        )
        table.add_column("#", style="cyan", justify="right")
        table.add_column("DATASET", style="green")
        table.add_column("SAMPLES", style="dim", justify="right")

        total_samples = 0
        row_idx = 0
        for name, dataset_cfg in cfg.datasets.items():
            if "_target_" in dataset_cfg:
                # Leaf dataset
                row_idx += 1
                num_samples = dataset_cfg.get("num_samples", None)
                if num_samples is None:
                    target = dataset_cfg.get("_target_", "")
                    samples_display = self._get_dataset_total_samples(target)
                else:
                    samples_display = num_samples
                table.add_row(str(row_idx), name, str(samples_display))
                total_samples += (
                    samples_display if isinstance(samples_display, int) else 0
                )
            else:
                # Nested group — show group header then children
                table.add_row("", f"[bold yellow]{name}[/bold yellow]", "")
                for child_name, child_cfg in dataset_cfg.items():
                    if not hasattr(child_cfg, "get"):
                        continue
                    row_idx += 1
                    num_samples = child_cfg.get("num_samples", None)
                    if num_samples is None:
                        target = child_cfg.get("_target_", "")
                        samples_display = self._get_dataset_total_samples(target)
                    else:
                        samples_display = num_samples
                    table.add_row(str(row_idx), f"  {child_name}", str(samples_display))
                    total_samples += (
                        samples_display if isinstance(samples_display, int) else 0
                    )

        # Add summary row
        num_datasets = row_idx
        table.add_section()
        table.add_row(
            "", f"[bold]{num_datasets} datasets[/bold]", f"[bold]{total_samples}[/bold]"
        )

        self._console.print(table)
        self._console.print()

    def _print_model_info(self, cfg: Any) -> None:
        """Print model/client information from config."""
        # Try to extract model info from runner.client or client config
        model_name = "unknown"
        api_base = ""
        client_type = ""

        # Check runner config for client info
        if hasattr(cfg, "runner") and cfg.runner:
            runner_cfg = cfg.runner
            if hasattr(runner_cfg, "client") and runner_cfg.client:
                client_cfg = runner_cfg.client
                model_name = getattr(client_cfg, "model", model_name)
                api_base = getattr(client_cfg, "api_base", "")
                # Extract client type from _target_
                target = getattr(client_cfg, "_target_", "")
                if target:
                    client_type = target.split(".")[-1].replace("Client", "")

        # Also check top-level client config (some configs use this)
        if hasattr(cfg, "client") and cfg.client:
            client_cfg = cfg.client
            if hasattr(client_cfg, "model"):
                model_name = client_cfg.model
            if hasattr(client_cfg, "api_base"):
                api_base = client_cfg.api_base
            target = getattr(client_cfg, "_target_", "")
            if target:
                client_type = target.split(".")[-1].replace("Client", "")

        # Build model info panel
        info_parts = []
        if client_type:
            info_parts.append(f"[dim]Client:[/dim] [yellow]{client_type}[/yellow]")
        info_parts.append(f"[dim]Model:[/dim] [bold cyan]{model_name}[/bold cyan]")
        if api_base:
            # Truncate long URLs
            display_url = api_base if len(api_base) <= 50 else api_base[:47] + "..."
            info_parts.append(f"[dim]Endpoint:[/dim] [dim]{display_url}[/dim]")

        model_panel = Panel(
            "  ".join(info_parts),
            title="[bold white]🤖 Model Under Test[/bold white]",
            box=box.ROUNDED,
            border_style="cyan",
            padding=(0, 1),
        )
        self._console.print(model_panel)
        self._console.print()

    def _get_dataset_total_samples(self, target: str) -> int | str:
        """Get TOTAL_SAMPLES from dataset class target string."""
        try:
            # Parse target like "coeval.datasets.MedQADataset"
            if not target:
                return "all"
            parts = target.rsplit(".", 1)
            if len(parts) != 2:
                return "all"
            module_path, class_name = parts
            import importlib

            module = importlib.import_module(module_path)
            dataset_class = getattr(module, class_name, None)
            if dataset_class and hasattr(dataset_class, "TOTAL_SAMPLES"):
                return dataset_class.TOTAL_SAMPLES
            return "all"
        except Exception:
            return "all"

    def summary(self, summary: dict[str, Any]) -> None:
        """Print evaluation summary with Rich formatting."""
        self._console.print()

        num_passed = summary.get("num_passed", 0)
        num_samples = summary.get("num_samples", 1)
        pass_rate = num_passed / num_samples if num_samples > 0 else 0
        pass_style = (
            "green" if pass_rate >= 0.7 else "yellow" if pass_rate >= 0.5 else "red"
        )

        summary_table = Table(
            title="[bold white]📊 Evaluation Summary[/bold white]",
            box=box.ROUNDED,
            show_header=True,
            header_style="bold",
            title_justify="center",
        )
        summary_table.add_column("", style="bold cyan", justify="right", width=18)
        summary_table.add_column("", justify="left")

        summary_table.add_row(
            "DATASET",
            f"[bold yellow]{summary.get('dataset', 'unknown')}[/bold yellow]",
        )
        summary_table.add_row(
            "SAMPLES",
            f"[white]{summary.get('num_samples', 0)}[/white]",
        )
        summary_table.add_row(
            "PASSED",
            f"[bold {pass_style}]{num_passed}/{num_samples} ({pass_rate:.1%})[/bold {pass_style}]",
        )

        self._console.print(summary_table)
        self._console.print()

        perf_table = Table(
            title="[bold white]⏱️  Performance[/bold white]",
            box=box.ROUNDED,
            show_header=True,
            header_style="bold",
            title_justify="center",
        )
        perf_table.add_column("", style="bold magenta", justify="right", width=18)
        perf_table.add_column("", justify="left")

        perf_table.add_row(
            "TOTAL TIME",
            f"[white]{summary.get('total_time_s', 0):.2f}s[/white]",
        )
        perf_table.add_row(
            "AVG GENERATION",
            f"[dim]{summary.get('avg_generation_ms', 0):.1f}ms[/dim]",
        )
        perf_table.add_row(
            "AVG SCORING",
            f"[dim]{summary.get('avg_scoring_ms', 0):.1f}ms[/dim]",
        )

        self._console.print(perf_table)

        metric_scores = summary.get("metric_scores", {})
        if metric_scores:
            self._console.print()

            metric_table = Table(
                title="[bold white]🎯 Metric Scores[/bold white]",
                box=box.ROUNDED,
                show_header=True,
                header_style="bold",
                title_justify="center",
            )
            metric_table.add_column(
                "METRIC", style="bold blue", justify="right", width=18
            )
            metric_table.add_column("SCORE", justify="left", width=12)
            metric_table.add_column("", justify="left", width=20)

            for name, score in metric_scores.items():
                import math

                if score is None or math.isnan(score):
                    metric_table.add_row(
                        name.upper().replace("_", " "),
                        "[dim]N/A[/dim]",
                        "[dim]──────────[/dim]",
                    )
                    continue
                score_style = (
                    "green" if score >= 0.7 else "yellow" if score >= 0.5 else "red"
                )
                bar_width = int(score * 10)
                bar = "█" * bar_width + "░" * (10 - bar_width)
                metric_table.add_row(
                    name.upper().replace("_", " "),
                    f"[bold {score_style}]{score:.3f}[/bold {score_style}]",
                    f"[{score_style}]{bar}[/{score_style}]",
                )

            self._console.print(metric_table)

        self._console.print()

    def metric_scores(
        self,
        dataset_name: str,
        metric_scores: dict[str, float],
        breakdown: dict[str, dict[str, Any]] | None = None,
    ) -> None:
        """Print only metric scores for a dataset.

        For datasets with hierarchical sub-scores (theme/category/cluster),
        only the top-level metric and ``theme:*`` entries are shown in the
        main table.  Deeper breakdowns (``physician_agreed_category:*``,
        ``cluster:*``) are omitted to keep the output readable.
        """
        if not metric_scores:
            return

        # Detect hierarchical HealthBench-style sub-scores and filter
        has_themes = any(k.startswith("theme:") for k in metric_scores)
        if has_themes:
            display_scores = {
                k: v
                for k, v in metric_scores.items()
                if not k.startswith(("physician_agreed_category:", "cluster:"))
            }
        else:
            display_scores = metric_scores

        metric_table = Table(
            title=f"[bold white]🎯 {dataset_name} - Metric Scores[/bold white]",
            box=box.ROUNDED,
            show_header=True,
            header_style="bold",
            title_justify="center",
        )
        metric_table.add_column("METRIC", style="bold blue", justify="right", width=18)
        metric_table.add_column("SCORE", justify="left", width=12)
        metric_table.add_column("", justify="left", width=20)

        for name, score in display_scores.items():
            import math
            import re

            display = re.sub(r"\s*\[.*?\]\s*$", "", name).upper().replace("_", " ")

            if score is None or math.isnan(score):
                metric_table.add_row(
                    display,
                    "[dim]N/A[/dim]",
                    "[dim]──────────[/dim]",
                )
                continue
            score_style = (
                "green" if score >= 0.7 else "yellow" if score >= 0.5 else "red"
            )
            bar_width = int(score * 10)
            bar = "█" * bar_width + "░" * (10 - bar_width)
            metric_table.add_row(
                display,
                f"[bold {score_style}]{score:.3f}[/bold {score_style}]",
                f"[{score_style}]{bar}[/{score_style}]",
            )

        self._console.print(metric_table)

        if breakdown:
            self._print_classification_breakdown(breakdown)

        self._console.print()

    def _print_classification_breakdown(
        self, breakdown: dict[str, dict[str, Any]]
    ) -> None:
        """Print per-class precision/recall/F1 from classification report."""
        for metric_name, report in breakdown.items():
            # ordinal_classification_aggregator nests the sklearn report
            # under a "classification_report" key; unwrap it.
            if "classification_report" in report and isinstance(
                report["classification_report"], dict
            ):
                report = report["classification_report"]

            # Skip non-classification breakdowns (e.g. avg_aggregator
            # produces {mean, n_samples, min_score, max_score}).
            has_clf_data = any(
                isinstance(v, dict) and "precision" in v for v in report.values()
            )
            if not has_clf_data:
                continue

            bt = Table(
                title=f"[dim]{metric_name} - Classification Report[/dim]",
                box=box.SIMPLE,
                show_header=True,
                header_style="bold dim",
                padding=(0, 1),
            )
            bt.add_column("CLASS", style="dim", justify="right", width=20)
            bt.add_column("PRECISION", justify="center", width=10)
            bt.add_column("RECALL", justify="center", width=10)
            bt.add_column("F1", justify="center", width=10)
            bt.add_column("SUPPORT", justify="center", width=8)

            for label, values in report.items():
                if not isinstance(values, dict):
                    continue
                p = values.get("precision", 0)
                r = values.get("recall", 0)
                f1 = values.get("f1-score", 0)
                support = values.get("support", 0)
                f1_style = "green" if f1 >= 0.7 else "yellow" if f1 >= 0.5 else "red"
                bt.add_row(
                    label,
                    f"{p:.3f}",
                    f"{r:.3f}",
                    f"[{f1_style}]{f1:.3f}[/{f1_style}]",
                    str(int(support)),
                )

            self._console.print(bt)

    def results(self, results: list[dict[str, Any]], max_rows: int = 10) -> None:
        """Print results as a formatted table."""
        self._console.print()
        self._console.rule("[bold cyan]Results[/bold cyan]", style="cyan")
        self._console.print()

        num_passed = sum(1 for r in results if r.get("passed"))
        num_failed = len(results) - num_passed

        table = Table(
            title=f"[bold white]📋 Sample Results[/bold white]  [dim]({num_passed} ✓ / {num_failed} ✗)[/dim]",
            box=box.ROUNDED,
            show_header=True,
            header_style="bold cyan",
            title_justify="center",
        )
        table.add_column("#", style="dim", width=5, justify="center")
        table.add_column("STATUS", width=8, justify="center")
        table.add_column("SCORE", width=8, justify="center")
        table.add_column("METHOD", style="magenta", width=16, justify="left")
        table.add_column("REASON", max_width=45)

        for i, r in enumerate(results[:max_rows]):
            passed = r.get("passed", False)
            status = "✓ PASS" if passed else "✗ FAIL"
            status_style = "bold green" if passed else "bold red"

            metrics = r.get("metrics", [{}])
            first_metric = metrics[0] if metrics else {}

            score = first_metric.get("score", 0)
            score_style = (
                "green" if score >= 0.7 else "yellow" if score >= 0.5 else "red"
            )

            method = first_metric.get("details", {}).get("method", "-")
            reason = first_metric.get("reason", "")[:45]

            table.add_row(
                str(r.get("sample_id", i)),
                Text(status, style=status_style),
                Text(f"{score:.2f}", style=score_style),
                method.upper().replace("_", " ") if method != "-" else "-",
                f"[dim]{reason}[/dim]",
            )

        if len(results) > max_rows:
            remaining = len(results) - max_rows
            table.add_row(
                "...",
                "",
                "",
                "",
                f"[dim italic]and {remaining} more samples[/dim italic]",
            )

        self._console.print(table)

    def progress(self) -> Progress:
        """Create a Rich progress bar for evaluation."""
        return Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=self._console,
        )

    def multi_progress(self) -> Progress:
        """Create a dual-level progress bar for multi-dataset evaluation."""
        return Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(bar_width=40),
            TaskProgressColumn(),
            TextColumn("•"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=self._console,
            expand=True,
        )

    def load_with_status(
        self,
        datasets_cfg: DictConfig,
        instantiate_fn: Any,
    ) -> list[Any]:
        """
        Load datasets with a live status display.

        Args:
            datasets_cfg: Hydra dataset configuration
            instantiate_fn: Function to instantiate datasets (hydra.utils.instantiate)

        Returns:
            List of instantiated dataset objects
        """
        # DictConfig.keys() is typed broadly, so normalize to plain str keys for type checkers
        dataset_names: list[str] = [str(k) for k in datasets_cfg]
        datasets: list[Any] = []

        self._console.print("[bold blue]ℹ Loading datasets...[/bold blue]")

        Status = Literal["loading", "done"]
        status_lines: list[tuple[str, Status]] = []

        with Live(console=self._console, refresh_per_second=10) as live:
            for name in dataset_names:
                # Update display with current loading status
                display: list[str] = []
                for prev_name, status in status_lines:
                    if status == "done":
                        display.append(f"  [green]✓[/green] [cyan]{prev_name}[/cyan]")
                    else:
                        display.append(
                            f"  [yellow]⠋[/yellow] [dim]{prev_name}[/dim] [dim italic]loading...[/dim italic]"
                        )

                # Add current loading item
                display.append(
                    f"  [yellow]⠋[/yellow] [yellow]{name}[/yellow] [dim italic]loading...[/dim italic]"
                )
                live.update(Text.from_markup("\n".join(display)))

                # Actually load the dataset
                ds = instantiate_fn(datasets_cfg[name])
                datasets.append(ds)

                # Mark as done
                status_lines.append((name, "done"))

                # Update display with completed status
                done_display: list[str] = [
                    f"  [green]✓[/green] [cyan]{prev_name}[/cyan]"
                    for prev_name, _ in status_lines
                ]
                live.update(Text.from_markup("\n".join(done_display)))

        self._console.print(
            f"[bold green]✓[/bold green] Loaded [magenta]{len(datasets)}[/magenta] datasets"
        )
        return datasets

    def error(self, message: str) -> None:
        """Print an error message."""
        self._console.print(f"[bold red]✗ Error:[/bold red] {message}")

    def success(self, message: str) -> None:
        """Print a success message."""
        self._console.print(f"[bold green]✓[/bold green] {message}")

    def info(self, message: str) -> None:
        """Print an info message."""
        self._console.print(f"[bold blue]ℹ[/bold blue] {message}")

    def warn(self, message: str) -> None:
        """Print a warning message."""
        self._console.print(f"[bold yellow]⚠[/bold yellow] {message}")

    def start_eval(self, num_samples: int, dataset_name: str) -> None:
        """Print evaluation start message (no-op, handled by dataset_header)."""
        pass  # Message already shown in dataset_header

    def end_eval(
        self,
        num_passed: int,
        num_total: int,
        pass_rate: float,
        time_s: float,
        num_inference_failed: int = 0,
    ) -> None:
        """Print evaluation completion message with highlighted stats."""
        rate_style = (
            "green" if pass_rate >= 0.7 else "yellow" if pass_rate >= 0.5 else "red"
        )
        num_evaluated = num_total - num_inference_failed
        msg = (
            f"[bold cyan]✓ Evaluation complete:[/bold cyan] "
            f"[bold {rate_style}]{num_passed}/{num_evaluated}[/bold {rate_style}] passed "
            f"([bold {rate_style}]{pass_rate:.1%}[/bold {rate_style}]) "
            f"in [bold white]{time_s:.1f}s[/bold white]"
        )
        if num_inference_failed > 0:
            msg += f" [bold red]({num_inference_failed} inference failures)[/bold red]"
        self._console.print(msg)

    def saved(self, path: str) -> None:
        """Print file saved message."""
        self._console.print(f"\n[bold green]💾 Saved:[/bold green] [dim]{path}[/dim]")

    def status(self, message: str) -> Status:
        """Create a live status context."""
        return self._console.status(message, spinner="dots")

    def get_target_names(self, items: list | ListConfig) -> str:
        """Extract class names from _target_ fields for display."""
        names = []
        for item in items:
            if hasattr(item, "_target_"):
                name = (
                    item._target_.split(".")[-1]
                    .replace("Metric", "")
                    .replace("Dataset", "")
                )
                names.append(name)
        return ", ".join(names) if names else "unknown"


console = EvalConsole()
