"""Auto-ChemInstruct CLI — Typer-based command-line interface.

Commands:
    generate  — Generate reaction hypotheses
    verify    — Physically validate hypotheses
    reflect   — Generate causal reflection traces
    compile   — Build DPO/RLHF preference pairs
    pipeline  — Run the full end-to-end pipeline
    config    — Show current configuration
"""

from __future__ import annotations

from uuid import UUID

import typer
from rich.console import Console
from rich.table import Table

from src.config import load_config
from src.logger import setup_logging
from src.pipeline.orchestrator import PipelineOrchestrator

app = typer.Typer(
    name="auto-chem",
    help="Auto-ChemInstruct: Agent-Driven Synthesization of RLHF Data for Chemistry DSLMs",
    add_completion=False,
)
console = Console()


@app.command()
def generate(
    num_hypotheses: int = typer.Option(10, "--num", "-n", help="Number of hypotheses to generate"),
    temperature: float | None = typer.Option(
        None, "--temperature", "-t", help="LLM temperature override"
    ),
    config_path: str | None = typer.Option(None, "--config", "-c", help="Path to config file"),
):
    """Generate reaction hypotheses using the Hypothesis Agent."""
    config = load_config(config_path)
    setup_logging(level=config.logging.level)

    if temperature is not None:
        config.hypothesis_agent.temperature = temperature

    orch = PipelineOrchestrator(config)
    session_id = orch.start_session()

    hypotheses = orch._generate_hypotheses(num_hypotheses=num_hypotheses)

    console.print(f"[green]Generated {len(hypotheses)} hypotheses[/green]")
    console.print(f"Session ID: {session_id}")

    table = Table(title="Generated Hypotheses")
    table.add_column("ID", style="dim")
    table.add_column("Reactants")
    table.add_column("Products")
    table.add_column("Type")

    for h in hypotheses[-10:]:
        table.add_row(
            str(h.id)[:8],
            ", ".join(r.smiles[:30] for r in h.reactants),
            ", ".join(p.smiles[:30] for p in h.products),
            h.reaction_type.value,
        )

    console.print(table)


@app.command()
def verify(
    config_path: str | None = typer.Option(None, "--config", "-c", help="Path to config file"),
    session: str | None = typer.Option(None, "--session", "-s", help="Session UUID to resume"),
    list_sessions: bool = typer.Option(False, "--list", "-l", help="List available sessions"),
):
    """Verify previously generated hypotheses (load from checkpoint)."""
    config = load_config(config_path)
    setup_logging(level=config.logging.level)

    if list_sessions:
        _show_sessions()
        return

    if not session:
        console.print(
            "[yellow]No --session specified. Use --list to see available sessions.[/yellow]"
        )
        return

    try:
        session_id = UUID(session)
    except ValueError:
        console.print(f"[red]Invalid session UUID: {session}[/red]")
        raise typer.Exit(1)

    orch = PipelineOrchestrator(config)
    results = orch.verify_session(session_id)
    passed = sum(1 for r in results if r.status.value == "passed")
    console.print(f"[green]Verification complete: {len(results)} verified, {passed} passed[/green]")


@app.command()
def reflect(
    config_path: str | None = typer.Option(None, "--config", "-c", help="Path to config file"),
    session: str | None = typer.Option(None, "--session", "-s", help="Session UUID to resume"),
    list_sessions: bool = typer.Option(False, "--list", "-l", help="List available sessions"),
):
    """Generate reflection traces for failed reactions."""
    config = load_config(config_path)
    setup_logging(level=config.logging.level)

    if list_sessions:
        _show_sessions()
        return

    if not session:
        console.print(
            "[yellow]No --session specified. Use --list to see available sessions.[/yellow]"
        )
        return

    try:
        session_id = UUID(session)
    except ValueError:
        console.print(f"[red]Invalid session UUID: {session}[/red]")
        raise typer.Exit(1)

    orch = PipelineOrchestrator(config)
    traces = orch.reflect_session(session_id)
    console.print(f"[green]Reflection complete: {len(traces)} traces generated[/green]")


@app.command()
def compile(
    config_path: str | None = typer.Option(None, "--config", "-c", help="Path to config file"),
    session: str | None = typer.Option(None, "--session", "-s", help="Session UUID to resume"),
    list_sessions: bool = typer.Option(False, "--list", "-l", help="List available sessions"),
):
    """Compile verified data into DPO/RLHF preference pairs."""
    config = load_config(config_path)
    setup_logging(level=config.logging.level)

    if list_sessions:
        _show_sessions()
        return

    if not session:
        console.print(
            "[yellow]No --session specified. Use --list to see available sessions.[/yellow]"
        )
        return

    try:
        session_id = UUID(session)
    except ValueError:
        console.print(f"[red]Invalid session UUID: {session}[/red]")
        raise typer.Exit(1)

    orch = PipelineOrchestrator(config)
    compilation = orch.compile_session(session_id)
    pairs = compilation.get("pairs", [])
    console.print(f"[green]Compilation complete: {len(pairs)} preference pairs[/green]")


@app.command()
def pipeline(
    num_hypotheses: int = typer.Option(
        100, "--num-hypotheses", "-n", help="Number of hypotheses to generate"
    ),
    batch_size: int | None = typer.Option(None, "--batch-size", "-b", help="Batch size override"),
    bootstrap_iterations: int = typer.Option(
        1, "--bootstrap", "-B", help="Self-bootstrapping iterations (1=off, 2+=on)"
    ),
    config_path: str | None = typer.Option(None, "--config", "-c", help="Path to config file"),
    seed: int | None = typer.Option(None, "--seed", "-s", help="Random seed override"),
):
    """Run the full Auto-ChemInstruct pipeline end-to-end."""
    config = load_config(config_path)
    setup_logging(level=config.logging.level)

    if batch_size is not None:
        config.pipeline.batch_size = batch_size
    if seed is not None:
        config.pipeline.seed = seed

    orch = PipelineOrchestrator(config)

    console.print("[bold blue]Auto-ChemInstruct Pipeline[/bold blue]")
    console.print(f"LLM Provider: [cyan]{config.llm.provider}[/cyan] ({config.llm.model})")
    console.print(f"Target hypotheses: {num_hypotheses}")
    console.print(f"Batch size: {config.pipeline.batch_size}")
    if bootstrap_iterations > 1:
        console.print(
            f"[green]Self-bootstrapping: [bold]{bootstrap_iterations} iterations[/bold][/green]"
        )
    console.print()

    result = orch.run_pipeline(
        num_hypotheses=num_hypotheses,
        bootstrap_iterations=bootstrap_iterations,
    )

    summary = result["summary"]
    console.print()
    console.print("[bold green]Pipeline Complete![/bold green]")

    table = Table(title="Summary")
    table.add_column("Metric", style="bold")
    table.add_column("Value")

    table.add_row("Hypotheses Generated", str(summary["hypotheses_generated"]))
    table.add_row("Passed Verification", str(summary["hypotheses_passed"]))
    table.add_row("Failed Verification", str(summary["hypotheses_failed"]))
    table.add_row("Reflection Traces", str(summary["reflections_generated"]))
    table.add_row("Preference Pairs", str(summary["pairs_compiled"]))
    table.add_row(
        "Pass Rate",
        f"{summary['hypotheses_passed'] / max(1, summary['hypotheses_generated']) * 100:.1f}%",
    )
    table.add_row("Session ID", str(result["session_id"]))

    console.print(table)

    if summary["pairs_compiled"] > 0:
        output_dir = f"datasets/autochem-{result['session_id']}"
        console.print(f"\nDataset saved to: [bold]{output_dir}[/bold]")


@app.command()
def ablation(
    num_hypotheses: int = typer.Option(5, "--num-hypotheses", "-n", help="Hypotheses per variant"),
    config_path: str | None = typer.Option(None, "--config", "-c", help="Path to config file"),
    output_dir: str | None = typer.Option(
        None, "--output", "-o", help="Output directory for reports"
    ),
    mode: str = typer.Option(
        "pipeline",
        "-m",
        "--mode",
        help="Ablation mode: pipeline (4 variants) or evolution (7 variants)",
    ),
) -> None:
    """Run ablation study comparing architectural variants.

    pipeline: Baseline → Bootstrap-Only → Temp-Schedule-Only → Full-System
    evolution: 7 variants including MAP-Elites, CARL, and combined.
    """
    config = load_config(config_path)
    setup_logging(level=config.logging.level)

    if mode == "evolution":
        from src.benchmarks.evolution_ablation import run_evolution_ablation

        console.print("[bold blue]Evolution Ablation Study (7 variants)[/bold blue]")
        output = output_dir or "benchmarks"
        report = run_evolution_ablation(
            config=config,
            generations=5,
            output_dir=output,
        )
        console.print()
        console.print("[bold green]Evolution Ablation Complete![/bold green]")
        console.print(report.summary_table())
        console.print(f"\nReport saved to: [bold]{output}/evolution_ablation.json[/bold]")
        return

    from src.benchmarks.ablation import run_full_ablation

    console.print("[bold blue]Auto-ChemInstruct Ablation Study[/bold blue]")
    console.print(f"LLM Provider: [cyan]{config.llm.provider}[/cyan] ({config.llm.model})")
    console.print(f"Hypotheses per variant: {num_hypotheses}")
    console.print()

    output = output_dir or "benchmarks"
    report = run_full_ablation(
        config=config,
        num_hypotheses=num_hypotheses,
        output_dir=output,
    )

    console.print()
    console.print("[bold green]Ablation Study Complete![/bold green]")
    console.print(report.summary_table())
    console.print()
    console.print(f"Full report saved to: [bold]{output}/ablation_report.json[/bold]")


@app.command()
def chemcot(
    dataset_dir: str = typer.Option(..., "--dataset", "-d", help="Path to dataset directory"),
    output_file: str | None = typer.Option(None, "--output", "-o", help="Output JSON file path"),
) -> None:
    """Compare compiled dataset against ChemCoTBench metrics."""
    from src.benchmarks.chemcot_comparison import compare_to_chemcot

    setup_logging(level="INFO")

    console.print("[bold blue]ChemCoTBench Comparison[/bold blue]")
    console.print(f"Dataset: [cyan]{dataset_dir}[/cyan]")
    console.print()

    result = compare_to_chemcot(dataset_dir, output_file)

    console.print()
    console.print(f"Report: [bold]{output_file or 'chemcot_comparison.json'}[/bold]")


@app.command()
def config_cmd(
    config_path: str | None = typer.Option(None, "--config", "-c", help="Path to config file"),
):
    """Show the current configuration."""
    config = load_config(config_path)

    console.print("[bold]Auto-ChemInstruct Configuration[/bold]")
    console.print(f"[bold]LLM:[/bold] {config.llm.provider} → {config.llm.model}")
    console.print(f"[bold]Provider URL:[/bold] {config.llm.base_url}")
    console.print(
        f"[bold]Pipeline:[/bold] batch={config.pipeline.batch_size}, seed={config.pipeline.seed}"
    )
    console.print(
        f"[bold]Chemistry:[/bold] xTB={'enabled' if config.verification_agent.enable_xtb else 'disabled'}"
    )

    import json as json_module

    console.print("\n[bold]Full config:[/bold]")
    console.print_json(json_module.dumps(config.model_dump(), indent=2, default=str))


@app.command()
def status(
    session_id: str | None = typer.Option(None, "--session", "-s", help="Session ID to check"),
    list_sessions: bool = typer.Option(False, "--list", "-l", help="List available sessions"),
):
    """Check pipeline status."""
    if list_sessions:
        _show_sessions()
        return

    if not session_id:
        console.print(
            "[yellow]No --session specified. Use --list to see available sessions.[/yellow]"
        )
        return

    try:
        sid = UUID(session_id)
    except ValueError:
        console.print(f"[red]Invalid session UUID: {session_id}[/red]")
        raise typer.Exit(1)

    config = load_config(None)
    orch = PipelineOrchestrator(config)
    try:
        orch.resume_session(sid)
        status_val = orch.session.status
        console.print(f"Session: [cyan]{sid}[/cyan]")
        console.print(f"Status: [bold]{status_val.value}[/bold]")
        console.print(f"Hypotheses: {orch.session.hypotheses_generated}")
        console.print(f"Passed: {orch.session.hypotheses_passed}")
        console.print(f"Failed: {orch.session.hypotheses_failed}")
        console.print(f"Pairs: {orch.session.pairs_compiled}")
    except Exception as e:
        console.print(f"[red]Failed to load session: {e}[/red]")


def _show_sessions() -> None:
    """Display all available checkpoint sessions."""
    sessions = PipelineOrchestrator.list_sessions()
    if not sessions:
        console.print("[yellow]No saved sessions found.[/yellow]")
        return

    table = Table(title="Available Sessions")
    table.add_column("Session ID", style="cyan")
    for s in sessions:
        table.add_row(str(s))
    console.print(table)


def main():
    app()


if __name__ == "__main__":
    main()
