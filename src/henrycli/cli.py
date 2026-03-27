"""HenryCLI - Multi-agent LLM orchestration for LM Studio."""

import asyncio
from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

from . import __version__
from .agents.router import RouterAgent, TaskType
from .context.manager import ContextManager
from .lmstudio import LMStudioClient
from .models.config import ModelConfig
from .models.pool import ModelPool

app = typer.Typer(
    name="henry",
    help="Multi-agent LLM orchestration for LM Studio",
    add_completion=False,
)
console = Console()


def get_client() -> LMStudioClient:
    """Create LM Studio client."""
    return LMStudioClient(base_url="http://localhost:1234")


@app.command()
def version() -> None:
    """Show version."""
    console.print(f"[bold blue]HenryCLI[/bold blue] v{__version__}")


@app.command()
def health() -> None:
    """Check LM Studio connection."""

    async def check() -> None:
        client = get_client()
        try:
            healthy = await client.health_check()
            if healthy:
                console.print("[green]✓[/green] LM Studio is available")

                models = await client.get_models()
                if models.data:
                    console.print(f"\n[bold]Loaded Models ({len(models.data)}):[/bold]")
                    for model in models.data:
                        console.print(f"  • {model.id}")
                else:
                    console.print("[yellow]⚠ No models loaded[/yellow]")
            else:
                console.print("[red]✗[/red] LM Studio is not available")
        except Exception as e:
            console.print(f"[red]✗[/red] Cannot connect to LM Studio: {e}")
        finally:
            await client.close()

    asyncio.run(check())


@app.command()
def analyze(
    task: str = typer.Argument(..., help="Task to analyze"),
) -> None:
    """Analyze a task without executing."""

    async def run() -> None:
        client = get_client()
        try:
            router = RouterAgent(client)

            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                progress.add_task("Analyzing task...", total=None)

                result = await router.execute(task)

            if result.success:
                console.print(Panel(result.output, title="Task Analysis"))
            else:
                console.print(f"[red]Error:[/red] {result.error}")
        finally:
            await client.close()

    asyncio.run(run())


@app.command()
def run(
    task: str = typer.Argument(..., help="Task to execute"),
    tier: str | None = typer.Option(
        None,
        "--tier",
        "-t",
        help="Force specific tier (T1-T4)",
    ),
    interactive: bool = typer.Option(
        False,
        "--interactive",
        "-i",
        help="Run in interactive mode",
    ),
) -> None:
    """Execute a task with automatic agent routing."""

    async def run_task() -> None:
        client = get_client()
        config = ModelConfig()
        model_pool = ModelPool(client, config)
        context_manager = ContextManager()

        try:
            # Refresh model status
            await model_pool.refresh_model_status()

            # Analyze task
            router = RouterAgent(client)
            console.print("[bold blue]Analyzing task...[/bold blue]")

            analysis = await router.analyze(task)

            console.print(
                f"  Type: [cyan]{analysis.task_type.value}[/cyan]\n"
                f"  Complexity: [cyan]{analysis.complexity.value}[/cyan]\n"
                f"  Recommended Tier: [cyan]{analysis.recommended_tier}[/cyan]\n"
                f"  Confidence: [cyan]{analysis.confidence:.0%}[/cyan]"
            )

            # Determine target tier
            target_tier = tier or analysis.recommended_tier

            # Get model for tier
            target_model = model_pool.get_model_for_tier(target_tier)
            if not target_model:
                console.print(f"[red]No model available for tier {target_tier}[/red]")
                return

            console.print(f"\n[bold blue]Using model:[/bold blue] {target_model}")

            # Check if model switch needed
            if model_pool.current_model != target_model:
                console.print("\n[yellow]Model switch required...[/yellow]")
                console.print(
                    "[dim]Please load the model in LM Studio, then press Enter[/dim]"
                )
                if interactive:
                    input()

                success, actual_model = await model_pool.switch_with_fallback(
                    target_model
                )

                if not success:
                    console.print("[red]Failed to switch model[/red]")
                    return

                console.print(f"[green]✓[/green] Switched to: {actual_model}")

            # Create context
            context_manager.create_context(
                agent_id=f"task-{hash(task) % 10000}",
                model=target_model,
                task=task,
            )

            # Execute based on task type
            console.print("\n[bold blue]Executing task...[/bold blue]")

            if analysis.subtasks:
                console.print(f"\n[dim]Subtasks: {len(analysis.subtasks)}[/dim]")

            # For now, use router to generate response
            # In full implementation, would use specialist agents
            router.state.conversation_history = []
            router._add_user_message(task)

            response = await router._call_model(temperature=0.7, max_tokens=2000)

            console.print("\n[bold green]Result:[/bold green]")
            console.print(Panel(response, title="Output"))

            # Update context
            context_manager.add_message("user", task)
            context_manager.add_message("assistant", response)
            await context_manager.save_state()

        except KeyboardInterrupt:
            console.print("\n[yellow]Task interrupted[/yellow]")
        except Exception as e:
            console.print(f"[red]Error:[/red] {e}")
        finally:
            await client.close()

    asyncio.run(run_task())


@app.command()
def models(
    list_models: bool = typer.Option(False, "--list", "-l", help="List all models"),
    stats: bool = typer.Option(False, "--stats", "-s", help="Show model statistics"),
) -> None:
    """Manage model configuration."""

    async def run() -> None:
        config = ModelConfig()
        client = get_client()
        model_pool = ModelPool(client, config)

        try:
            if list_models:
                console.print("[bold]Configured Models:[/bold]\n")
                for tier_id in ["T1", "T2", "T3", "T4"]:
                    tier = config.get_tier(tier_id)
                    if tier:
                        console.print(f"[cyan]{tier_id}[/cyan] ({tier.purpose}):")
                        for model in tier.models:
                            default = " [green](default)[/green]" if model == tier.default else ""
                            console.print(f"  • {model}{default}")
                        console.print()

            if stats:
                await model_pool.refresh_model_status()
                console.print("[bold]Model Pool Statistics:[/bold]\n")

                vram = model_pool.get_vram_usage()
                console.print(
                    f"VRAM: [cyan]{vram['used_gb']:.1f} / {vram['total_gb']:.1f} GB[/cyan] "
                    f"({vram['available_gb']:.1f} GB available)\n"
                )

                console.print("[bold]Models:[/bold]")
                for stat in model_pool.get_model_stats():
                    status = "[green]●[/green]" if stat["is_loaded"] else "[gray]○[/gray]"
                    resident = " [yellow](R)[/yellow]" if stat["is_resident"] else ""
                    console.print(
                        f"  {status} {stat['model_id']}{resident}\n"
                        f"     Tier: {stat['tier']}, VRAM: {stat['vram_gb']} GB, Loads: {stat['load_count']}"
                    )
        finally:
            await client.close()

    asyncio.run(run())


@app.command()
def context(
    show: bool = typer.Option(False, "--show", help="Show active contexts"),
    clear: bool = typer.Option(False, "--clear", help="Clear all contexts"),
) -> None:
    """Manage context storage."""

    def run() -> None:
        context_manager = ContextManager()

        if show:
            active_dir = context_manager.active_dir
            if not active_dir.exists():
                console.print("[dim]No active contexts[/dim]")
                return

            contexts = list(active_dir.iterdir())
            if not contexts:
                console.print("[dim]No active contexts[/dim]")
                return

            console.print(f"[bold]Active Contexts ({len(contexts)}):[/bold]\n")
            for ctx_dir in contexts:
                if ctx_dir.is_dir():
                    console.print(f"  • {ctx_dir.name}")

        if clear:
            console.print("[yellow]Clearing all contexts...[/yellow]")
            import shutil

            if context_manager.active_dir.exists():
                shutil.rmtree(context_manager.active_dir)
            context_manager.active_dir.mkdir(parents=True, exist_ok=True)
            console.print("[green]✓[/green] Contexts cleared")

    run()


@app.command()
def config(
    show: bool = typer.Option(False, "--show", help="Show current configuration"),
    edit: bool = typer.Option(False, "--edit", help="Open config in editor"),
) -> None:
    """Manage configuration."""

    def run() -> None:
        config = ModelConfig()

        if show:
            console.print("[bold]Current Configuration:[/bold]\n")
            console.print(f"Config path: [dim]{config.config_path}[/dim]\n")

            console.print("[bold]Hardware:[/bold]")
            console.print(f"  VRAM: {config.hardware.vram_gb} GB")
            console.print(f"  RAM: {config.hardware.ram_gb} GB")
            console.print(f"  GPU: {config.hardware.gpu}\n")

            console.print("[bold]Context:[/bold]")
            console.print(
                f"  Compression threshold: {config.context.compression_threshold:.0%}"
            )
            console.print(
                f"  Offload limit: {config.context.offload_token_limit} tokens"
            )
            console.print(
                f"  Keep recent: {config.context.keep_recent_messages} messages\n"
            )

            console.print("[bold]Performance:[/bold]")
            console.print(
                f"  Model switch timeout: {config.performance.model_switch_timeout_sec}s"
            )
            console.print(
                f"  Poll interval: {config.performance.model_switch_poll_interval_ms}ms"
            )
            console.print(
                f"  Inference timeout: {config.performance.inference_timeout_sec}s"
            )

        if edit:
            config_path = config.config_path
            if not config_path.exists():
                # Create default config
                config_path.parent.mkdir(parents=True, exist_ok=True)
                config.save_config()
                console.print(f"[green]✓[/green] Created config: {config_path}")

            console.print(f"Config file: {config_path}")
            console.print("[dim]Edit this file to customize settings[/dim]")

    run()


def main() -> None:
    """Entry point."""
    app()


if __name__ == "__main__":
    main()
