"""HenryCLI - Multi-agent LLM orchestration for LM Studio."""

import asyncio
from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from . import __version__
from .agents.router import RouterAgent, TaskType
from .auto_tier import AutoTier, AutoTierClassifier
from .context.manager import ContextManager
from .lmstudio import LMStudioClient
from .models.config import ModelConfig
from .models.pool import ModelPool
from .plugins import PluginManager

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


@app.command()
def plugins(
    list_plugins: bool = typer.Option(False, "--list", "-l", help="List plugins"),
    enable: str | None = typer.Option(None, "--enable", "-e", help="Enable plugin"),
    disable: str | None = typer.Option(None, "--disable", "-d", help="Disable plugin"),
    configure_rag: bool = typer.Option(
        False, "--configure-rag", help="Configure BigRAG plugin"
    ),
) -> None:
    """Manage LM Studio plugins."""

    def run() -> None:
        plugin_manager = PluginManager()

        if list_plugins:
            console.print("[bold]Available Plugins:[/bold]\n")
            plugins = plugin_manager.list_plugins()
            for plugin in plugins:
                status = "[green]enabled[/green]" if plugin["enabled"] else "[red]disabled[/red]"
                console.print(
                    f"[cyan]{plugin['name']}[/cyan] ({plugin['type']}): {status}"
                )
                if plugin["parameters"]:
                    for key, value in plugin["parameters"].items():
                        console.print(f"    {key}: [dim]{value}[/dim]")
                console.print()

        if enable:
            if plugin_manager.enable_tool(enable):
                console.print(f"[green]✓[/green] Enabled plugin: {enable}")
            else:
                console.print(f"[red]✗[/red] Unknown plugin: {enable}")

        if disable:
            if plugin_manager.disable_tool(disable):
                console.print(f"[yellow]✓[/yellow] Disabled plugin: {disable}")
            else:
                console.print(f"[red]✗[/red] Unknown plugin: {disable}")

        if configure_rag:
            console.print("[bold]BigRAG Configuration[/bold]\n")
            docs_dir = typer.prompt("Documents directory", default=str(Path.home() / "Documents"))
            vector_dir = typer.prompt(
                "Vector store directory",
                default=str(Path.home() / ".henrycli" / "rag-db"),
            )

            if plugin_manager.configure_rag(docs_dir, vector_dir):
                console.print(f"[green]✓[/green] BigRAG configured")
                console.print(f"  Documents: {docs_dir}")
                console.print(f"  Vector DB: {vector_dir}")
            else:
                console.print("[red]✗[/red] Failed to configure BigRAG")

    run()


@app.command()
def load(
    model: str = typer.Argument(..., help="Model to load (e.g., TheBloke/phi-3-mini-4k-instruct-GGUF)"),
    gpu_layers: str = typer.Option("auto", "--gpu", "-g", help="GPU layers (auto, max, or 0.0-1.0)"),
) -> None:
    """Load a model in LM Studio."""

    async def run() -> None:
        client = get_client()
        try:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                progress.add_task("Loading model...", total=None)

                result = await client.load_model(
                    model_key=model,
                    gpu_layers=gpu_layers,
                )

            console.print(f"[green]✓[/green] Loaded: {model}")
            console.print(f"  Instance ID: {result.get('instance_id', 'N/A')}")
        except Exception as e:
            console.print(f"[red]✗[/red] Failed to load model: {e}")
        finally:
            await client.close()

    asyncio.run(run())


@app.command()
def unload(
    all_models: bool = typer.Option(False, "--all", "-a", help="Unload all models"),
    model_id: str | None = typer.Argument(None, help="Model instance ID to unload"),
) -> None:
    """Unload models from LM Studio."""

    async def run() -> None:
        client = get_client()
        try:
            if all_models:
                results = await client.unload_all_models()
                console.print(f"[green]✓[/green] Unloaded {len(results)} models")
            elif model_id:
                result = await client.unload_model(model_id)
                console.print(f"[green]✓[/green] Unloaded: {result.get('instance_id', model_id)}")
            else:
                # Show loaded models
                models = await client.get_models()
                if models.data:
                    console.print("[bold]Loaded Models:[/bold]")
                    for model in models.data:
                        console.print(f"  • {model.id}")
                else:
                    console.print("[dim]No models loaded[/dim]")
        except Exception as e:
            console.print(f"[red]✗[/red] Error: {e}")
        finally:
            await client.close()

    asyncio.run(run())


@app.command()
def discover(
    auto_configure: bool = typer.Option(
        False, "--auto-configure", "-c", help="Auto-configure tiers based on discovered models"
    ),
) -> None:
    """
    [EXPERIMENTAL] Discover and classify local models into tiers.
    
    Analyzes model names to estimate:
    - Parameter count
    - VRAM requirements  
    - Appropriate tier (T1-T4)
    """

    async def run() -> None:
        client = get_client()
        config = ModelConfig()

        try:
            local_models = await client.list_local_models()

            if not local_models:
                console.print("[yellow]⚠ No local models found[/yellow]")
                console.print("Download models using: lms get <model-name>")
                return

            classifier = AutoTierClassifier(
                hardware_vram_gb=config.hardware.vram_gb
            )
            analyses = classifier.classify_local_models(local_models)

            # Display results
            console.print("[bold]Discovered Models by Tier:[/bold]\n")

            for tier in [AutoTier.T1, AutoTier.T2, AutoTier.T3, AutoTier.T4]:
                tier_models = [a for a in analyses if a.tier == tier]
                if tier_models:
                    console.print(f"[cyan]{tier.value}[/cyan] ({tier.value} models):")
                    for model in tier_models:
                        confidence = (
                            "[green]high[/green]"
                            if model.confidence == "high"
                            else "[yellow]medium[/yellow]"
                            if model.confidence == "medium"
                            else "[red]low[/red]"
                        )
                        console.print(
                            f"  • {model.model_key}"
                        )
                        console.print(
                            f"    ~{model.estimated_params_b}B params, "
                            f"~{model.estimated_vram_q4}GB VRAM (Q4), "
                            f"confidence: {confidence}"
                        )
                    console.print()

            if auto_configure:
                console.print("\n[bold]Generating tier configuration...[/bold]")
                tier_config = classifier.generate_tier_config(local_models)
                console.print("[green]✓[/green] Configuration generated")
                console.print("[dim]Use --show with models command to view[/dim]")

        except Exception as e:
            console.print(f"[red]✗[/red] Error: {e}")
        finally:
            await client.close()

    asyncio.run(run())


@app.command()
def get(
    url: str = typer.Argument(..., help="URL to download (GitHub, arXiv, direct file)"),
    filename: str | None = typer.Option(None, "--name", "-n", help="Custom filename"),
    list_files: bool = typer.Option(False, "--list", "-l", help="List downloaded files"),
    delete: str | None = typer.Option(None, "--delete", "-d", help="Delete a downloaded file"),
) -> None:
    """
    Download files to the RAG documents directory.
    
    Supports:
    - GitHub files (auto-converts to raw URL)
    - arXiv papers (downloads HTML version)
    - Direct file URLs (PDF, TXT, MD, code files)
    
    Examples:
        henry get https://github.com/user/repo/blob/main/file.py
        henry get https://arxiv.org/abs/2301.12345
        henry get https://example.com/document.pdf
        henry get --list
    """

    async def run() -> None:
        from .downloader import DocumentDownloader

        downloader = DocumentDownloader()

        if list_files:
            files = downloader.list_downloaded()
            if not files:
                console.print("[dim]No downloaded files[/dim]")
                console.print(f"RAG directory: {downloader.get_rag_directory()}")
                return

            console.print(f"[bold]Downloaded Files ({len(files)}):[/bold]\n")
            for file in files:
                size_kb = file["size_bytes"] / 1024
                console.print(
                    f"  [cyan]{file['filename']}[/cyan] - [dim]{size_kb:.1f} KB[/dim]"
                )
            console.print(f"\n[dim]Location: {downloader.get_rag_directory()}[/dim]")
            return

        if delete:
            if downloader.delete(delete):
                console.print(f"[green]✓[/green] Deleted: {delete}")
            else:
                console.print(f"[red]✗[/red] File not found: {delete}")
            return

        # Download file
        console.print(f"[bold blue]Downloading:[/bold blue] {url}")

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            progress.add_task("Downloading...", total=None)

            result = await downloader.download(url, filename=filename)

        if result.get("success"):
            console.print(f"[green]✓[/green] Downloaded: {result['filename']}")
            console.print(f"  Size: {result['size_bytes'] / 1024:.1f} KB")
            console.print(f"  Path: {result['path']}")

            # Prompt to configure BigRAG if not already done
            console.print("\n[dim]Tip: Run 'henry plugins --configure-rag' to use with BigRAG[/dim]")
        elif result.get("skipped"):
            console.print(f"[yellow]⚠[/yellow] {result['message']}")
            console.print(f"  Path: {result['path']}")
        else:
            console.print(f"[red]✗[/red] {result.get('message', 'Download failed')}")

    asyncio.run(run())


def main() -> None:
    """Entry point."""
    app()


if __name__ == "__main__":
    main()
