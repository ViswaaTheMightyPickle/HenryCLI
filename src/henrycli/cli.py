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
            # Initialize model manager
            from .model_manager import ModelManager
            model_mgr = ModelManager(client, config)
            
            # Initialize and get available models
            await model_mgr.initialize()
            await model_mgr.get_available_models()

            # Load T1 router for task analysis
            console.print("[bold blue]Analyzing task...[/bold blue]")
            router_model = await model_mgr.load_router()
            
            if not router_model:
                console.print("[red]Failed to load T1 router model[/red]")
                return
            
            console.print(f"[dim]Router: {router_model}[/dim]")

            # Analyze task
            router = RouterAgent(client)
            analysis = await router.analyze(task)

            console.print(
                f"  Type: [cyan]{analysis.task_type.value}[/cyan]\n"
                f"  Complexity: [cyan]{analysis.complexity.value}[/cyan]\n"
                f"  Recommended Tier: [cyan]{analysis.recommended_tier}[/cyan]\n"
                f"  Confidence: [cyan]{analysis.confidence:.0%}[/cyan]"
            )

            # Determine target tier
            target_tier = tier or analysis.recommended_tier
            
            # Upgrade tier for code tasks - small models (<7B) can't do ReAct reliably
            if analysis.task_type.value == "code" and target_tier == "T1":
                console.print("[dim]Upgrading tier for code task (T1 → T2)[/dim]")
                target_tier = "T2"

            # Allow user override with --tier flag
            if tier:
                target_tier = tier

            # Get model for tier
            target_model = model_pool.get_model_for_tier(target_tier)
            if not target_model:
                console.print(f"[red]No model available for tier {target_tier}[/red]")
                return

            # For code tasks, prefer 14B+ models (more reliable at ReAct)
            if analysis.task_type.value == "code" and target_tier == "T2":
                # Find a 14B+ model in T2 from config
                tier_config = config.get_tier("T2")
                if tier_config and tier_config.models:
                    for mdl in tier_config.models:
                        mdl_lower = mdl.lower()
                        if "14b" in mdl_lower or "ministral" in mdl_lower:
                            console.print("[dim]Using 14B+ model for reliable code generation[/dim]")
                            target_model = mdl
                            break

            console.print(f"\n[bold blue]Using model:[/bold blue] {target_model}")

            # Algorithmic model switching: unload all, load specialist
            if model_pool.current_model != target_model:
                console.print("\n[yellow]Switching models...[/yellow]")
                
                # Unload all models first
                console.print("[dim]Unloading all models...[/dim]")
                await model_mgr.unload_all()
                
                # Get context length for target model
                context_length = config.get_context_length_for_model(target_model)
                console.print(f"[dim]Context length: {context_length} tokens[/dim]")

                # Load specialist model
                console.print("[dim]Loading specialist model...[/dim]")
                success = await model_mgr.load_specialist(target_model)
                
                if not success:
                    console.print("[red]Failed to load specialist model[/red]")
                    return
                    
                console.print(f"[green]✓[/green] Ready: {target_model}")

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

            # Create specialist agent for task type
            from .agents import create_agent_for_type
            
            specialist = create_agent_for_type(
                task_type=analysis.task_type.value,
                client=client,
                model=target_model,
            )
            
            console.print(f"[dim]Using {specialist.agent_id} for {analysis.task_type.value} task[/dim]")
            
            # Execute with specialist agent
            result = await specialist.execute(task)

            if result.success:
                console.print("\n[bold green]Result:[/bold green]")
                console.print(Panel(result.output, title="Output"))

                if result.artifacts:
                    console.print(f"\n[yellow]Artifacts generated: {len(result.artifacts)}[/yellow]")

                # Update context
                context_manager.add_message("user", task)
                context_manager.add_message("assistant", result.output)
                await context_manager.save_state()

                # Reload T1 router for verification
                console.print("\n[bold blue]Verifying results...[/bold blue]")
                router_model = await model_mgr.reload_router()
                
                if router_model:
                    console.print(f"[dim]Router: {router_model}[/dim]")
                    
                    # Create verification prompt
                    verification_prompt = (
                        f"Review this completed task and verify the output is correct:\n\n"
                        f"Task: {task}\n\n"
                        f"Output: {result.output[:500]}..."  # Truncate for context
                        f"\n\nConfirm the task was completed successfully and suggest any follow-up actions."
                    )
                    
                    verifier = RouterAgent(client)
                    verification = await verifier.analyze(verification_prompt)
                    console.print(f"[dim]Verification: {verification.reasoning[:200]}...[/dim]")
                else:
                    console.print("[dim]Verification skipped (router not available)[/dim]")
            else:
                console.print(f"[red]Error:[/red] {result.error}")

        except KeyboardInterrupt:
            console.print("\n[yellow]Task interrupted[/yellow]")
        except Exception as e:
            console.print(f"[red]Error:[/red] {e}")
        finally:
            await client.close()

    asyncio.run(run_task())


@app.command()
def models(
    list_models: bool = typer.Option(False, "--list", "-l", help="List configured models"),
    local: bool = typer.Option(False, "--local", help="List local installed models"),
    stats: bool = typer.Option(False, "--stats", "-s", help="Show model statistics"),
) -> None:
    """Manage model configuration."""

    async def run() -> None:
        config = ModelConfig()
        client = get_client()
        model_pool = ModelPool(client, config)

        try:
            if local:
                # Show locally installed models from lms CLI
                local_models = await client.list_downloaded_models()
                if not local_models:
                    console.print("[dim]No local models found[/dim]")
                    return

                console.print("[bold]Local Models (from lms):[/bold]\n")
                llm_models = [m for m in local_models if m.get("type") == "llm"]
                embedding_models = [m for m in local_models if m.get("type") == "embedding"]

                if llm_models:
                    console.print(f"[cyan]LLM Models ({len(llm_models)}):[/cyan]")
                    for model in llm_models:
                        model_key = model.get("modelKey", model.get("name", "Unknown"))
                        params = model.get("paramsString", "?")
                        size_gb = model.get("sizeBytes", 0) / (1024**3)
                        arch = model.get("architecture", "?")
                        loaded = " [green]✓ LOADED[/green]" if model.get("identifier") else ""
                        console.print(
                            f"  • {model_key} - {params} ({arch}) - {size_gb:.2f} GB{loaded}"
                        )
                    console.print()

                if embedding_models:
                    console.print(f"[cyan]Embedding Models ({len(embedding_models)}):[/cyan]")
                    for model in embedding_models:
                        model_key = model.get("modelKey", model.get("name", "Unknown"))
                        size_gb = model.get("sizeBytes", 0) / (1024**3)
                        console.print(
                            f"  • {model_key} - {size_gb:.2f} GB"
                        )
                    console.print()

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
    use_cli: bool = typer.Option(
        False, "--use-cli", help="Use lms CLI instead of REST API"
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
        from henrycli.auto_tier import AutoTier, AutoTierClassifier

        client = get_client()
        config = ModelConfig()

        try:
            # Try to get models from API first
            local_models = await client.list_local_models()
            
            # If API returns empty, try CLI fallback
            if not local_models and use_cli:
                local_models = await client.list_downloaded_models()
            
            # If still no models, show helpful message
            if not local_models:
                console.print("[yellow]⚠ No local models found via API[/yellow]")
                console.print("\n[dim]Possible solutions:[/dim]")
                console.print("  1. Make sure LM Studio server is running (port 1234)")
                console.print("  2. Download models using: lms get <model-name>")
                console.print("  3. Try with --use-cli flag: henry discover --use-cli")
                console.print("\n[dim]Note: LM Studio REST API may not expose local models in all versions.[/dim]")
                return

            classifier = AutoTierClassifier(
                hardware_vram_gb=config.hardware.vram_gb
            )
            analyses = classifier.classify_local_models(local_models)

            # Display results
            console.print("[bold]Discovered Models by Tier:[/bold]\n")

            has_models = False
            for tier in [AutoTier.T1, AutoTier.T2, AutoTier.T3, AutoTier.T4]:
                tier_models = [a for a in analyses if a.tier == tier]
                if tier_models:
                    has_models = True
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

            if not has_models:
                console.print("[yellow]⚠ Models found but could not classify[/yellow]")
                console.print("[dim]Model names may not follow standard naming conventions[/dim]")

            if auto_configure and has_models:
                console.print("\n[bold]Generating tier configuration...[/bold]")
                tier_config = classifier.generate_tier_config(local_models)
                console.print("[green]✓[/green] Configuration generated")
                console.print("[dim]Edit ~/.henrycli/models/config.yaml to apply[/dim]")

        except Exception as e:
            console.print(f"[red]✗[/red] Error: {e}")
            console.print("[dim]Tip: Try 'henry discover --use-cli' as fallback[/dim]")
        finally:
            await client.close()

    asyncio.run(run())


@app.command()
def get(
    url: str | None = typer.Argument(None, help="URL to download (GitHub, arXiv, direct file)"),
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


@app.command()
def download(
    model: str = typer.Argument(..., help="Model to download (e.g., TheBloke/phi-3-mini-4k-instruct-GGUF)"),
    quantization: str | None = typer.Option(
        None, "--quant", "-q", help="Specific quantization (e.g., q4_k_m)"
    ),
    yes: bool = typer.Option(
        False, "--yes", "-y", help="Auto-confirm prompts"
    ),
) -> None:
    """
    Download a model from LM Studio Hub.

    Wraps 'lms get' command with HenryCLI interface.

    Examples:
        henry download TheBloke/phi-3-mini-4k-instruct-GGUF
        henry download llama-3.1-8b --quant q4_k_m
        henry download qwen2.5-7b -y
    """

    async def run() -> None:
        client = get_client()
        try:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                progress.add_task("Downloading model...", total=None)

                result = await client.download_model(
                    model_key=model,
                    quantization=quantization,
                    yes=yes,
                )

            if result.get("success"):
                console.print(f"[green]✓[/green] Downloaded: {model}")
                if result.get("output"):
                    console.print(f"[dim]{result['output']}[/dim]")
            else:
                console.print(f"[red]✗[/red] Failed to download: {model}")
                if result.get("error"):
                    console.print(f"[dim]Error: {result['error']}[/dim]")
        except Exception as e:
            console.print(f"[red]✗[/red] Error: {e}")
        finally:
            await client.close()

    asyncio.run(run())


@app.command()
def server(
    show_status: bool = typer.Option(False, "--status", "-s", help="Show server status"),
    start_server: bool = typer.Option(False, "--start", help="Start server"),
    stop_server: bool = typer.Option(False, "--stop", help="Stop server"),
) -> None:
    """
    Manage LM Studio server.

    Wraps 'lms server' commands.

    Examples:
        henry server --status
        henry server --start
        henry server --stop
    """
    client = get_client()

    # Default to status if no option provided
    is_status = not (show_status or start_server or stop_server)

    async def run() -> None:
        try:
            if start_server:
                result = await client.server_start()
                if result.get("success"):
                    console.print("[green]✓[/green] Server started")
                else:
                    console.print(f"[red]✗[/red] Failed to start: {result.get('error', 'Unknown error')}")

            elif stop_server:
                result = await client.server_stop()
                if result.get("success"):
                    console.print("[green]✓[/green] Server stopped")
                else:
                    console.print(f"[red]✗[/red] Failed to stop: {result.get('error', 'Unknown error')}")

            else:  # status
                result = await client.server_status()
                if result.get("running"):
                    console.print("[green]●[/green] Server is running")
                else:
                    console.print("[red]○[/red] Server is not running")
                if result.get("output") and result["output"] != "Server is running":
                    console.print(f"[dim]{result['output']}[/dim]")
        except Exception as e:
            console.print(f"[red]✗[/red] Error: {e}")
        finally:
            await client.close()

    if start_server:
        asyncio.run(run())
    elif stop_server:
        asyncio.run(run())
    else:  # status (default)
        asyncio.run(run())


@app.command()
def import_model(
    file_path: str = typer.Argument(..., help="Path to model file to import"),
    user_repo: str | None = typer.Option(
        None, "--user-repo", help="User/repo format (e.g., TheBloke/phi-3-mini)"
    ),
    copy: bool = typer.Option(
        False, "--copy", "-c", help="Copy file instead of moving"
    ),
    yes: bool = typer.Option(
        False, "--yes", "-y", help="Auto-confirm prompts"
    ),
) -> None:
    """
    Import a model file into LM Studio.

    Wraps 'lms import' command.

    Examples:
        henry import ~/Downloads/model.gguf
        henry import model.gguf --user-repo TheBloke/phi-3-mini
        henry import model.gguf --copy
    """

    async def run() -> None:
        client = get_client()
        try:
            result = await client.import_model(
                file_path=file_path,
                user_repo=user_repo,
                copy=copy,
                yes=yes,
            )

            if result.get("success"):
                console.print(f"[green]✓[/green] Imported: {file_path}")
                if result.get("output"):
                    console.print(f"[dim]{result['output']}[/dim]")
            else:
                console.print(f"[red]✗[/red] Failed to import: {result.get('error', 'Unknown error')}")
        except Exception as e:
            console.print(f"[red]✗[/red] Error: {e}")
        finally:
            await client.close()

    asyncio.run(run())


@app.command()
def init(
    auto_load: bool = typer.Option(
        True, "--load/--no-load", help="Auto-load the T1 routing model after init"
    ),
    use_cli: bool = typer.Option(
        False, "--use-cli", help="Use lms CLI instead of REST API for discovery"
    ),
) -> None:
    """
    Initialize HenryCLI by discovering models and auto-configuring tiers.

    This command:
    1. Discovers all local models
    2. Classifies them into tiers (T1-T4) based on parameter count
    3. Generates and saves tier configuration
    4. Optionally loads the T1 routing model

    Examples:
        henry init
        henry init --no-load
        henry init --use-cli
    """

    async def run() -> None:
        from .auto_tier import AutoTier, AutoTierClassifier

        client = get_client()
        config = ModelConfig()

        try:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                # Step 1: Discover models
                task = progress.add_task("[cyan]Discovering models...", total=None)
                
                # Try API first
                local_models = await client.list_local_models()
                
                # If API returns empty, try CLI fallback
                if not local_models and use_cli:
                    local_models = await client.list_downloaded_models()
                
                # If still no models, show helpful message
                if not local_models:
                    console.print("[yellow]⚠ No local models found[/yellow]")
                    console.print("\n[dim]Possible solutions:[/dim]")
                    console.print("  1. Make sure LM Studio server is running (port 1234)")
                    console.print("  2. Download models using: henry download <model-name>")
                    console.print("  3. Try with --use-cli flag: henry init --use-cli")
                    console.print("\n[dim]Note: LM Studio REST API may not expose local models in all versions.[/dim]")
                    return

                progress.update(task, description="[green]✓[/green] Found models")

            # Step 2: Classify models
            console.print("\n[bold blue]Classifying models...[/bold blue]")
            classifier = AutoTierClassifier(
                hardware_vram_gb=config.hardware.vram_gb
            )
            analyses = classifier.classify_local_models(local_models)

            # Display classification results
            console.print("[bold]Discovered Models by Tier:[/bold]\n")
            for tier in [AutoTier.T1, AutoTier.T2, AutoTier.T3, AutoTier.T4]:
                tier_models = [a for a in analyses if a.tier == tier]
                if tier_models:
                    console.print(f"[cyan]{tier.value}[/cyan] ({tier.value} models):")
                    for model in tier_models:
                        console.print(
                            f"  • {model.model_key} (~{model.estimated_params_b}B params, "
                            f"~{model.estimated_vram_q4}GB VRAM)"
                        )
                    console.print()

            if not any(a for a in analyses):
                console.print("[yellow]⚠ Models found but could not classify[/yellow]")
                return

            # Step 3: Generate and save configuration
            console.print("[bold blue]Generating configuration...[/bold blue]")
            tier_config = classifier.generate_tier_config(local_models)

            # Update config with discovered models
            for tier_id, models in tier_config.items():
                tier = config.get_tier(tier_id)
                if tier and models:
                    # Update models list
                    tier.models = [m["model_key"] for m in models]
                    # Set default to first (largest that fits)
                    tier.default = models[0]["model_key"]
                    # Update VRAM estimate
                    tier.vram_gb = models[0]["vram_q4"]
                    # Mark T1 as resident (routing model)
                    if tier_id == "T1":
                        tier.resident = True

            # Save configuration
            config.save_config()
            console.print(f"[green]✓[/green] Configuration saved to: {config.config_path}")

            # Step 4: Load T1 routing model if requested
            if auto_load:
                t1_models = [a for a in analyses if a.tier == AutoTier.T1]
                if t1_models:
                    console.print("\n[bold blue]Loading T1 routing model...[/bold blue]")
                    routing_model = t1_models[0].model_key

                    try:
                        # Get appropriate context length for model
                        context_length = config.get_context_length_for_model(routing_model)

                        # Unload all models first (including all instances)
                        console.print("[dim]Unloading all models...[/dim]")
                        await client.unload_all_models()

                        result = await client.load_model(
                            model_key=routing_model,
                            gpu_layers="auto",
                            context_length=context_length,
                        )
                        console.print(f"[green]✓[/green] Loaded: {routing_model}")
                        console.print(f"  Context length: {context_length} tokens")
                        console.print(f"  Instance ID: {result.get('instance_id', 'N/A')}")
                        console.print("\n[bold green]HenryCLI is ready to use![/bold green]")
                        console.print("[dim]Run 'henry run <task>' to execute tasks[/dim]")
                    except Exception as e:
                        console.print(f"[yellow]⚠ Model load failed: {e}[/yellow]")
                        console.print("[dim]You can manually load the model in LM Studio[/dim]")
                else:
                    console.print("\n[yellow]⚠ No T1 models found for routing[/yellow]")
                    console.print("[dim]Consider downloading a small model for routing tasks[/dim]")

        except Exception as e:
            console.print(f"[red]✗[/red] Error: {e}")
            import traceback
            console.print(f"[dim]{traceback.format_exc()}[/dim]")
        finally:
            await client.close()

    asyncio.run(run())


@app.command()
def tui() -> None:
    """Launch the HenryCLI Terminal User Interface."""
    from .tui import run_tui
    run_tui()


def main() -> None:
    """Entry point."""
    app()


if __name__ == "__main__":
    main()
