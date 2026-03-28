"""Terminal UI for HenryCLI using Textual."""

from pathlib import Path
from typing import Any

from textual import on
from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal, Vertical, ScrollableContainer
from textual.widgets import (
    Header,
    Footer,
    Static,
    Button,
    Input,
    DataTable,
    TabbedContent,
    TabPane,
    RichLog,
    Label,
    LoadingIndicator,
    ProgressBar,
)
from textual.binding import Binding
from rich.markdown import Markdown
from rich.panel import Panel
from rich.text import Text

from .lmstudio import LMStudioClient
from .agents.router import RouterAgent, TaskType
from .context.manager import ContextManager
from .models.config import ModelConfig
from .models.pool import ModelPool
from .auto_tier import AutoTier, AutoTierClassifier


class ModelStatus(Static):
    """Widget showing model status."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loaded_model = "None"
        self.vram_usage = "0.0 / 8.0 GB"

    def update_status(self, model: str, vram: str) -> None:
        """Update the status display."""
        self.loaded_model = model
        self.vram_usage = vram
        self.update(
            f"[bold cyan]Model:[/bold cyan] {model}\n"
            f"[bold cyan]VRAM:[/bold cyan] {vram}"
        )


class TaskPanel(Static):
    """Panel for entering and displaying tasks."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.current_task = ""
        self.task_history: list[dict[str, Any]] = []

    def add_task(self, task: str, result: str, status: str = "completed") -> None:
        """Add a task to history."""
        self.task_history.append({
            "task": task,
            "result": result,
            "status": status,
        })
        self.current_task = task


class HenryTUI(App):
    """HenryCLI Terminal User Interface."""

    CSS = """
    Screen {
        background: $surface;
    }

    #main-container {
        height: 100%;
        padding: 1;
    }

    #status-bar {
        height: auto;
        min-height: 3;
        background: $primary-background;
        border: solid $primary;
        margin-bottom: 1;
        padding: 1;
    }

    #input-area {
        height: auto;
        max-height: 4;
        margin-bottom: 1;
    }

    #input-area Input {
        width: 1fr;
    }

    #input-area Button {
        width: auto;
        min-width: 10;
        margin-left: 1;
    }

    #content-area {
        height: 1fr;
        border: solid $secondary;
        background: $surface;
        padding: 1;
    }

    #content-area RichLog {
        height: 100%;
    }

    #sidebar {
        width: 32;
        height: 100%;
        border-left: solid $primary;
        padding: 1;
    }

    .model-info {
        background: $primary-background;
        padding: 1;
        margin-bottom: 1;
    }

    .tier-badge {
        background: $accent;
        color: $text;
        padding: 0 1;
    }

    .tier-item {
        background: $surface;
        padding: 1;
        margin-bottom: 1;
    }

    Button {
        margin-bottom: 1;
    }

    #task-history DataTable {
        height: 1fr;
    }

    #quick-actions {
        height: auto;
        max-height: 10;
    }

    #tier-list {
        height: auto;
        max-height: 12;
    }
    """

    BINDINGS = [
        Binding("ctrl+q", "quit", "Quit", show=True),
        Binding("ctrl+n", "new_task", "New Task", show=True),
        Binding("ctrl+r", "refresh", "Refresh", show=True),
        Binding("ctrl+l", "toggle_log", "Toggle Log", show=True),
        Binding("f1", "help", "Help", show=True),
        Binding("escape", "clear_input", "Clear", show=True),
    ]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.client = LMStudioClient()
        self.config = ModelConfig()
        self.context_manager = ContextManager()
        self.model_pool: ModelPool | None = None
        self.router: RouterAgent | None = None
        self.task_history: list[dict[str, Any]] = []
        self.discovered_models: dict[str, list] = {"T1": [], "T2": [], "T3": [], "T4": []}
        self.initialized = False
        self.current_model = "None"
        self.vram_available = 0.0

    def compose(self) -> ComposeResult:
        """Compose the UI layout."""
        yield Header()

        with Horizontal(id="main-container"):
            # Main content area
            with Vertical():
                # Status bar
                yield ModelStatus(id="status-bar", classes="model-info")

                # Input area
                with Horizontal(id="input-area"):
                    yield Input(
                        placeholder="Enter task... (Ctrl+N=new, Ctrl+R=refresh, Ctrl+Q=quit)",
                        id="task-input",
                    )
                    yield Button("Run", id="run-btn", variant="primary")
                    yield Button("Analyze", id="analyze-btn", variant="default")

                # Content/Log area
                yield RichLog(id="content-area", highlight=True, markup=True, auto_scroll=True)

            # Sidebar
            with Vertical(id="sidebar"):
                yield Static("[bold]Model Tiers[/bold]", id="tier-header")

                with ScrollableContainer(id="tier-list"):
                    yield Static(self._render_tiers(), id="tier-display")

                yield Static("\n[bold]Quick Actions[/bold]", id="actions-header")

                with Vertical(id="quick-actions"):
                    yield Button("Discover", id="discover-btn", variant="default")
                    yield Button("Load T1", id="load-t1-btn", variant="default")
                    yield Button("Load T2", id="load-t2-btn", variant="default")
                    yield Button("Load T3", id="load-t3-btn", variant="default")
                    yield Button("Unload All", id="unload-btn", variant="error")

                yield Static("\n[bold]Task History[/bold]", id="history-header")
                yield DataTable(id="task-history")

        yield Footer()

    def _render_tiers(self) -> str:
        """Render tier information."""
        lines = []
        for tier_id, models in self.discovered_models.items():
            count = len(models)
            if count > 0:
                model_names = [m.model_key.split("/")[-1][:20] for m in models[:2]]
                lines.append(f"[cyan]{tier_id}[/cyan]: {count} models")
                for name in model_names:
                    lines.append(f"  • {name}")
                if count > 2:
                    lines.append(f"  ... and {count - 2} more")
        if not lines:
            lines.append("[dim]No models discovered[/dim]")
            lines.append("[dim]Click 'Discover' to scan[/dim]")
        return "\n".join(lines)

    def on_mount(self) -> None:
        """Called when app is mounted."""
        self.query_one("#task-input", Input).focus()

        # Setup task history table
        table = self.query_one("#task-history", DataTable)
        table.add_columns("Status", "Task", "Tier")
        table.cursor_type = "row"

        # Initialize model pool
        self._initialize_components()

        # Auto-discover models on startup
        self.call_later(self.discover_models)

    def _initialize_components(self) -> None:
        """Initialize HenryCLI components."""
        self.model_pool = ModelPool(self.client, self.config)
        self.router = RouterAgent(self.client)

    async def discover_models(self) -> None:
        """Discover and classify local models."""
        log = self.query_one("#content-area", RichLog)
        log.write("[dim]Discovering models...[/dim]\n")

        try:
            # Try API first
            local_models = await self.client.list_local_models()

            if not local_models:
                log.write("[yellow]No models found via API, trying CLI...[/yellow]\n")
                local_models = await self.client.list_downloaded_models()

            if not local_models:
                log.write("[red]No models found. Download models first.[/red]\n")
                log.write("[dim]Use: henry download <model-name>[/dim]\n")
                return

            classifier = AutoTierClassifier(
                hardware_vram_gb=self.config.hardware.vram_gb
            )
            analyses = classifier.classify_local_models(local_models)

            # Organize by tier
            self.discovered_models = {"T1": [], "T2": [], "T3": [], "T4": []}
            for analysis in analyses:
                tier = analysis.tier.value
                if tier in self.discovered_models:
                    self.discovered_models[tier].append(analysis)

            # Update display
            self.query_one("#tier-display", Static).update(self._render_tiers())

            # Update status bar with loaded models
            await self._update_status_bar()

            total = sum(len(m) for m in self.discovered_models.values())
            log.write(f"[green]✓[/green] Discovered {total} models\n")
            self.initialized = True

        except Exception as e:
            log.write(f"[red]Error discovering models: {e}[/red]\n")

    async def _update_status_bar(self) -> None:
        """Update the status bar with current model info."""
        try:
            loaded = await self.client.get_models()
            status_widget = self.query_one("#status-bar", ModelStatus)
            
            if loaded.data:
                current = loaded.data[0].id
                self.current_model = current
                status_widget.update_status(
                    current[:40] + "..." if len(current) > 40 else current,
                    f"~{self.config.hardware.vram_gb} GB available",
                )
            else:
                self.current_model = "None"
                status_widget.update_status("None", f"{self.config.hardware.vram_gb} GB available")
        except Exception:
            pass

    @on(Input.Submitted, "#task-input")
    def on_input_submitted(self, event: Input.Submitted) -> None:
        """Handle task input submission."""
        self.run_task(event.value)

    @on(Button.Pressed, "#run-btn")
    def on_run_pressed(self, event: Button.Pressed) -> None:
        """Handle run button press."""
        input_widget = self.query_one("#task-input", Input)
        self.run_task(input_widget.value)

    @on(Button.Pressed, "#analyze-btn")
    def on_analyze_pressed(self, event: Button.Pressed) -> None:
        """Handle analyze button press."""
        input_widget = self.query_one("#task-input", Input)
        self.analyze_task(input_widget.value)

    @on(Button.Pressed, "#discover-btn")
    def on_discover_pressed(self, event: Button.Pressed) -> None:
        """Handle discover button press."""
        self.discover_models()

    @on(Button.Pressed, "#load-t1-btn")
    async def on_load_t1_pressed(self, event: Button.Pressed) -> None:
        """Handle T1 model load button."""
        log = self.query_one("#content-area", RichLog)
        log.write("[dim]Loading T1 model...[/dim]\n")

        if self.discovered_models["T1"]:
            model = self.discovered_models["T1"][0]
            log.write(f"Loading: {model.model_key}\n")
            try:
                result = await self.client.load_model(
                    model_key=model.model_key,
                    gpu_layers="auto",
                )
                log.write(f"[green]✓[/green] Loaded: {model.model_key}\n")
                await self._update_status_bar()
            except Exception as e:
                log.write(f"[red]✗[/red] Failed: {e}[/red]\n")
        else:
            log.write("[yellow]No T1 models discovered[/yellow]\n")

    @on(Button.Pressed, "#load-t2-btn")
    async def on_load_t2_pressed(self, event: Button.Pressed) -> None:
        """Handle T2 model load button."""
        log = self.query_one("#content-area", RichLog)
        log.write("[dim]Loading T2 model...[/dim]\n")

        if self.discovered_models["T2"]:
            model = self.discovered_models["T2"][0]
            log.write(f"Loading: {model.model_key}\n")
            try:
                result = await self.client.load_model(
                    model_key=model.model_key,
                    gpu_layers="auto",
                )
                log.write(f"[green]✓[/green] Loaded: {model.model_key}\n")
                await self._update_status_bar()
            except Exception as e:
                log.write(f"[red]✗[/red] Failed: {e}[/red]\n")
        else:
            log.write("[yellow]No T2 models discovered[/yellow]\n")

    @on(Button.Pressed, "#load-t3-btn")
    async def on_load_t3_pressed(self, event: Button.Pressed) -> None:
        """Handle T3 model load button."""
        log = self.query_one("#content-area", RichLog)
        log.write("[dim]Loading T3 model...[/dim]\n")

        if self.discovered_models["T3"]:
            model = self.discovered_models["T3"][0]
            log.write(f"Loading: {model.model_key}\n")
            try:
                result = await self.client.load_model(
                    model_key=model.model_key,
                    gpu_layers="auto",
                )
                log.write(f"[green]✓[/green] Loaded: {model.model_key}\n")
                await self._update_status_bar()
            except Exception as e:
                log.write(f"[red]✗[/red] Failed: {e}[/red]\n")
        else:
            log.write("[yellow]No T3 models discovered[/yellow]\n")

    @on(Button.Pressed, "#unload-btn")
    async def on_unload_pressed(self, event: Button.Pressed) -> None:
        """Handle unload button press."""
        log = self.query_one("#content-area", RichLog)
        log.write("[dim]Unloading all models...[/dim]\n")

        try:
            unloaded = await self.client.unload_all_models()
            log.write(f"[green]✓[/green] Unloaded {len(unloaded)} models\n")

            self.query_one("#status-bar", ModelStatus).update_status(
                "None", "0.0 / 8.0 GB"
            )
        except Exception as e:
            log.write(f"[red]Error: {e}[/red]\n")

    def run_task(self, task: str) -> None:
        """Run a task."""
        if not task.strip():
            return

        log = self.query_one("#content-area", RichLog)
        log.write(f"\n[bold cyan]Task:[/bold cyan] {task}\n")
        log.write("[dim]Analyzing...[/dim]\n")

        # Clear input
        self.query_one("#task-input", Input).value = ""

        # Process task asynchronously
        self._process_task(task)

    async def _process_task(self, task: str) -> None:
        """Process task asynchronously."""
        log = self.query_one("#content-area", RichLog)

        try:
            if not self.router:
                log.write("[red]Router not initialized[/red]\n")
                return

            # Analyze task
            analysis = await self.router.analyze(task)

            log.write(
                f"[bold]Analysis:[/bold]\n"
                f"  Type: [cyan]{analysis.task_type.value}[/cyan]\n"
                f"  Complexity: [cyan]{analysis.complexity.value}[/cyan]\n"
                f"  Tier: [cyan]{analysis.recommended_tier}[/cyan]\n"
                f"  Confidence: [cyan]{analysis.confidence:.0%}[/cyan]\n\n"
            )

            # Add to task history
            table = self.query_one("#task-history", DataTable)
            table.add_row(
                "✓",
                task[:30] + "..." if len(task) > 30 else task,
                analysis.recommended_tier,
            )

            # Show subtasks if any
            if analysis.subtasks:
                log.write(f"[bold]Subtasks ({len(analysis.subtasks)}):[/bold]\n")
                for i, subtask in enumerate(analysis.subtasks, 1):
                    log.write(
                        f"  {i}. {subtask.get('description', 'N/A')} "
                        f"[dim][{subtask.get('type', '?')}, {subtask.get('complexity', '?')}][/dim]\n"
                    )

            log.write("\n[green]✓ Task analysis complete[/green]\n")

        except Exception as e:
            log.write(f"[red]Error: {e}[/red]\n")

    def analyze_task(self, task: str) -> None:
        """Analyze a task without executing."""
        if not task.strip():
            return

        log = self.query_one("#content-area", RichLog)
        log.write(f"\n[bold yellow]Analyze:[/bold yellow] {task}\n")
        self.run_task(task)

    def action_quit(self) -> None:
        """Quit the app."""
        self.exit()

    def action_new_task(self) -> None:
        """Focus on task input."""
        self.query_one("#task-input", Input).focus()
        self.query_one("#task-input", Input).value = ""

    def action_refresh(self) -> None:
        """Refresh model discovery."""
        self.discover_models()

    def action_toggle_log(self) -> None:
        """Toggle log visibility."""
        log = self.query_one("#content-area", RichLog)
        log.clear()

    def action_help(self) -> None:
        """Show help."""
        log = self.query_one("#content-area", RichLog)
        log.write(
            "\n[bold]Keyboard Shortcuts:[/bold]\n"
            "  [cyan]Ctrl+Q[/cyan] - Quit\n"
            "  [cyan]Ctrl+N[/cyan] - New task\n"
            "  [cyan]Ctrl+R[/cyan] - Refresh models\n"
            "  [cyan]Ctrl+L[/cyan] - Clear log\n"
            "  [cyan]F1[/cyan] - This help\n"
            "  [cyan]Escape[/cyan] - Clear input\n"
        )

    def action_clear_input(self) -> None:
        """Clear the input field."""
        self.query_one("#task-input", Input).value = ""

    async def on_unmount(self) -> None:
        """Cleanup on unmount."""
        await self.client.close()


def run_tui() -> None:
    """Run the HenryCLI TUI."""
    app = HenryTUI()
    app.run()


if __name__ == "__main__":
    run_tui()
