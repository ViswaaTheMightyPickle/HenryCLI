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
        height: 3;
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
        width: 100%;
    }

    #content-area {
        height: 1fr;
        border: solid $secondary;
        background: $surface;
    }

    #content-area RichLog {
        height: 100%;
    }

    #sidebar {
        width: 30;
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

    Button {
        margin-bottom: 1;
    }

    #task-history DataTable {
        height: 1fr;
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
                        placeholder="Enter your task... (e.g., 'Write a hello world program')",
                        id="task-input",
                    )
                    yield Button("Run", id="run-btn", variant="primary")
                    yield Button("Analyze", id="analyze-btn", variant="default")

                # Content/Log area
                yield RichLog(id="content-area", highlight=True, markup=True)

            # Sidebar
            with Vertical(id="sidebar"):
                yield Static("[bold]Model Tiers[/bold]", id="tier-header")

                with ScrollableContainer(id="tier-list"):
                    yield Static(self._render_tiers(), id="tier-display")

                yield Static("\n[bold]Quick Actions[/bold]", id="actions-header")

                with Vertical(id="quick-actions"):
                    yield Button("Discover Models", id="discover-btn", variant="default")
                    yield Button("Load T1 Model", id="load-t1-btn", variant="default")
                    yield Button("Load T2 Model", id="load-t2-btn", variant="default")
                    yield Button("Load T3 Model", id="load-t3-btn", variant="default")
                    yield Button("Unload All", id="unload-btn", variant="error")

                yield Static("\n[bold]Task History[/bold]", id="history-header")
                yield DataTable(id="task-history")

        yield Footer()

    def _render_tiers(self) -> str:
        """Render tier information."""
        lines = []
        for tier_id, models in self.discovered_models.items():
            count = len(models)
            lines.append(f"[cyan]{tier_id}[/cyan]: {count} models")
        return "\n".join(lines) if lines else "[dim]No models discovered[/dim]"

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
                log.write("[yellow]No models found via API[/yellow]\n")
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

            # Update status bar
            loaded = await self.client.get_models()
            if loaded.data:
                current = loaded.data[0].id
                self.query_one("#status-bar", ModelStatus).update_status(
                    current[:40] + "..." if len(current) > 40 else current,
                    f"~{self.config.hardware.vram_gb} GB available",
                )

            log.write(f"[green]✓[/green] Discovered {len(local_models)} models\n")

        except Exception as e:
            log.write(f"[red]Error discovering models: {e}[/red]\n")

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
            log.write(f"Found: {model.model_key}\n")
            # Would call load_model here
        else:
            log.write("[yellow]No T1 models discovered[/yellow]\n")

    @on(Button.Pressed, "#load-t2-btn")
    async def on_load_t2_pressed(self, event: Button.Pressed) -> None:
        """Handle T2 model load button."""
        log = self.query_one("#content-area", RichLog)
        log.write("[dim]Loading T2 model...[/dim]\n")

        if self.discovered_models["T2"]:
            model = self.discovered_models["T2"][0]
            log.write(f"Found: {model.model_key}\n")
        else:
            log.write("[yellow]No T2 models discovered[/yellow]\n")

    @on(Button.Pressed, "#load-t3-btn")
    async def on_load_t3_pressed(self, event: Button.Pressed) -> None:
        """Handle T3 model load button."""
        log = self.query_one("#content-area", RichLog)
        log.write("[dim]Loading T3 model...[/dim]\n")

        if self.discovered_models["T3"]:
            model = self.discovered_models["T3"][0]
            log.write(f"Found: {model.model_key}\n")
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
