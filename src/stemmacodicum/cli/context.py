from __future__ import annotations

from dataclasses import dataclass

from rich.console import Console

from stemmacodicum.core.config import AppPaths


@dataclass(slots=True)
class CLIContext:
    paths: AppPaths
    console: Console
