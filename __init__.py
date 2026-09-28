def main():
    """Import the GUI only when launching, so analysis tools remain headless."""
    from .ui import main as launch
    launch()

__all__ = ["main"]
