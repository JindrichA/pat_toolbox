# pat_toolbox/paths.py
from pathlib import Path
from . import config


def get_root_dir() -> Path:
    """
    Return the directory where this package is located.
    (Repository root – still useful for data, configs, etc.)
    """
    return Path(__file__).resolve().parent.parent


def get_output_root() -> Path:
    """
    Base output directory for ALL runs.
    """
    base = Path(getattr(config, "BASE_OUTPUT_DIR", None))
    if base is None:
        raise RuntimeError(
            "config.BASE_OUTPUT_DIR is not set. "
            "Please define a base output directory."
        )

    base.mkdir(parents=True, exist_ok=True)
    return base


def get_output_folder(subfolder: str | None = None) -> Path:
    """
    Create (if necessary) and return the path to the output folder.

    If subfolder is None, use config.OUTPUT_SUBFOLDER
    (which already includes RUN_ID + sleep policy).
    """
    if subfolder is None:
        subfolder = config.OUTPUT_SUBFOLDER

    out_folder = get_output_root() / subfolder
    out_folder.mkdir(parents=True, exist_ok=True)
    return out_folder
