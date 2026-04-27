import os
from typing import Optional


def setup_wandb_for_sft(wandb_project: str = "", wandb_name: str = "") -> bool:
    """Initialize wandb for SFT and return whether wandb logging is enabled."""
    project = (wandb_project or "").strip()
    run_name: Optional[str] = (wandb_name or "").strip() or None

    if not project:
        os.environ.pop("WANDB_PROJECT", None)
        return False

    os.environ["WANDB_PROJECT"] = project
    os.environ.setdefault("WANDB_SILENT", "true")

    import wandb

    wandb.init(
        name=run_name,
        project=project,
        settings=wandb.Settings(silent=True),
    )
    return True
