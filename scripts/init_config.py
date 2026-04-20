"""
Config Initialization Helper
=============================
Auto-generates missing config files from .example templates on first run.
Safely skips if files already exist (preserves user customizations).
"""

import shutil
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def initialize_configs(project_root: Path) -> None:
    """
    Auto-copy .example files to their runtime names if they don't exist.

    This allows first-time users to run the bot without manually copying
    example configs. Existing user configs are preserved.

    Args:
        project_root: Path to repository root (where config.yaml lives)
    """
    example_configs = {
        "notifications.yaml.example": "notifications.yaml",
        "news_blackouts.yaml.example": "news_blackouts.yaml",
    }

    for src_name, dst_name in example_configs.items():
        src = project_root / src_name
        dst = project_root / dst_name

        # Skip if user already has a custom config
        if dst.exists():
            logger.debug(f"Skipping {dst_name} (already exists)")
            continue

        # Copy from example if it exists
        if src.exists():
            try:
                shutil.copy(src, dst)
                logger.info(f"✓ Created {dst_name} from {src_name}")
            except Exception as e:
                logger.warning(f"Failed to copy {src_name}: {e}")
        else:
            logger.debug(f"Skipping {src_name} (not found; optional)")


if __name__ == "__main__":
    # Allow direct execution for testing
    logging.basicConfig(level=logging.INFO)
    project_root = Path(__file__).resolve().parent.parent
    initialize_configs(project_root)
    print("✓ Config initialization complete")
