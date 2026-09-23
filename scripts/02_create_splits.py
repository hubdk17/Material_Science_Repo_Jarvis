"""Script to generate all frozen benchmark splits for Protocol A and Protocol B."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.splits import create_protocol_a_splits, create_protocol_b_splits
from src.training.logger import setup_logger


def main():
    logger = setup_logger("create_splits")
    logger.info("Generating Protocol A official JARVIS benchmark split...")
    split_a = create_protocol_a_splits()
    logger.info(
        f"Protocol A: Train={len(split_a['train']['jids'])}, "
        f"Val={len(split_a['val']['jids'])}, "
        f"Test={len(split_a['test']['jids'])}"
    )

    logger.info("Generating Protocol B 5-fold stratified chemical-system grouped splits...")
    split_b_files = create_protocol_b_splits(seed=42)
    logger.info(f"Protocol B generated {len(split_b_files)} split files: {list(split_b_files.keys())}")


if __name__ == "__main__":
    main()
