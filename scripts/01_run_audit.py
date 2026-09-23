"""Script to execute full dataset audit and generate summary tables."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.audit import run_data_audit
from src.data.manifest import build_source_manifest
from src.training.logger import setup_logger


def main():
    logger = setup_logger("run_audit")
    logger.info("Building source manifest...")
    manifest = build_source_manifest()
    logger.info(f"Manifest created with {len(manifest['files'])} files.")

    logger.info("Executing comprehensive dataset audit...")
    audit_res = run_data_audit()
    logger.info(f"Audit completed: {audit_res['num_structures']} structures, {audit_res['total_sites']} sites.")


if __name__ == "__main__":
    main()
