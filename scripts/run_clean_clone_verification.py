"""Clean clone verification script.

Clones the repo into a temporary clean directory, checks out the specified commit,
verifies git status --porcelain is empty, and runs manifest verifier and tests.
"""

from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


def main():
    target_commit = sys.argv[1] if len(sys.argv) > 1 else "HEAD"
    repo_url = "https://github.com/hubdk17/Material_Science_Repo_Jarvis.git"

    temp_dir = Path(tempfile.mkdtemp(prefix="clean_clone_verify_"))
    clone_dir = temp_dir / "Material_Science_Repo_Jarvis"

    print(f"=== CLONING REPOSITORY TO CLEAN DIRECTORY: {clone_dir} ===")
    subprocess.run(["git", "clone", repo_url, str(clone_dir)], check=True)

    print(f"\n=== CHECKING OUT COMMIT: {target_commit} ===")
    subprocess.run(["git", "checkout", target_commit], cwd=str(clone_dir), check=True)

    print("\n=== RUNNING: git status --porcelain ===")
    res_status = subprocess.run(["git", "status", "--porcelain"], cwd=str(clone_dir), capture_output=True, text=True, check=True)
    status_out = res_status.stdout.strip()
    print(f"git status --porcelain output: '{status_out}'")
    if status_out:
        print(f"[FAILED] Working tree is not clean in clean clone:\n{status_out}")
        sys.exit(1)
    print("[PASSED] Working tree is completely clean (zero output).")

    print("\n=== RUNNING: python scripts/verify_manifest.py results/final_manifest.json ===")
    subprocess.run([sys.executable, "scripts/verify_manifest.py", "results/final_manifest.json"], cwd=str(clone_dir), check=True)

    print("\n=== RUNNING: python scripts/evaluate_saved_predictions.py ===")
    subprocess.run([sys.executable, "scripts/evaluate_saved_predictions.py"], cwd=str(clone_dir), check=True)

    print("\n=== RUNNING: python scripts/generate_forensic_report.py ===")
    subprocess.run([sys.executable, "scripts/generate_forensic_report.py"], cwd=str(clone_dir), check=True)

    print("\n=== RUNNING: pytest tests/ ===")
    subprocess.run([sys.executable, "-m", "pytest", "tests/"], cwd=str(clone_dir), check=True)

    print("\n========================================================")
    print("ALL CLEAN-CLONE VERIFICATION CHECKS PASSED SUCCESSFULLY!")
    print("========================================================")


if __name__ == "__main__":
    main()
