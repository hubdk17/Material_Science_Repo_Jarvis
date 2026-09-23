"""Source data manifest generator for data provenance tracking."""

import hashlib
import json
from pathlib import Path
from typing import Any, Dict


def compute_file_hashes(file_path: Path) -> tuple[str, str, int]:
    """Computes SHA-256, MD5, and size in bytes of a file."""
    sha256 = hashlib.sha256()
    md5 = hashlib.md5()
    size = 0
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            sha256.update(chunk)
            md5.update(chunk)
            size += len(chunk)
    return sha256.hexdigest(), md5.hexdigest(), size


def build_source_manifest(raw_dir: str = "data/raw") -> Dict[str, Any]:
    """Constructs the publication-grade source manifest documenting raw data provenance."""
    raw_path = Path(raw_dir)

    json_file = raw_path / "JARVIS-EFG4.json"
    csv_file = raw_path / "JARVIS-EFG4.csv"
    code_file = raw_path / "example.py"
    bench_file = raw_path / "dft_3d_max_efg.json.zip"

    manifest: Dict[str, Any] = {
        "dataset_metadata": {
            "title": "Density Functional Theory-based Electric Field Gradient Database",
            "doi": "10.6084/m9.figshare.12307700.v2",
            "source_url": "https://doi.org/10.6084/m9.figshare.12307700.v2",
            "figshare_article_id": 12307700,
            "version": 2,
            "publication_date": "2020-08-07T01:45:32Z",
            "license": "CC BY 4.0",
            "citation": "Choudhary, K., Ansari, J. N., Mazin, I. I. & Sauer, K. L. Density functional theory-based electric field gradient database. Scientific Data 7, 351 (2020). https://doi.org/10.1038/s41597-020-00707-8",
            "benchmark_source": {
                "repository": "https://github.com/atomgptlab/jarvis_leaderboard",
                "benchmark_name": "AI/SinglePropertyPrediction/dft_3d_max_efg"
            }
        },
        "files": {}
    }

    for f_path, label in [
        (json_file, "primary_structures_and_tensors"),
        (csv_file, "wyckoff_site_table"),
        (code_file, "reference_script"),
        (bench_file, "official_leaderboard_benchmark")
    ]:
        if f_path.exists():
            sha256, md5, size = compute_file_hashes(f_path)
            manifest["files"][f_path.name] = {
                "role": label,
                "size_bytes": size,
                "sha256": sha256,
                "md5": md5
            }

    # Inspect JSON schema & types
    if json_file.exists():
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        manifest["schema_analysis"] = {
            "total_records": len(data),
            "record_type": "list of dict",
            "fields": {
                "jid": {
                    "inferred_type": "str",
                    "description": "Unique JARVIS material identifier"
                },
                "atoms": {
                    "inferred_type": "dict",
                    "subfields": {
                        "lattice_mat": "List[List[float]] (3x3 lattice vectors in Angstrom)",
                        "coords": "List[List[float]] (Nx3 fractional positions)",
                        "elements": "List[str] (length-N IUPAC chemical symbols)",
                        "abc": "List[float] (lengths [a, b, c] in Angstrom)",
                        "angles": "List[float] (angles [alpha, beta, gamma] in degrees)",
                        "cartesian": "bool (False indicates fractional coordinates)",
                        "props": "List[str] (length-N property tags)"
                    }
                },
                "efg_raw_tensor": {
                    "inferred_type": "List[List[List[float]]]",
                    "shape": "Nx3x3",
                    "unit": "V/Angstrom^2",
                    "description": "Raw Cartesian EFG tensors computed by VASP"
                },
                "efg_diag_tensor": {
                    "inferred_type": "List[List[float]]",
                    "shape": "Nx4",
                    "unit": "[Vxx, Vyy, Vzz] in V/Angstrom^2, eta dimensionless",
                    "description": "Principal EFG components and asymmetry parameter"
                }
            }
        }

    out_file = raw_path / "source_manifest.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    return manifest


if __name__ == "__main__":
    m = build_source_manifest()
    print(f"Built source manifest with {len(m['files'])} files.")
