"""Round-trip consistency test asserting that every table in the report matches its source CSV."""

from pathlib import Path
import re
import numpy as np
import pandas as pd
import pytest

from scripts.generate_forensic_report import generate_forensic_report


def parse_markdown_tables_with_sources(md_text: str):
    """Parses markdown tables and their associated source CSV paths from footers."""
    # Split text into lines
    lines = md_text.splitlines()
    tables = []
    
    current_table_lines = []
    in_table = False
    
    for i, line in enumerate(lines):
        striped = line.strip()
        if striped.startswith("|") and striped.endswith("|"):
            in_table = True
            current_table_lines.append(striped)
        else:
            if in_table:
                # Table ended; look ahead for source footer
                source_csv = None
                for lookahead in lines[i:i + 5]:
                    match = re.search(r"Source:\s*\[`([^`]+)`\]\(([^)]+)\)", lookahead)
                    if match:
                        source_csv = match.group(2)
                        break
                
                if current_table_lines and source_csv:
                    tables.append((current_table_lines, source_csv))
                current_table_lines = []
                in_table = False
                
    if in_table and current_table_lines:
        tables.append((current_table_lines, None))
        
    return tables


def parse_markdown_table_to_df(table_lines):
    """Converts raw markdown table lines to pandas DataFrame."""
    # First line is header
    header = [c.strip().replace("&#124;", "|") for c in table_lines[0].split("|")[1:-1]]
    # Skip separator line (line 1)
    data = []
    for line in table_lines[2:]:
        row = [c.strip().replace("&#124;", "|") for c in line.split("|")[1:-1]]
        data.append(row)
    return pd.DataFrame(data, columns=header)


def test_report_matches_csv_round_trip():
    """Generates the report, parses every table, and asserts consistency with source CSV."""
    report_text = generate_forensic_report()
    tables = parse_markdown_tables_with_sources(report_text)
    
    assert len(tables) >= 10, f"Expected at least 10 audited tables in report, got {len(tables)}"
    
    for table_lines, source_rel_path in tables:
        assert source_rel_path is not None, "Every table must have an associated source CSV footer"
        source_path = Path(source_rel_path)
        assert source_path.exists(), f"Source CSV {source_path} does not exist"
        
        df_md = parse_markdown_table_to_df(table_lines)
        df_csv = pd.read_csv(source_path)
        
        # Verify row and column counts
        assert len(df_md) == len(df_csv), (
            f"Table {source_path.name} row count mismatch: MD has {len(df_md)}, CSV has {len(df_csv)}"
        )
        assert len(df_md.columns) == len(df_csv.columns), (
            f"Table {source_path.name} column count mismatch: MD has {len(df_md.columns)}, CSV has {len(df_csv.columns)}"
        )
        
        # Verify numeric and string values
        for col_idx, col in enumerate(df_csv.columns):
            md_col = df_md.columns[col_idx]
            csv_vals = df_csv[col].values
            md_vals = df_md[md_col].values
            
            for row_idx, (c_val, m_val) in enumerate(zip(csv_vals, md_vals)):
                # Check NaNs
                if pd.isna(c_val) or str(c_val) in ["nan", "NaN", "N/A"]:
                    assert m_val in ["nan", "NaN", "N/A", ""], (
                        f"Row {row_idx} col {col} expected NaN, got '{m_val}' in {source_path.name}"
                    )
                    continue
                
                # Check floats
                try:
                    c_float = float(c_val)
                    m_float = float(m_val)
                    # Compare within rounding tolerance
                    if abs(c_float) > 1e-4:
                        rel_err = abs(c_float - m_float) / abs(c_float)
                        assert rel_err < 1e-2, (
                            f"Mismatch in {source_path.name} at row {row_idx} col '{col}': "
                            f"CSV={c_float}, MD={m_float}"
                        )
                    else:
                        assert abs(c_float - m_float) < 1e-4, (
                            f"Mismatch in {source_path.name} at row {row_idx} col '{col}': "
                            f"CSV={c_float}, MD={m_float}"
                        )
                except ValueError:
                    # Compare string equality
                    assert str(c_val).strip() == str(m_val).strip(), (
                        f"String mismatch in {source_path.name} at row {row_idx} col '{col}': "
                        f"CSV='{c_val}', MD='{m_val}'"
                    )
