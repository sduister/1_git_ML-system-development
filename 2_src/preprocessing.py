import os
import re
import xlwings as xw
from collections import defaultdict
import pandas as pd
from pathlib import Path
from tabulate import tabulate

# Config
BUILD_NUMBER = "BN716"
PROJECT_ROOT = r"L:\\01_Projects"
TARGET_SHEET = "CFD Resistance"

ROW_START = 6
ROW_END = 17

COLUMNS = {
    "I": "V_kn",
    "J": "V_ms",
    "K": "Fn",
    "L": "Rp_kN",
    "M": "Rf_kN",
    "N": "Rtot_kN"
}

HYDROSTAT_CELLS = {
    "T_m": "C14",
    "lcg_m": "C15",
    "vcg_m": "C17",
    "disp_kg": "C18"
}

def extract_metadata_from_filename(filename):
    revision_match = re.search(r'Rev([A-Z])', filename)
    revision = revision_match.group(1) if revision_match else None
    return revision

def find_candidate_excels(build_folder, build_number):
    candidates = []
    for root, _, files in os.walk(build_folder):
        for file in files:
            if (
                file.startswith("TDS_")
                and build_number in file
                and "BH" in file
                and file.endswith(".xlsx")
            ):
                revision = extract_metadata_from_filename(file)
                if revision:
                    full_path = os.path.join(root, file)
                    candidates.append({
                        "path": full_path,
                        "filename": file,
                        "revision": revision
                    })
    return candidates

def evaluate_excel(filepath, app, build_number):
    try:
        wb = app.books.open(filepath)
        sht = wb.sheets[TARGET_SHEET]

        # Extract hydrostatic data
        hydrostat_data = {}
        for key, cell in HYDROSTAT_CELLS.items():
            hydrostat_data[key] = sht.range(cell).value

        # Extract speed-resistance data
        data_rows = []
        for row in range(ROW_START, ROW_END + 1):
            v_kn = sht.range(f"I{row}").value
            rtot = sht.range(f"N{row}").value
            if v_kn in [None, 0] or rtot in [None, 0]:
                continue

            raw_data = {
                "build_number": build_number,
                "source_file": filepath,
                "V_kn": v_kn,
                "V_ms": sht.range(f"J{row}").value,
                "Fn": sht.range(f"K{row}").value,
                "Rp_kN": sht.range(f"L{row}").value,
                "Rf_kN": sht.range(f"M{row}").value,
                "Rtot_kN": rtot,
                **hydrostat_data
            }

            # Rearranged column order
            row_data = {
                key: raw_data.get(key, None) for key in [
                    "build_number", "source_file",
                    "V_kn", "V_ms", "Fn", "Rp_kN", "Rf_kN", "Rtot_kN",
                    "T_m", "lcg_m", "vcg_m", "disp_kg"
                ]
            }

            data_rows.append(row_data)

        return data_rows

    except Exception as e:
        print(f"\n❌ Error reading {filepath}: {e}")
        return []
    finally:
        wb.close()

def group_and_select_best(candidates, app, build_number):
    revision_map = defaultdict(list)

    for entry in candidates:
        revision_map[entry["revision"]].append(entry)

    for revision in sorted(revision_map.keys(), reverse=True):
        files = revision_map[revision]
        valid_files = []

        for file_info in files:
            data = evaluate_excel(file_info["path"], app, build_number)
            if data:
                file_info["data"] = data
                valid_files.append(file_info)

        if len(valid_files) >= 2:
            print(f"✅ Using revision Rev{revision} with {len(valid_files)} valid CFD files.")
            return valid_files

    print("❌ No revision with ≥2 valid CFD draught files found.")
    return []

if __name__ == "__main__":
    build_dir = os.path.join(PROJECT_ROOT, BUILD_NUMBER)
    print(f"🔍 Searching in: {build_dir}")

    candidates = find_candidate_excels(build_dir, BUILD_NUMBER)
    print(f"📂 Found {len(candidates)} candidate BH Excel files.")

    app = xw.App(visible=False)
    app.display_alerts = False
    app.screen_updating = False

    try:
        selected_files = group_and_select_best(candidates, app, BUILD_NUMBER)

        all_data = []
        for file in selected_files:
            for row in file["data"]:
                all_data.append(row)

        if all_data:
            df = pd.DataFrame(all_data)

            # Save to Parquet (custom location)
            parquet_path = Path(r"C:\Users\sietse.duister\OneDrive - De Voogt Naval Architects\00_specialists group\1_projects\2_ML system development\1_git_ML system development\1_data\raw_generated.parquet")
            df.to_parquet(parquet_path, index=False)

            print(f"\n✅ Data saved to: {parquet_path}")

            # Pretty print
            print("\n📋 Preview of saved data:")
            print(tabulate(df.head(10), headers="keys", tablefmt="fancy_grid"))

        else:
            print("❌ No valid CFD data extracted.")
    finally:
        app.quit()