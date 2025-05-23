import os
import re
import json
import pandas as pd
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# --- Configuration ---

BASE_PROJECT_PATH = r"L:\01_Projects"
DESIGN_NUMBERS_PATH = os.path.join(BASE_PROJECT_PATH, "Design Numbers")
SAVE_PATH = Path(r"C:\Users\sietse.duister\OneDrive - De Voogt Naval Architects\00_specialists group\1_projects\2_ML system development\1_git_ML system development\1_data\raw_CFD.xlsx")
CACHE_PATH = r"C:\Users\sietse.duister\OneDrive - De Voogt Naval Architects\00_specialists group\1_projects\2_ML system development\1_git_ML system development\1_data\cache.json"

# Dynamically find all BN directories in BASE_PROJECT_PATH
def get_bn_directories(base_path):
    return [
        os.path.join(base_path, d)
        for d in os.listdir(base_path)
        if d.upper().startswith("BN") and os.path.isdir(os.path.join(base_path, d))
    ]

BN_DIRECTORIES = get_bn_directories(BASE_PROJECT_PATH)

# The code will scan all BN directories and the Design Numbers folder
INCLUDE_PATHS = BN_DIRECTORIES + [DESIGN_NUMBERS_PATH]

def find_output_csv_files():
    """
    Scans through INCLUDE_PATHS for output.csv files.
    Returns a list of tuples: (full_file_path, geometry_id, revision_hint)
    """
    output_files = []
    found_drafts = {}
    
    for base_path in INCLUDE_PATHS:
        print(f"\n🔍 Scanning: {base_path}")
        if not os.path.isdir(base_path):
            continue
        
        for root, dirs, files in os.walk(base_path):
            for file in files:
                if re.fullmatch(r'output_?\.csv', file, re.IGNORECASE):
                    full_path = os.path.join(root, file)
                    with open(full_path, 'r', encoding='utf-8') as f:
                        lines = f.readlines()
                    
                    # Get the draft value (assumes it is the second comma-separated value)
                    draft = None
                    for line in lines:
                        if line.lower().startswith("draft"):
                            draft = line.split(",")[1].strip()
                            break
                    if draft is None:
                        continue
                    
                    try:
                        draft_value = float(draft)
                        if draft_value < 1.0:
                            continue
                    except ValueError:
                        continue
                    
                    # Use the immediate parent folder name as the revision hint
                    revision_hint = os.path.basename(os.path.dirname(full_path))
                    
                    if revision_hint not in found_drafts:
                        found_drafts[revision_hint] = set()
                    
                    if draft not in found_drafts[revision_hint]:
                        found_drafts[revision_hint].add(draft)
                        
                        # Extract geometry id from the full_path by searching for BNxxxx or DNxxxx
                        geometry_id = None
                        bn_match = re.search(r'(BN\d+)', full_path, re.IGNORECASE)
                        dn_match = re.search(r'(DN\d+)', full_path, re.IGNORECASE)
                        if bn_match:
                            geometry_id = bn_match.group(1).upper()
                        elif dn_match:
                            geometry_id = dn_match.group(1).upper()
                        else:
                            # Fallback: use the immediate parent folder's name (though usually revision_hint)
                            geometry_id = os.path.basename(os.path.dirname(full_path))
                        
                        output_files.append((full_path, geometry_id, revision_hint))
                        print(f"✅ Found OUTPUT.csv for geometry {geometry_id} Revision: {revision_hint} with draft {draft_value}")
    
    return output_files

def is_bare_hull(first_line):
    """Simple check to see if the first line indicates a bare hull file."""
    return "BH" in first_line

def parse_output_csv(file_path, geometry_id, revision_hint):
    """
    Parses an output CSV file and returns a DataFrame with the hydrostatic and CFD data.
    Adds the geometry, revision, and source file columns.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f if line.strip()]
        
        if not lines or not is_bare_hull(lines[0]):
            return pd.DataFrame()
        
        # Instead of extracting revision from the CSV, use the revision hint directly.
        revision_label = revision_hint
        
        hydrostatic_data = {}
        for line in lines:
            if line.startswith("-"):
                break
            parts = line.split(",")
            if len(parts) >= 2:
                key = parts[0].strip().lower().replace(" ", "_")
                value = parts[1].strip()
                if key in ["length_waterline", "draft", "lcg", "vcg", "displacement"]:
                    try:
                        hydrostatic_data[key] = float(value)
                    except ValueError:
                        hydrostatic_data[key] = value
        
        table_start_idx = next(i for i, line in enumerate(lines) if line.startswith("Velocity"))
        raw_headers = lines[table_start_idx].split(",")
        data_lines = lines[table_start_idx + 2:]
        
        seen = {}
        headers = []
        for h in raw_headers:
            h = h.strip() or "Unnamed"
            if h in seen:
                seen[h] += 1
                h = f"{h}_{seen[h]}"
            else:
                seen[h] = 0
            headers.append(h)
        
        cfd_data = pd.DataFrame([line.split(",") for line in data_lines], columns=headers)
        cfd_data = cfd_data.apply(pd.to_numeric, errors="coerce")
        
        for key, val in hydrostatic_data.items():
            cfd_data[key] = val
        
        cfd_data["Geometry"] = geometry_id
        cfd_data["Rev"] = revision_label
        cfd_data["source_file"] = file_path
        
        column_order = [
            "Geometry", "Rev", "Velocity", "Rf", "Rp", "Rtot", "Heave", "Wake fraction", "CPU time",
            "draft", "length_waterline", "lcg", "vcg", "displacement", "source_file"
        ]
        cfd_data = cfd_data[[col for col in column_order if col in cfd_data.columns]]
        return cfd_data
    
    except Exception as e:
        print(f"❌ Error parsing {file_path}: {e}")
        return pd.DataFrame()

def parse_task(task_tuple):
    file_path, geometry_id, revision_hint = task_tuple
    return parse_output_csv(file_path, geometry_id, revision_hint)

def build_sort_key(geometry):
    match = re.search(r'\d+', geometry)
    return int(match.group()) if match else float('inf')

def revision_sort_key(rev_str):
    # Here we simply use the revision string as-is for sorting (if desired, adjust sorting logic)
    return rev_str

def clean_combined_df(df):
    """
    Cleans the combined DataFrame by applying multiple cleaning steps:
      1. Remove rows where either "Vs [kn]" or "Rtot [kN]" is NaN or zero.
      2. Remove rows whose "geometry" value does not match the BNxxxx or DNxxxx format.
      3. For DN rows, if the "Lwl [m]" value is already present in a BN row, remove the DN row.
    Returns the cleaned DataFrame and a dictionary with deletion info.
    """
    deletion_info = {}
    original_count = df.shape[0]
    
    # Step 1: Remove rows with NaN or zero values in "Vs [kn]" or "Rtot [kN]"
    df1 = df.dropna(subset=["Vs [kn]", "Rtot [kN]"])
    condition = (df1["Vs [kn]"] != 0) & (df1["Rtot [kN]"] != 0)
    df1 = df1[condition]
    deletion_info["Invalid Vs/Rtot"] = original_count - df1.shape[0]
    
    # Step 2: Keep only rows where the geometry matches BNxxxx or DNxxxx (case insensitive)
    df2 = df1[df1["geometry"].str.match(r'^(BN\d+|DN\d+)$', na=False)]
    deletion_info["Invalid Geometry"] = df1.shape[0] - df2.shape[0]
    
    # Step 3: For DN rows, remove them if their Lwl [m] (length waterline) is already present in a BN row.
    bn_rows = df2[df2["geometry"].str.startswith("BN")]
    dn_rows = df2[df2["geometry"].str.startswith("DN")]
    
    bn_lwl_values = set(bn_rows["Lwl [m]"].dropna().unique())
    before_dn = dn_rows.shape[0]
    dn_rows = dn_rows[~dn_rows["Lwl [m]"].isin(bn_lwl_values)]
    deletion_info["Duplicate DN Lwl"] = before_dn - dn_rows.shape[0]
    
    cleaned_df = pd.concat([bn_rows, dn_rows], ignore_index=True)
    deletion_info["Total Deleted"] = original_count - cleaned_df.shape[0]
    return cleaned_df, deletion_info

def clean_and_save_excel(df, original_save_path):
    """
    Cleans the combined DataFrame, saves it to a new Excel file named 'cleaned_CFD.xlsx' 
    (in the same directory as the original file), and returns the cleaned DataFrame and deletion info.
    """
    cleaned_df, deletion_info = clean_combined_df(df)
    cleaned_save_path = original_save_path.parent / "cleaned_CFD.xlsx"
    cleaned_df.to_excel(cleaned_save_path, index=False)
    return cleaned_df, deletion_info

if __name__ == "__main__":
    print("🚀 Starting scan for CFD OUTPUT files...\n")
    
    # Try loading cache
    all_tasks = []
    use_cache = False
    if os.path.exists(CACHE_PATH):
        try:
            with open(CACHE_PATH, 'r', encoding='utf-8') as f:
                cached_tasks = json.load(f)
            # Verify that cached_tasks is a list of dicts with the expected keys.
            if isinstance(cached_tasks, list) and all(isinstance(t, dict) for t in cached_tasks):
                all_tasks = [(t["file"], t["geometry"], t["revision"]) for t in cached_tasks]
                use_cache = True
                print(f"Loaded {len(all_tasks)} tasks from cache.")
            else:
                print("Cache format is invalid. Rescanning directories.")
        except Exception as e:
            print(f"Error loading cache: {e}. Rescanning directories.")
    
    if not use_cache:
        all_tasks = find_output_csv_files()
        # Save tasks to cache as a list of dictionaries
        try:
            with open(CACHE_PATH, 'w', encoding='utf-8') as f:
                json.dump([{"file": t[0], "geometry": t[1], "revision": t[2]} for t in all_tasks], f, indent=4)
        except Exception as e:
            print(f"Error saving cache: {e}")
    
    print(f"\n📁 Total OUTPUT.csv variants found: {len(all_tasks)}\n")
    
    all_builds_data = []
    with ThreadPoolExecutor() as executor:
        future_to_task = {executor.submit(parse_task, task): task for task in all_tasks}
        for future in tqdm(as_completed(future_to_task), total=len(future_to_task), desc="⏳ Parsing OUTPUT.csv files"):
            df = future.result()
            if not df.empty:
                all_builds_data.append(df)
    
    if all_builds_data:
        combined_df = pd.concat(all_builds_data, ignore_index=True)
        
        # Rename columns for consistency (to lowercase)
        COLUMN_RENAME_MAP = {
            "Geometry": "geometry",
            "Rev": "rev",
            "Velocity": "Vs [kn]",
            "Rf": "Rf [kN]",
            "Rp": "Rp [kN]",
            "Rtot": "Rtot [kN]",
            "Heave": "heave [m]",
            "Wake fraction": "wake [-]",
            "CPU time": "cpu_time [s]",
            "draft": "T [m]",
            "length_waterline": "Lwl [m]",
            "lcg": "LCG [m]",
            "vcg": "VCG [m]",
            "displacement": "disp [m3]",
            "source_file": "source_file"
        }
        combined_df.rename(columns=COLUMN_RENAME_MAP, inplace=True)
        
        combined_df["sort_order"] = combined_df["geometry"].apply(build_sort_key)
        combined_df["rev_order"] = combined_df["rev"].apply(revision_sort_key)
        combined_df.sort_values(by=["sort_order", "rev_order"], inplace=True)
        combined_df.drop(columns=["sort_order", "rev_order"], inplace=True)
        
        # --- Print Raw Data Preview and Summary ---
        print("\n📄 Raw DataFrame preview (first 5 rows):")
        print(combined_df.head())
        
        raw_total_builds = combined_df["geometry"].nunique() if "geometry" in combined_df.columns else 0
        raw_total_revisions = combined_df[["geometry", "rev"]].drop_duplicates().shape[0] if "geometry" in combined_df.columns and "rev" in combined_df.columns else 0
        raw_total_samples = len(combined_df)
        
        print(f"\n📊 Raw Data Summary: {raw_total_builds} builds | {raw_total_revisions} revisions | {raw_total_samples} rows")
        
        # Save raw combined data to Excel (print statement removed)
        combined_df.to_excel(SAVE_PATH, index=False)
        
        # --- Data Cleaning Section (Moved to the End) ---
        cleaned_df, deletion_info = clean_and_save_excel(combined_df, SAVE_PATH)
        
        print("\n📄 Cleaned DataFrame preview (first 5 rows):")
        print(cleaned_df.head())
        
        cleaned_total_builds = cleaned_df["geometry"].nunique() if "geometry" in cleaned_df.columns else 0
        cleaned_total_revisions = cleaned_df[["geometry", "rev"]].drop_duplicates().shape[0] if "geometry" in cleaned_df.columns and "rev" in cleaned_df.columns else 0
        cleaned_total_samples = len(cleaned_df)
        
        print(f"\n📊 Cleaned Data Summary: {cleaned_total_builds} builds | {cleaned_total_revisions} revisions | {cleaned_total_samples} rows")
        
        # --- Print a Compact Summary of Deleted Rows ---
        print("\n🗑️ Rows Deleted:")
        print(f" - Invalid Vs/Rtot: {deletion_info.get('Invalid Vs/Rtot', 0)} rows removed (NaN/zero in Vs [kn] or Rtot [kN]).")
        print(f" - Invalid Geometry: {deletion_info.get('Invalid Geometry', 0)} rows removed (geometry not matching BNxxxx or DNxxxx).")
        print(f" - Duplicate DN Lwl: {deletion_info.get('Duplicate DN Lwl', 0)} DN rows removed (Lwl [m] already present in a BN row).")
        print(f" - Total Deleted: {deletion_info.get('Total Deleted', 0)} rows removed in total.")
        
        # --- Print Unique Geometry Names and Revisions in Cleaned Data ---
        # unique_geometries = cleaned_df[['geometry', 'rev']].drop_duplicates()
        # print("\n📌 Geometries and Revisions included in cleaned data:")
        # for geom, rev in unique_geometries.values:
        #     print(f" - {geom} {rev}")
    else:
        print("❌ No usable CFD data found across selected builds and design numbers.")
