import os

def generate_kcl_import_bn711():
    # Stap 1: Pad naar specifieke revisie map "Rev I"
    base_folder = r"C:\Users\sietse.duister\OneDrive - De Voogt Naval Architects\00_specialists group\1_projects\2_ML system development\2_parasolids\BN711\Rev I"
    output_folder = r"C:\Users\sietse.duister\OneDrive - De Voogt Naval Architects\00_specialists group\1_projects\2_ML system development\1_git_ML system development\2_src"
    romp_naam = "BN711"

    # Stap 2: Zoek naar eerste .x_t bestand
    parasolid_file = None
    for file in os.listdir(base_folder):
        if file.lower().endswith(".x_t"):
            parasolid_file = os.path.join(base_folder, file)
            break

    if not parasolid_file:
        print("❌ Geen .x_t bestand gevonden in Rev I map.")
        return None

    # Stap 3: Output .design pad
    output_design = os.path.join(output_folder, f"{romp_naam}_met_hull.design")

    # Stap 4: Genereer KCL-inhoud
    kcl_content = f"""
ImportParasolid "{parasolid_file}" As "ImportedHalfHull"
SaveDesignAs "{output_design}"
"""

    # Stap 5: Schrijf het KCL-bestand weg
    kcl_path = os.path.join(output_folder, f"import_{romp_naam}.kcl")
    with open(kcl_path, "w") as f:
        f.write(kcl_content.strip())

    print(f"✅ Parasolid bestand gevonden: {parasolid_file}")
    print(f"✅ KCL-script opgeslagen naar: {kcl_path}")
    return kcl_path

generate_kcl_import_bn711()
