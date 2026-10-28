from pathlib import Path
import shutil

FOLDER_A = Path(r"processed_RSNA_output")
FOLDER_B = Path(r"./dataset/RSNA_jpeg")
FOLDER_C = Path(r"RSNA_output")

FOLDER_C.mkdir(parents=True, exist_ok=True)

for file_in_a in FOLDER_A.iterdir():
    if not file_in_a.is_file():
        continue

    source = FOLDER_B / file_in_a.name
    if source.is_file():
        shutil.copy2(source, FOLDER_C / file_in_a.name)
        print(f"Copied: {file_in_a.name}")
    else:
        print(f"Not found in folder B: {file_in_a.name}")