# Pilih folder FGD6 yang telah diekstrak dan terpasang sebagai Kaggle Input.
from collections import Counter
from pathlib import Path, PurePosixPath
import csv
import hashlib
import json
import shutil

fgd6_input = Path("/kaggle/input/datasets/ouvvee/fgd6-handoff-dan-checkpoint/FGD6")
fgd6_staged = Path("/kaggle/working/FGD6_staged")
handoff_rel = Path("handoff/20261028T104133_556236Z")
run_dense_rel = Path("runs/DenseNet121/20261028T104136_712764Z")
run_lwdn_rel = Path("runs/LWDN/20261028T151525_228962Z")
expected_manifests = {
    handoff_rel: "76bce00bfa7d2d45cdfb16f412cc3d14cc233004a0bd327c6dd732cc796a9150",
    run_dense_rel: "d9d3a5488a5c019529ce0024acb3ec02381deab50251e9027607d277898bd348",
    run_lwdn_rel: "060c95849972297c8075efd4b1fd5756f924261970fac3bdc3bec71e365a987b",
}

# Memeriksa isi input dan salinannya terhadap manifest arsip Drive yang diperiksa.
for location in (fgd6_input, fgd6_staged):
    if location == fgd6_staged and not fgd6_staged.exists():
        fgd6_staged.mkdir(parents=True, exist_ok=False)
        for relative in expected_manifests:
            destination = fgd6_staged / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(fgd6_input / relative, destination)
    for relative, expected in expected_manifests.items():
        root = location / relative
        if not root.is_dir() or root.is_symlink():
            raise FileNotFoundError(f"Folder artefak tidak tersedia: {root}")
        entries = list(root.rglob("*"))
        if any(path.is_symlink() for path in entries):
            raise ValueError(f"Artefak berisi symlink: {root}")
        digest = hashlib.sha256()
        for path in sorted((p for p in entries if p.is_file()), key=lambda p: p.relative_to(root).as_posix()):
            digest.update(path.relative_to(root).as_posix().encode("utf-8") + b"\0")
            with path.open("rb") as file:
                digest.update(hashlib.file_digest(file, "sha256").digest())
        if digest.hexdigest() != expected:
            raise ValueError(f"Isi artefak berbeda dari arsip Drive: {root}")
    print("Manifest cocok:", location)

# Memetakan path citra ke salinan Kaggle tanpa mengubah split atau label.
handoff_root = fgd6_staged / handoff_rel
metadata = json.loads((handoff_root / "metadata.json").read_text(encoding="utf-8"))
with (handoff_root / "df_final.csv").open(newline="", encoding="utf-8") as file:
    reader = csv.DictReader(file)
    columns = reader.fieldnames
    rows = list(reader)
if metadata["counts"] != {"train": 4000, "val": 342, "test": 343} or len(metadata["classes"]) != 10:
    raise ValueError("Metadata handoff tidak cocok dengan eksperimen yang tersimpan.")
if Counter(row["split"] for row in rows) != metadata["counts"]:
    raise ValueError("Jumlah split dalam tabel tidak cocok dengan metadata.")

image_root = handoff_root / "images"
mapped = set()
for row in rows:
    relative = PurePosixPath(row["prep_path"]).relative_to("/content/work/dataset_224")
    if len(relative.parts) != 3 or ".." in relative.parts or relative.parts[:2] != (row["split"], row["label"]):
        raise ValueError(f"Path atau label citra tidak cocok: {row['prep_path']}")
    path = image_root.joinpath(*relative.parts)
    if not path.is_file() or relative in mapped:
        raise ValueError(f"Citra tidak tersedia atau path ganda: {path}")
    mapped.add(relative)
    row["prep_path"] = str(path)
actual = {PurePosixPath(path.relative_to(image_root).as_posix()) for path in image_root.rglob("*.jpg")}
if mapped != actual:
    raise ValueError("File gambar dan tabel handoff tidak cocok.")

mapped_table = fgd6_staged / "df_final_kaggle.csv"
if mapped_table.exists():
    with mapped_table.open(newline="", encoding="utf-8") as file:
        if list(csv.DictReader(file)) != rows:
            raise ValueError(f"Tabel yang sudah ada berbeda dan tidak ditimpa: {mapped_table}")
else:
    with mapped_table.open("x", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
print("Handoff siap:", handoff_root)
print("Tabel dengan path Kaggle:", mapped_table)
print("Gambar:", dict(metadata["counts"]), "| kelas:", len(metadata["classes"]))
print("Run tersalin; pemulihan model belum dijalankan.")
