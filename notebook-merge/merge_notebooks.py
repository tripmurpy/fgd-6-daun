"""Assemble source notebooks and validate their structure; never execute cells."""
import ast
import copy
import hashlib
import json
import textwrap
import pprint
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parent
COLAB = nbformat.read(ROOT / "sources/colab-20261031.ipynb", as_version=4)
KAGGLE = nbformat.read(ROOT / "sources/kaggle-saved-364146947.ipynb", as_version=4)
DRAFT = nbformat.read(ROOT / "sources/kaggle-draft-20261031.ipynb", as_version=4)
RUN_ID = "20261031T071608_511672Z"
ARTIFACTS = ROOT / "artifacts/runs" / RUN_ID
cells = []
mapping = []


def markdown(source):
    cells.append(nbformat.v4.new_markdown_cell(source))


def code(source):
    cells.append(nbformat.v4.new_code_cell(source.strip() + "\n"))


def imported(notebook, index, origin, guard=None, replacement=None):
    cell = copy.deepcopy(notebook.cells[index])
    original = cell.source
    cell.metadata = {"fgd_source": {"notebook": origin, "cell_index": index,
                                   "source_sha256": hashlib.sha256(original.encode()).hexdigest()}}
    cell.id = f"{origin}-{index:03}"
    if replacement is not None:
        cell.source = replacement
    if cell.cell_type == "code":
        cell.execution_count = None
        for output in cell.outputs:
            if "execution_count" in output:
                output.execution_count = None
        if guard:
            cell.source = f"if {guard}:\n" + textwrap.indent(cell.source, "    ")
    mapping.append({"destination_cell": len(cells), "source": origin, "source_cell": index,
                    "guard": guard, "adapted": replacement is not None})
    cells.append(cell)


imported(COLAB, 0, "colab")
markdown("""### Status penggabungan — belum dijalankan

Phase 1–3 disalin lengkap dari [Colab FGD-6](https://colab.research.google.com/drive/1CKksYUrcdfcZtOxGZq2EDlbh_3Jc2DOL), termasuk kode dan output historis. Modeling dan Evaluation kedua model berasal dari [Kaggle Version 4](https://www.kaggle.com/code/ouvvee/notebookc755ab8a8a?scriptVersionId=364146947), run `20261031T071608_511672Z`. Draft Kaggle terbaru memiliki kode yang sama dengan Version 4, tetapi narasi Phase 2–3 berbeda.

**Tidak ada sel notebook ini yang dijalankan saat penggabungan.** Semua grafik, tabel, dan progres training yang sudah tampil adalah output salinan sumber, bukan hasil eksekusi notebook baru. Nomor eksekusi dikosongkan; asal setiap sel dicatat pada metadata `fgd_source`.

Mode bawaan membaca handoff asli dan artefak Version 4. Kode persiapan Colab tetap tersedia pada posisinya, tetapi dilewati agar cleaning, autoencoder, split, serta augmentasi offline tidak diulang. Kode training dan evaluasi Kaggle juga dipertahankan, tetapi dilewati dalam mode artefak. Test pada eksperimen sumber **sudah dievaluasi**; gunakan hasil tersimpan.

Input Kaggle yang diperlukan: dataset **FGD6 Handoff dan Checkpoint**, serta **output notebook Version 4 (364146947)**. Version 5 adalah Quick Version dengan 0 file output dan tidak dapat menggantikan input model Version 4. File `.ipynb` tidak menyimpan bobot `.keras` di dalamnya.
""")
code('''# Mode bawaan mempertahankan data dan hasil eksperimen yang sudah selesai.
RUN_PREPARATION = False
TRAIN_NEW_EXPERIMENT = False
assert not (RUN_PREPARATION and not TRAIN_NEW_EXPERIMENT), (
    "Data yang dibuat ulang tidak boleh digabung dengan hasil model lama."
)
''')
for i in range(1, 33):
    replacement = None
    guard = None
    if i == 3:
        replacement = COLAB.cells[i].source.replace(
            "!pip -q install gdown opencv-python-headless tensorflow",
            "# Kaggle menyediakan TensorFlow dan OpenCV; tidak memasang ulang versinya.\n"
            "if RUN_PREPARATION:\n    %pip -q install gdown")
        replacement += "\nimport gc, platform, time\nfrom datetime import datetime, timezone\nfrom IPython.display import display\nfrom sklearn.metrics import classification_report, confusion_matrix\n"
    elif i == 5:
        replacement = COLAB.cells[i].source.replace('Path("/content/work")', 'Path("/kaggle/working/fgd6_preparation")')
    elif COLAB.cells[i].cell_type == "code":
        guard = "RUN_PREPARATION"
    imported(COLAB, i, "colab", guard, replacement)

markdown("""### Sambungan handoff Colab → Kaggle

Mode artefak memuat handoff yang dipakai saat training Version 4: 4.000 train (termasuk 1.263 augmentasi offline), 342 validation, dan 343 test. Path `prep_path` dipetakan ke Kaggle tanpa mengganti label atau split. Normalisasi kedua model mengikuti eksperimen Kaggle: `densenet.preprocess_input` sekali. Pipeline identitas `Lightweight_DenseNet` pada Colab tidak dipakai oleh LWDN terlatih ini.
""")
code('''# Memulihkan tabel handoff dan path citra tanpa membuat ulang data.
from pathlib import PurePosixPath
HANDOFF_DIR = Path("/kaggle/input/datasets/ouvvee/fgd6-handoff-dan-checkpoint/FGD6/handoff/20261028T104133_556236Z")
if not RUN_PREPARATION:
    data_metadata = json.loads((HANDOFF_DIR / "metadata.json").read_text(encoding="utf-8"))
    PREP_DIR = HANDOFF_DIR / "images"
    CLASS_NAMES = data_metadata["classes"]
    IMG_SIZE, BATCH, SEED = (data_metadata[k] for k in ("image_size", "batch", "seed"))
    SPLIT = tuple(data_metadata["split"])
    FILE_ID = data_metadata["dataset_file_id"]
    assert data_metadata["counts"] == {"train": 4000, "val": 342, "test": 343}
    assert data_metadata["checks"] and all(row[1] is True for row in data_metadata["checks"])
    df_final = pd.read_csv(HANDOFF_DIR / "df_final.csv")
    train_df = pd.read_csv(HANDOFF_DIR / "train_df.csv")
    mapped_paths = []
    for row in df_final.itertuples():
        relative = PurePosixPath(row.prep_path).relative_to("/content/work/dataset_224")
        assert len(relative.parts) == 3 and relative.parts[:2] == (row.split, row.label)
        assert ".." not in relative.parts
        path = PREP_DIR.joinpath(*relative.parts)
        assert path.is_file(), f"Citra handoff hilang: {path}"
        mapped_paths.append(str(path))
    df_final["prep_path"] = mapped_paths
    assert not df_final["prep_path"].duplicated().any()
    assert df_final["split"].value_counts().to_dict() == data_metadata["counts"]
    augmenter = tf.keras.models.model_from_json(data_metadata["augmenter"])
    tf.keras.utils.set_random_seed(SEED)
    AUG_TARGET = 400
else:
    data_metadata = {
        "classes": list(CLASS_NAMES), "image_size": IMG_SIZE, "batch": BATCH,
        "seed": SEED, "dataset_file_id": FILE_ID, "split": list(SPLIT),
        "counts": df_final["split"].value_counts().to_dict(),
        "augmenter": augmenter.to_json(),
    }
WORK_DIR = Path("/kaggle/working")
''')
for i in range(33, 38):
    replacement = None
    if i == 34:
        replacement = COLAB.cells[i].source.replace(
            '    "Lightweight_DenseNet": lambda x: x,               # model pembanding\n', '')
        replacement += '\ntrain_dataset = datasets["DenseNet121"]["train"]\nval_dataset = datasets["DenseNet121"]["val"]\nassert N_IMAGES == data_metadata["counts"]\n'
    imported(COLAB, i, "colab", replacement=replacement)
    if i == 36:
        code('''# Hentikan alur jika pemeriksaan data gagal.
assert checks and all(row[1] for row in checks), "QC Phase 3 belum lulus."''')

markdown("""### Memulihkan eksperimen Version 4

Keluaran yang diharapkan saat kelak dijalankan: lokasi artefak, konfigurasi, checkpoint terpilih dan tabel hasil tersimpan. Pencarian dibatasi pada **run ID yang persis sama**; notebook berhenti jika input hilang atau ambigu. SHA-256 memeriksa bobot dan metadata terhadap arsip Version 4 yang diunduh. Pemeriksaan ini belum dijalankan di Kaggle.
""")
hashes = {p.relative_to(ARTIFACTS).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
          for p in ARTIFACTS.rglob("*") if p.is_file()}
code('''# Menghubungkan bobot asli dan hasil yang tersimpan, tanpa fit/evaluate/predict.
if not TRAIN_NEW_EXPERIMENT:
    expected_hashes = ''' + pprint.pformat(hashes, width=110).replace("\n", "\n    ") + '''
    matching_runs = [p for p in Path("/kaggle/input").rglob("''' + RUN_ID + '''")
                     if p.is_dir() and (p / "LWDN/selection.json").is_file()]
    assert len(matching_runs) == 1, "Pasang output Kaggle Version 4; run hilang atau ambigu."
    experiment_dir = matching_runs[0]
    for relative, expected in expected_hashes.items():
        path = experiment_dir / relative
        assert path.is_file(), f"Artefak hilang: {relative}"
        with path.open("rb") as handle:
            assert hashlib.file_digest(handle, "sha256").hexdigest() == expected, relative
    source_metadata = json.loads((experiment_dir / "handoff_metadata.json").read_text())
    assert source_metadata == data_metadata, "Metadata handoff berbeda dari eksperimen sumber."
    assert (experiment_dir / "handoff_manifest.csv").read_bytes() == (HANDOFF_DIR / "df_final.csv").read_bytes()
    experiments = {}
    for name in ("DenseNet121", "LWDN"):
        saved_dir = experiment_dir / name
        saved_config = json.loads((saved_dir / "config.json").read_text())
        saved_selection = json.loads((saved_dir / "selection.json").read_text())
        assert saved_config["classes"] == CLASS_NAMES and saved_selection["test_evaluated"] is True
        log = pd.read_csv(saved_dir / (saved_selection["stage"] + ".csv"))
        best = log.loc[log["val_loss"].idxmin()]
        assert np.isclose(best["val_loss"], saved_selection["val_loss"])
        assert int(best["epoch"]) + 1 == saved_selection["epoch_in_stage"]
        experiments[name] = {"run_dir": saved_dir, "selection": saved_selection}
    run_dir = experiments["DenseNet121"]["run_dir"]
    lwdn_run_dir = experiments["LWDN"]["run_dir"]
    config = json.loads((run_dir / "config.json").read_text())
    lwdn_config = json.loads((lwdn_run_dir / "config.json").read_text())
    selected_model_path = run_dir / Path(experiments["DenseNet121"]["selection"]["checkpoint"]).name
    lwdn_selected_model_path = lwdn_run_dir / Path(experiments["LWDN"]["selection"]["checkpoint"]).name
    print("Artefak Version 4 terhubung:", experiment_dir)
''')
for i in range(9, 44):
    if i == 9:
        markdown("**Asal Phase 4–5: Kaggle Version 4.** Sel training berikut dilewati ketika `TRAIN_NEW_EXPERIMENT=False`; output yang tampil tetap merupakan rekaman sumber. Arsitektur serta hyperparameter dipindahkan sebagaimana adanya, tanpa menyatakan adaptasi ini sebagai replikasi penuh paper.")
    if i == 33:
        markdown("**Evaluasi sumber sudah selesai.** Dalam mode artefak, bagian berikut membaca tabel dan grafik tersimpan. Kode evaluasi asli hanya aktif untuk eksperimen training baru. Tidak ada evaluasi test ulang pada pekerjaan penggabungan ini.")
    imported(KAGGLE, i, "kaggle-v4", guard="TRAIN_NEW_EXPERIMENT" if KAGGLE.cells[i].cell_type == "code" else None)
    if i in (34, 36, 38, 40, 42):
        name = {34: "validation_summary.csv", 38: "test_summary.csv", 40: "efficiency.csv", 42: "comparison.csv"}.get(i)
        if name:
            code(f'''# Membaca laporan yang sudah tersimpan tanpa menjalankan inferensi.
if not TRAIN_NEW_EXPERIMENT:
    display(pd.read_csv(experiment_dir / "{name}"))''')
        if i == 36:
            code('''# Menampilkan grafik sumber tanpa menghitung ulang prediksi.
if not TRAIN_NEW_EXPERIMENT:
    from IPython.display import Image as DisplayImage
    for name, experiment in experiments.items():
        for metric in ("loss", "accuracy"):
            display(DisplayImage(filename=str(experiment["run_dir"] / f"learning_curve_{metric}.png")))''')

markdown("""### Batas hasil dan fase terakhir

Sumber berhenti pada **Phase 5: Evaluation**. Tidak ada implementasi Phase 6 Deployment dalam gabungan ini; integrasi aplikasi menunggu pemilihan model tim. Output klasifikasi dan efisiensi di atas berasal dari eksperimen sumber satu seed, bukan pengukuran baru.

Phase 3 sumber mengidentifikasi kandidat salah label dan anomali menggunakan seluruh data sebelum split. Pemeriksaan MD5 hanya mendeteksi duplikat identik; kemiripan foto daun yang sama belum dibuktikan bebas kebocoran. Temuan ini dicatat tanpa mengubah langkah preparation milik engineer data atau mengulang eksperimen.
""")
notebook = nbformat.v4.new_notebook(cells=cells, metadata={
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"},
    "fgd_merge": {"executed": False, "date": "2026-10-31", "run_id": RUN_ID,
                  "kaggle_version": 4, "kaggle_script_version_id": 364146947},
})
nbformat.validate(notebook)
from IPython.core.inputtransformer2 import TransformerManager
transformer = TransformerManager()
for i, cell in enumerate(notebook.cells):
    if cell.cell_type == "code":
        ast.parse(transformer.transform_cell(cell.source), filename=f"cell-{i}")
assert [c.source for c in DRAFT.cells if c.cell_type == "code"] == [c.source for c in KAGGLE.cells if c.cell_type == "code"]
assert all(not c.get("execution_count") for c in notebook.cells)
assert sum(len(c.get("outputs", [])) for c in notebook.cells) == (
    sum(len(c.get("outputs", [])) for c in COLAB.cells[:38])
    + sum(len(c.get("outputs", [])) for c in KAGGLE.cells[9:]))
nbformat.write(notebook, ROOT / "FGD6_End_to_End_Kaggle.ipynb")
(ROOT / "cell-map.json").write_text(json.dumps(mapping, indent=2), encoding="utf-8")
print(f"Validated {len(notebook.cells)} cells; no notebook cells executed.")
