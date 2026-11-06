"""Jalankan: python self_check.py (tanpa framework tes tambahan)."""

from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from PIL import Image
from streamlit.testing.v1 import AppTest

from app import CLASS_NAMES, MAX_BYTES, PLANT_INFO, load_model, predict, read_image


def image_bytes(image_format="PNG", size=(40, 20), exif=None, color="green"):
    buffer = BytesIO()
    kwargs = {"exif": exif} if exif else {}
    Image.new("RGB", size, color).save(buffer, format=image_format, **kwargs)
    return buffer.getvalue()


def preview_size(data):
    _, preview = read_image(data)
    with Image.open(BytesIO(preview)) as image:
        assert image.mode == "RGB"
        return image.size


def rejects(data, message):
    try:
        read_image(data)
    except ValueError as error:
        assert message in str(error), str(error)
    else:
        raise AssertionError("Input seharusnya ditolak")


assert preview_size(image_bytes()) == (40, 20)
assert preview_size(image_bytes(size=(2000, 1000))) == (1000, 500)
assert preview_size(image_bytes(size=(2001, 1))) == (1000, 1)
exif = Image.Exif()
exif[274] = 6
rotated, preview = read_image(image_bytes("JPEG", exif=exif))
with Image.open(BytesIO(preview)) as image:
    assert image.size == (20, 40)
assert (rotated[0, 94, 112] < 255).all() and (rotated[0, 112, 94] == 255).all()
rejects(b"ini bukan gambar", "tidak dapat dibaca")
rejects(image_bytes()[:30], "tidak dapat dibaca")
rejects(image_bytes("GIF"), "PNG atau JPG")
rejects(b"0" * (MAX_BYTES + 1), "10 MB")
rejects(image_bytes(size=(5000, 4001)), "20 megapiksel")
assert set(CLASS_NAMES) == set(PLANT_INFO)
assert all(info["latin"] and len(info["benefits"]) == 3 for info in PLANT_INFO.values())

photo_data = image_bytes()
batch, _ = read_image(photo_data)
assert batch.shape == (1, 224, 224, 3) and batch.dtype == "float32"
assert batch.min() == 0 and batch.max() == 255, "Model memerlukan RGB mentah 0-255"
white_photo = image_bytes(size=(224, 224), color="white")
blank, _ = read_image(white_photo)
label, score, elapsed_ms = predict(blank, load_model())
assert label is None and 0 <= score < 1 and elapsed_ms >= 0

app_path = str(Path(__file__).with_name("app.py"))
app = AppTest.from_file(app_path).run(timeout=120)
assert not app.exception, app.exception
assert app.button[0].disabled and app.radio[0].value == "Foto"
app.radio[0].set_value("Folder").run()
assert not app.exception, app.exception
assert app.button[0].disabled
assert app.get("file_uploader")[0].proto.max_upload_size_mb == 10

photo = SimpleNamespace(file_id="test-photo", name='foto"<uji>.png', size=len(white_photo), getvalue=Mock(return_value=white_photo))
with patch("streamlit.file_uploader", return_value=photo):
    app = AppTest.from_file(app_path).run(timeout=120)
    app.button[0].click().run(timeout=120)
    assert not app.exception, app.exception
    assert any("Foto tidak dikenali" in item.value for item in app.warning)
    assert not any("Nama daun:" in item.value or "Khasiat" in item.value for item in app.get("markdown"))
    assert any("Skor model:" in item.value and "Waktu inferensi:" in item.value for item in app.caption)
    assert any("Prediksi selesai" in item.label for item in app.get("status"))
    app.run(timeout=120)
    assert any("Foto tidak dikenali" in item.value for item in app.warning)
    photo.getvalue.assert_called_once()


def run_folder():
    from types import SimpleNamespace
    from unittest.mock import patch

    import numpy as np

    from app import main

    def scores(batch):
        values = np.full((1, 10), 0.1)
        if batch.min() < 255:
            values.fill(0.001)
            values[0, 0] = 0.991
        return SimpleNamespace(numpy=lambda: values)

    with patch("app.load_model", return_value=scores):
        main()


folder_files = [
    SimpleNamespace(file_id="folder-1", name="daun-1.png", size=len(photo_data), getvalue=lambda: photo_data),
    SimpleNamespace(file_id="folder-2", name="kosong.png", size=len(white_photo), getvalue=lambda: white_photo),
    SimpleNamespace(file_id="folder-3", name="rusak.png", size=5, getvalue=lambda: b"rusak"),
]
with patch("streamlit.file_uploader", side_effect=lambda *args, **kwargs: folder_files if kwargs["key"] == "folder" else None):
    app = AppTest.from_function(run_folder).run(timeout=120)
    app.radio[0].set_value("Folder").run(timeout=120)
    assert not app.button[0].disabled
    app.button[0].click().run(timeout=120)
    assert not app.exception, app.exception
    names = [item.value for item in app.get("markdown") if "Nama daun:" in item.value]
    assert names == [f"**Nama daun:** {CLASS_NAMES[0]}"]
    assert any(PLANT_INFO[CLASS_NAMES[0]]["latin"] in item.value for item in app.get("markdown"))
    assert any("Foto tidak dikenali" in item.value for item in app.warning)
    assert any("Foto tidak dapat dibaca" in item.value for item in app.error)

print("LULUS: validasi foto, inferensi LWDN, penggunaan ulang pratinjau, dan hasil folder campuran.")
