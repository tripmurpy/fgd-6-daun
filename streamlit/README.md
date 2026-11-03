# HerbLeaf AI

Aplikasi Streamlit untuk mengenali foto daun dengan DenseNet-121. Gambar ditampilkan di kiri, sementara nama daun, nama latin, dan khasiat yang diteliti ditampilkan langsung di kanan.

## Model

`model/densenet121.keras` dan `model/classes.json` berasal dari output Kaggle **Version 4 / 364146947**, run `20261031T071608_511672Z`. File sumber yang sudah diunduh tersimpan di `../notebook-merge/sources/kaggle-v4-output.zip`. Bobot yang dipakai adalah ekspor `inference.keras` dari checkpoint DenseNet-121 dengan `val_loss` terendah, bukan bobot dari versi Kaggle terbaru.

Input mengikuti training: orientasi EXIF, RGB, letterbox putih 224 × 224 dengan LANCZOS, lalu `densenet.preprocess_input` sekali. Urutan label berasal dari `classes.json`.

## Jalankan lokal

Dari folder `streamlit`:

```powershell
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Buka `http://localhost:8501`. Untuk mengecek aplikasi dan bobot DenseNet-121, jalankan `python self_check.py`.

## Tampilan dan batas hasil

Unggah satu foto atau folder PNG/JPG, maksimal 20 foto per proses dan 10 MB/20 megapiksel per foto. Saat diproses, aplikasi menampilkan status memuat dan memprediksi gambar. Skor keyakinan adalah probabilitas softmax, bukan bukti prediksi benar. Waktu inferensi diukur pada proses aplikasi untuk satu foto; waktu membaca dan menyiapkan gambar tidak termasuk. Waktu ini bergantung pada perangkat dan berbeda dari benchmark GPU Kaggle.

Foto tidak ditulis ke disk oleh aplikasi. Tidak ada masukan label asli atau penilaian benar/salah; aplikasi hanya menampilkan hasil prediksi.

Nama latin dan ringkasan khasiat tiap kelas disimpan di `model/plant_info.json` berdasarkan materi dari tim. Khasiat mengikuti label yang diprediksi, sehingga dapat merujuk pada tanaman yang salah bila prediksinya keliru. Sebagian temuan berasal dari studi laboratorium/hewan atau bagian buah, bukan daun; ringkasan ini bukan saran medis. Rujukan ilmiah untuk seluruh 30 poin perlu dilengkapi dan ditinjau sebelum publikasi.

Untuk Streamlit Community Cloud, pilih Python 3.13 dan arahkan entrypoint ke `streamlit/app.py`; paket Python dibaca dari `streamlit/requirements.txt`. Deployment publik belum dilakukan.
