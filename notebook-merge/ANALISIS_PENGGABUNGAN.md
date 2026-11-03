# Analisis penggabungan FGD-6

Tanggal pemeriksaan: 31 Oktober 2026. Tidak ada sel notebook, training, inferensi, atau evaluasi model yang dijalankan pada pekerjaan ini. Pemeriksaan lokal hanya membaca JSON, CSV, prediksi NPZ yang sudah tersimpan, dan struktur arsip.

## Sumber yang diperiksa

| Sumber | Bukti yang diperoleh | Peran dalam gabungan |
|---|---|---|
| [Colab FGD-6](https://colab.research.google.com/drive/1CKksYUrcdfcZtOxGZq2EDlbh_3Jc2DOL) | Unduhan terbaru dari menu Colab; 88 sel | Sel 0–37: judul, Business Understanding, setup, Data Understanding, preparation, pipeline, QC dan tabel kelas |
| [Kaggle editor](https://www.kaggle.com/code/ouvvee/notebookc755ab8a8a/edit) | Unduhan draft; 44 sel, seluruh code cell identik dengan Version 4 | Pembanding perubahan narasi; bukan bukti eksekusi baru |
| [Kaggle Version 4](https://www.kaggle.com/code/ouvvee/notebookc755ab8a8a?scriptVersionId=364146947) | Save & Run All; sumber dengan output; arsip output model terunduh | Modeling dan Evaluation kedua model, sel 9–43 |
| Kaggle Version 5, ID 364165790 | Quick Version; UI melaporkan 0 file output | Tidak dipakai sebagai sumber bobot |

Dua tautan Kaggle dari pengguna merujuk notebook dengan slug sama, bukan dua proyek berbeda. Editor/draft dan versi hasil eksekusi harus dibedakan. Pengguna memilih DenseNet-121 Kaggle Version 4 agar berpasangan dengan LWDN dari eksperimen yang sama.

## Akar masalah alur

1. Kaggle membaca handoff Phase 3 yang sudah disiapkan. Sumber Kaggle 44 sel tidak memiliki kode eksplorasi, cleansing, autoencoder, split dan augmentasi offline lengkap. Phase 2–3 berupa ringkasan dan loader. Ini menjelaskan mengapa kode Phase 1–3 Colab tidak terlihat diproses dari awal di Kaggle. Isi yang tersedia tidak membuktikan penyebab insiden teknis sebelumnya; tidak disimpulkan bahwa Kaggle gagal mengeksekusi kode yang memang tidak ada di sumber tersebut.
2. File notebook dan file bobot terpisah. Menyalin code cell, output training, atau grafik tidak membawa bobot `.keras`. Bobot yang benar ada pada output **Version 4**, bukan output Quick Version terbaru yang kosong.
3. Colab memakai path `/content/work` dan Drive; Kaggle memakai `/kaggle/input` yang read-only serta `/kaggle/working` yang writable. Menyalin path secara mentah memutus dependensi file.
4. Colab Phase 3 menyediakan `DenseNet121` dengan normalisasi DenseNet dan pipeline `Lightweight_DenseNet` identitas. Kaggle LWDN terlatih menggunakan normalisasi DenseNet. Menggunakan pipeline identitas untuk checkpoint Kaggle akan mengubah input yang diterimanya.
5. Bagian LWDN Colab menyimpan `NameError` pada sel 75 dan `KeyboardInterrupt` pada sel 76. Alur recovery Colab itu tidak dipakai menggantikan training Kaggle yang sudah selesai.
6. Sumber aktual berakhir pada Phase 5. Tidak ada Phase 6 Deployment yang dapat disalin; implementasi deployment tidak ditambahkan.

## Rantai dependensi yang direkonsiliasi

`FILE_ID → dataset ZIP → df → metadata/kualitas → df_clean_cleansing → autoencoder/anomali → df_clean → stratified split → letterbox RGB 224 → augmentasi offline train → df_final/PREP_DIR → BASE/make_dataset → datasets["DenseNet121"] → model/config/log → selection berdasarkan val_loss → hasil validation/test → efisiensi dan selisih per kelas`.

Cabang penggunaan ulang hasil:

`handoff 20261028T104133_556236Z → metadata + df_final.csv + images → pemetaan prep_path ke Kaggle → pipeline Colab yang dipakai ulang → QC → run 20261031T071608_511672Z → pemeriksaan SHA-256 → config/selection/checkpoint kedua model → laporan tersimpan`.

Kode Colab Phase 1–3 tetap disalin lengkap. Guard `RUN_PREPARATION=False` melewati pengunduhan data, analisis raw, training autoencoder, pembuatan split dan augmentasi offline yang telah selesai. Output sumber tetap ada dan diberi penjelasan sebagai rekaman historis. Loader handoff mengisi variabel yang dibutuhkan pipeline dan QC, sehingga runtime baru tidak bergantung pada variabel tersembunyi dari Colab.

Pipeline utama memakai kembali helper `load_base` dan `make_dataset` Colab, tanpa menduplikasi loader TensorFlow Kaggle. Pipeline identitas LWDN yang tidak dipakai pada eksperimen sumber dikeluarkan. `CLASS_WEIGHT` Colab masih dihitung sebagai bagian handoff; training Kaggle sumber memang tidak menggunakannya (`class_weight=null`), dan perilaku tersebut dipertahankan.

`TRAIN_NEW_EXPERIMENT=False` melewati pembuatan model ImageNet, fit, pemilihan ulang dan evaluasi asli Kaggle. Dalam mode ini notebook membaca model dan laporan Version 4; status `test_evaluated=true` tetap dipertahankan. Mode training baru hanya tersedia sebagai kode sumber yang dipindahkan, belum divalidasi dengan eksekusi. Jangan mengaktifkannya untuk pekerjaan penggabungan ini. Data yang dibuat ulang tidak boleh dicampurkan dengan hasil model lama.

## Artefak asli yang sudah diperiksa

Arsip di `sources/kaggle-v4-output.zip` lolos `ZipFile.testzip()`. Salinan hasil ekstraksi berada di `artifacts/runs/20261031T071608_511672Z/`.

Untuk masing-masing DenseNet121 dan LWDN tersedia `config.json`, `selection.json`, `warmup.csv`, `finetune.csv`, `warmup_best.keras`, `finetune_best.keras`, `inference.keras`, prediksi/metrics/confusion matrix validation dan test, kurva pembelajaran, serta catatan latensi. Tabel tingkat eksperimen mencakup metadata/manifest handoff, kelas, comparison, efficiency, per_class_comparison, test_summary dan validation_summary.

Keenam arsip `.keras` lolos pemeriksaan ZIP dan memiliki `config.json`, `metadata.json`, serta `model.weights.h5`. Model tidak dimuat ke TensorFlow, sehingga kompatibilitas deserialisasi pada runtime Kaggle baru belum dibuktikan.

| Pemeriksaan | Hasil dari file sumber |
|---|---|
| Handoff | 4.000 train, 342 validation, 343 test; 10 kelas; seed 42; batch 32; input 224 |
| Train asli + augmentasi | 2.737 + 1.263 = 4.000 |
| Konfigurasi sumber | Python 3.13.15, TensorFlow 2.20.0, Keras 3.13.2, Tesla T4, mixed_float16 |
| DenseNet checkpoint | Fine-tuning epoch 24; val_loss 0,0723770484; minimum dari log warm-up dan fine-tuning |
| LWDN checkpoint | Fine-tuning epoch 24; val_loss 0,9166226983; minimum dari log warm-up dan fine-tuning |
| Prediksi sumber | Argmax probabilitas sesuai y_pred; confusion matrix sesuai pasangan y_true/y_pred untuk kedua model dan kedua split |
| Format notebook gabungan | Lulus validasi nbformat; semua code cell lolos parsing setelah transformasi sintaks IPython |
| Preservasi | Jumlah output salinan sama dengan total output dari bagian sumber terpilih; code cell draft Kaggle identik dengan Version 4 |

Angka berikut adalah hasil **eksperimen sumber**, bukan hasil notebook baru:

| Metrik | DenseNet-121 | LWDN |
|---|---:|---:|
| Accuracy validation | 97,0760% | 65,7895% |
| Accuracy test | 98,8338% | 68,8047% |
| Parameter | 7.047.754 | 1.644.938 |
| Ukuran inference | 28,3148 MiB | 6,8345 MiB |
| Waktu training | 936,3250 detik | 726,3664 detik |
| Median inferensi sumber | 16,5397 ms/citra | 6,2275 ms/citra |

Selisih accuracy test LWDN − DenseNet-121 adalah −30,0292 poin persentase. Ini tidak dipakai untuk tuning atau pemilihan checkpoint baru. Latensi sumber memakai GPU:0, batch 1, 10 warm-up dan 100 pengulangan; preprocessing tidak termasuk. Tidak dilakukan benchmark ulang.

## Batas metodologis yang dipertahankan dan dicatat

- Deteksi kandidat salah label dan autoencoder pada sumber membaca seluruh data sebelum split. Itu perlu disebutkan saat membahas independensi test; gabungan ini tidak mengubah protokol engineer data atau mengklaim audit leakage menyeluruh.
- QC MD5 mendeteksi duplikat identik, belum membuktikan ketiadaan foto sangat mirip atau daun yang sama antar-split.
- Handoff sudah berisi augmentasi offline train. Pipeline sumber juga memberi augmentasi online train. Keduanya dipertahankan; tidak ditambah normalisasi kedua atau augmentasi validation/test.
- Dua cabang dan cutoff LWDN dipindahkan sebagaimana sudah dilatih pada Kaggle. Pekerjaan ini tidak menyetujui arsitektur baru atau membuktikan adaptasinya identik dengan paper.
- Output sumber Colab serta Kaggle berasal dari runtime berbeda. Nomor execution_count gabungan dikosongkan; metadata asal tiap sel dan penjelasan historis mencegahnya dibaca sebagai satu run baru yang sudah berhasil.

## Status impor Kaggle dan langkah melanjutkan

Notebook gabungan lokal: `FGD6_End_to_End_Kaggle.ipynb`. Peta asal sel: `cell-map.json`. Penyusun sekaligus pemeriksaan statis yang dapat diulang: `python notebook-merge/merge_notebooks.py`; perintah ini tidak mengeksekusi notebook.

Draft baru sempat dibuat melalui menu output Version 4 di [editor notebook baru](https://www.kaggle.com/code/ouvvee/notebookc3744918d5/edit). Editor tertahan pada **Loading Editor**, melaporkan gagal terhubung ke layanan reCAPTCHA, dan menu **Import Notebook** nonaktif. Pemuatan ulang belum menyelesaikannya. Isi gabungan dan sambungan input **belum berhasil dimasukkan atau diverifikasi di draft Kaggle tersebut**. Notebook sumber tidak diubah dan sesi training tidak dimulai.

Setelah editor Kaggle dapat dibuka:

1. Impor `FGD6_End_to_End_Kaggle.ipynb` ke notebook baru melalui File → Import Notebook. Jangan tekan Run All atau Save & Run All.
2. Tambahkan dataset `ouvvee/fgd6-handoff-dan-checkpoint` dan output notebook `ouvvee/notebookc755ab8a8a` yang dipilih secara eksplisit pada **Version 4 / 364146947**. Jangan memakai Version 5 yang tidak mempunyai file output.
3. Pastikan tree input memuat handoff `20261028T104133_556236Z` serta `runs/20261031T071608_511672Z/{DenseNet121,LWDN}`. Kode recovery mencari run ID persis itu dan menolak input hilang atau ambigu.
4. Biarkan kedua flag tetap `False`. Pada tahap ini cukup review isi; belum ada izin menjalankan notebook.

Pemeriksaan visual keseluruhan notebook hasil impor dan uji runtime ditunda karena editor belum dapat dimuat dan pengguna melarang eksekusi. Renderer `jupyter-nbconvert` juga tidak tersedia pada lingkungan lokal; tidak ada klaim bahwa tampilan akhir notebook baru telah lolos inspeksi visual.
