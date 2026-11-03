# Strategi 1 — Baseline fine-tuning DenseNet-121

Tanggal peninjauan: 24 Oktober 2026. Ruang lingkup: rencana kerja di notebook bersama, tanpa menulis kode training atau menjalankan eksperimen. Keputusan di bawah mengikuti diskusi dan jawaban pengguna; dokumen menunggu konfirmasi pemahaman akhir.

## 1. Tujuan dan batas selesai

Menghasilkan satu baseline DenseNet-121 yang dipilih menggunakan validation, tersimpan dengan konfigurasi dan bukti yang bisa ditelusuri. Baseline ini membantu memeriksa alur dan perilaku model sebelum penelitian dengan beberapa pengulangan serta perbandingan dengan LWDN.

Training terdiri atas dua tahap: latih classifier baru saat backbone dibekukan, kemudian sesuaikan blok terakhir backbone dengan learning rate lebih kecil. Backbone adalah bagian model yang mengekstrak ciri gambar; classifier mengubah ciri tersebut menjadi probabilitas 10 jenis daun. Fine-tuning dapat membaik atau memburuk; checkpoint warm-up tetap menjadi kandidat.

Tahap ini selesai setelah checkpoint terpilih bisa dimuat ulang, hasil validation tersimpan, waktu training tercatat, dan identitas data serta lingkungan eksperimen tersedia. Accuracy tertentu bukan syarat keberhasilan yang dijanjikan.

Target baseline yang diharapkan:

- Satu run dengan seed 42 menghasilkan checkpoint terpilih yang dapat dimuat ulang dan diverifikasi.
- Fine-tuning diharapkan menurunkan `val_loss` dibandingkan checkpoint warm-up terbaik. Ini hipotesis yang diperiksa; jika tidak tercapai, pilih checkpoint warm-up sesuai aturan yang disepakati.
- Seluruh data validation mendapat prediksi dan laporan metrik lengkap untuk 10 kelas, termasuk kelas yang paling sering tertukar.
- Konfigurasi, identitas data, riwayat training, waktu, perangkat, dan checkpoint tersimpan dalam satu folder run yang dapat ditelusuri.

Target angka accuracy dan macro F1 belum ditetapkan. Catat hasil pengukuran baseline terlebih dahulu sebagai dasar diskusi eksperimen berikutnya.

Implementasi LWDN, evaluasi test final, pencarian hyperparameter otomatis, dan integrasi Streamlit berada di tahap berikutnya. Satu seed belum menunjukkan kestabilan hasil antarpercobaan.

## 2. Dasar pemeriksaan notebook

Satu-satunya acuan notebook proyek: [FGD-6 Google Colab](https://colab.research.google.com/drive/1CKksYUrcdfcZtOxGZq2EDlbh_3Jc2DOL), dibaca melalui Google Drive pada 24 Oktober 2026. Pembacaan mencakup source dan output tersimpan, tanpa menghubungkan runtime atau menjalankan cell. Data, variabel, alur preparation, dan posisi penambahan cell mengikuti notebook bersama ini. Dasar metode memakai dokumentasi resmi TensorFlow/Keras; setelan baseline mengikuti keputusan diskusi pengguna.

| Hal | Hasil pemeriksaan |
| --- | --- |
| Notebook bersama | 25 cell: 13 kode dan 12 Markdown; berakhir di validasi Data Preparation |
| Modifikasi tersimpan | 23 Oktober 2026, 08:43:33 Asia/Bangkok |
| Revision Drive | `0B5Dtmh4ScTx_TzRvUENUMkNMc21wa0YxWm1jcWloMjdjSDc4PQ` |
| Modeling | Belum ada training DenseNet-121 di notebook bersama |
| Handoff tabel | `df_split`; `df_master` dibuat sebagai salinan untuk pemeriksaan |
| Handoff pipeline | `datasets["DenseNet121"]["train"]` dan `datasets["DenseNet121"]["val"]` |
| Pipeline input | Cell fisik 23, ID `e7xe5KJi0Xdf` |
| Data pada output tersimpan | 3.414 gambar: 2.731 train, 341 validation, 342 test |
| Konfigurasi preparation | Seed 42, split 80/10/10, batch 32, gambar RGB 224 × 224 |
| Label | One-hot 10 kelas, urutan mengikuti `CLASS_NAMES` |
| Lingkungan tersimpan | Output menyebut TensorFlow 2.21.0 dan GPU tersedia; metadata menyebut T4 |

Jumlah data dan informasi perangkat di atas berasal dari notebook tersimpan, bukan bukti runtime aktif. Validasi terakhir tersimpan melaporkan nol error; beberapa `execution_count` kosong. Pemeriksaan kesiapan training tetap diperlukan pada runtime yang dipakai nanti.

## 3. Library dan kontrak input

| Library | Tugas |
| --- | --- |
| TensorFlow melalui `tf.keras` | DenseNet121 ImageNet, classifier, loss, optimizer, callbacks, training, dan penyimpanan model |
| `tf.data` yang sudah tersedia | Memasok batch dari pipeline preparation yang sama |
| scikit-learn yang sudah tersedia | Accuracy, precision/recall/F1 macro dan per kelas, serta confusion matrix validation |
| NumPy, pandas, Matplotlib yang sudah tersedia | Pemeriksaan angka, tabel hasil, dan kurva training |
| Standard library Python | Waktu, konfigurasi JSON, identitas file, dan informasi lingkungan |

Gunakan instalasi notebook yang tersedia dan catat versinya. Tidak diperlukan framework training kedua atau dependency tambahan.

Alur input yang sudah ada:

**JPEG RGB hasil letterbox → cache gambar dasar → shuffle train → batch → float32 → augmentasi train → `densenet.preprocess_input` → prefetch → model.**

Validation dan test tidak diacak atau diaugmentasi; batch terakhir tetap dipakai. Pertahankan pipeline ini. Model menerima tensor yang sudah diproses DenseNet, sehingga tidak memerlukan `Rescaling(1/255)`, normalisasi kedua, atau augmenter kedua. Kontrak preprocessing perlu disimpan bersama checkpoint agar gambar tidak diproses dua kali saat model digunakan kembali. Lihat [dokumentasi preprocessing DenseNet](https://www.tensorflow.org/api_docs/python/tf/keras/applications/densenet/preprocess_input).

Pemeriksaan awal cukup menggunakan hasil QC yang sudah ada, metadata handoff, dan satu batch. Jangan mengulang inventaris gambar, cleaning, split, atau seluruh pemeriksaan file. Jika daftar `errors` berisi masalah, training harus berhenti; cell validasi saat ini hanya mencetak status.

## 4. Keputusan baseline

| Keputusan | Nilai dan alasan |
| --- | --- |
| Cakupan eksperimen | Satu run dengan training seed 42; split preparation tetap seed 42 |
| Bobot awal | DenseNet121 ImageNet, classifier ImageNet dilepas |
| Head | Global Average Pooling → Dropout 0,3 → Dense 10 dengan softmax |
| Loss | Categorical cross-entropy, sesuai label one-hot |
| Optimizer | Adam; optimizer baru saat compile tahap fine-tuning |
| Warm-up | Backbone beku; maksimal 10 epoch; learning rate 0,001 |
| Fine-tuning | Dari checkpoint warm-up terbaik; hanya blok `conv5_` dan classifier dilatih; maksimal 25 epoch; learning rate 0,00001 |
| Batch Normalization | Tetap beku; backbone dipanggil dalam inference mode untuk menjaga statistiknya |
| Pemilihan checkpoint | `val_loss` minimum, dengan aturan yang sama di kedua tahap |
| Early stopping | Pantau `val_loss`; usulan teknis: patience 3 pada warm-up dan 5 pada fine-tuning, min_delta 0 |
| Presisi | Mixed precision jika GPU mendukung; float32 jika tidak; output softmax tetap float32 |
| Bobot sampel | Seragam; `CLASS_WEIGHT` tidak diteruskan pada kedua tahap baseline |
| Perubahan learning rate | Tetap per tahap; belum memakai scheduler tambahan |

Angka training tersebut merupakan titik awal yang disetujui, belum terbukti optimal. Epoch merupakan batas maksimum. Jangan mengganti aturan checkpoint setelah melihat hasil, atau memilih berdasarkan test. Jika nilai minimum kedua tahap sama, pertahankan checkpoint warm-up.

Pola freezing, fine-tuning dengan learning rate kecil, compile ulang setelah perubahan `trainable`, dan perlakuan khusus BatchNorm mengikuti [panduan transfer learning Keras](https://keras.io/guides/transfer_learning/). Mixed precision dipilih setelah mengecek GPU aktual dan dicatat bersama versi software; kebijakan presisi perlu konsisten saat efisiensi dibandingkan dengan LWDN. Output softmax float32 dan pemilihan perangkat mengacu pada [panduan mixed precision TensorFlow](https://www.tensorflow.org/guide/mixed_precision).

### Alasan tidak memakai class weighting

Pengguna menyerahkan analisis ini kepada engineer. Distribusi train pada output Colab adalah:

| Kelas | Gambar train |
| --- | ---: |
| Belimbing Wuluh | 279 |
| Jambu Biji | 277 |
| Jeruk Nipis | 259 |
| Kemangi | 280 |
| Lidah Buaya | 278 |
| Nangka | 275 |
| Pandan | 279 |
| Pepaya | 252 |
| Seledri | 279 |
| Sirih | 273 |
| **Total** | **2.731** |

Rasio terbesar terhadap terkecil adalah 280/252 = 1,11×. Bobot balanced yang dihitung dari jumlah ini hanya berkisar 0,975–1,084, menggunakan rumus jumlah train / (10 × jumlah kelas tersebut). Ini perhitungan dari output tersimpan, bukan pembacaan objek runtime.

Distribusinya cukup dekat seimbang untuk memulai dengan bobot seragam. Itu keputusan baseline, bukan bukti bahwa reweighting tidak akan pernah berguna. Objek `CLASS_WEIGHT` tetap boleh tersedia; periksa recall dan F1 tiap kelas pada validation untuk memahami kesulitan model. Jangan mengubah loss weighting diam-diam di tengah eksperimen.

## 5. Scope per cell

Gunakan ID rencana D01–D08 agar tidak bergantung pada nomor posisi notebook. Sebelum masing-masing code cell, tambahkan satu Markdown singkat dengan dua bagian: **Goal** menjelaskan tujuan, dan **Target yang diharapkan** menyebut keluaran serta kriteria yang akan diperiksa. Rencana ini terdiri atas **8 code cell + 8 Markdown**, ditambahkan sesudah Data Preparation setelah memeriksa versi terbaru Colab bersama.

| Cell | Satu tanggung jawab | Input yang dipakai | Keluaran dan pemeriksaan |
| --- | --- | --- | --- |
| **D01 — Terima handoff** | Memastikan data siap dipakai training | Pipeline DenseNet, `df_split`/`df_master`, `CLASS_NAMES`, `N_IMAGES`, `errors` | Ringkasan jumlah data dan satu batch; berhenti jika QC gagal, dataset/variabel hilang, label bukan one-hot 10 kelas, bentuk input salah, atau angka tidak finite. Gunakan QC sebelumnya untuk pemeriksaan file dan duplikasi. |
| **D02 — Tetapkan eksperimen** | Membekukan konfigurasi serta identitas run | Keputusan baseline, seed preparation, metadata runtime, lokasi artefak | Config tersimpan sebelum training, perangkat/versi/precision tercatat, manifest dan urutan kelas disalin ke folder run unik. Pastikan GPU aktif; periksa Drive dan mount satu kali jika belum terpasang. |
| **D03 — Bangun model** | Membuat arsitektur sesuai kontrak input | Konfigurasi dan satu batch dari D01 | Backbone ImageNet + head 10 kelas; backbone beku; output `(batch, 10)` finite dan jumlah probabilitas mendekati 1; jumlah parameter total/trainable dicatat. Pengecekan model berada dalam cell ini. |
| **D04 — Latih classifier** | Menjalankan warm-up dan menyimpan hasil terbaik tahap awal | Model D03, train/validation, aturan callbacks | Compile, callbacks, fit, waktu warm-up, log epoch, dan `warmup_best.keras`. Sebelum lanjut, muat bobot warm-up terbaik; berhenti bila hasil tidak valid. |
| **D05 — Sesuaikan blok akhir** | Melakukan fine-tuning dari hasil warm-up | Bobot warm-up terbaik dan pipeline yang sama | Buka hanya `conv5_` serta head, pastikan BatchNorm beku, compile ulang dengan Adam baru dan LR kecil, lalu fit. Simpan waktu, log, dan `finetune_best.keras`. |
| **D06 — Pilih checkpoint** | Menentukan dan memverifikasi kandidat terbaik dari dua tahap | Checkpoint dan riwayat validation kedua tahap | Pilihan dengan `val_loss` terendah; muat ulang checkpoint dan cocokkan prediksi pada batch pemeriksaan. Catat tahap, epoch, nilai selection, dan path. |
| **D07 — Analisis validation** | Menjelaskan perilaku checkpoint terpilih | Model terpilih, validation tanpa augmentasi/shuffle, riwayat training | Accuracy, precision/recall/F1 macro, hasil per kelas, confusion matrix, serta kurva loss/accuracy train–validation. Label dan prediksi harus sejajar menurut urutan pipeline. |
| **D08 — Tutup baseline** | Memastikan artefak dan ringkasan eksperimen lengkap | Konfigurasi, manifest, log, checkpoint terpilih, hasil D07 | Ringkasan hasil validation, waktu per tahap/total, parameter, lokasi dan ukuran file checkpoint, keputusan pemilihan, serta keterbatasan tersimpan. Status test tetap belum dievaluasi. |

### Catatan Markdown sebelum setiap code cell

**Note D01 — Terima handoff**

> **Goal:** Memastikan hasil preparation di Colab siap digunakan oleh DenseNet-121.
>
> **Target yang diharapkan:** QC lulus, jumlah data sesuai handoff, serta satu batch memiliki gambar RGB 224 × 224 dan label one-hot 10 kelas dengan angka valid. Training berhenti jika pemeriksaan gagal.

**Note D02 — Tetapkan eksperimen**

> **Goal:** Menetapkan identitas dan konfigurasi satu run agar hasilnya dapat ditelusuri.
>
> **Target yang diharapkan:** Seed 42, setelan training, GPU aktual, versi library, dan kebijakan presisi tercatat; konfigurasi, manifest, serta urutan kelas tersimpan di folder run unik sebelum training dimulai.

**Note D03 — Bangun model**

> **Goal:** Menyiapkan DenseNet-121 ImageNet dengan classifier untuk 10 jenis daun.
>
> **Target yang diharapkan:** Backbone beku dan head siap dilatih; setiap gambar menghasilkan 10 probabilitas valid yang berjumlah mendekati 1. Jumlah parameter total dan trainable tercatat.

**Note D04 — Latih classifier**

> **Goal:** Mengajarkan classifier baru membedakan kelas daun menggunakan fitur pretrained yang tetap beku.
>
> **Target yang diharapkan:** Training berjalan maksimal 10 epoch tanpa angka tidak valid; checkpoint dengan `val_loss` warm-up terendah, log epoch, serta durasinya tersimpan. Bobot terbaik tahap ini dimuat sebelum fine-tuning.

**Note D05 — Sesuaikan blok akhir**

> **Goal:** Menyesuaikan fitur pada blok terakhir DenseNet dengan karakter daun herbal menggunakan learning rate kecil.
>
> **Target yang diharapkan:** Hanya blok `conv5_` dan classifier yang dilatih, BatchNorm tetap beku, serta training berjalan maksimal 25 epoch. Sasaran peningkatannya adalah `val_loss` lebih rendah daripada warm-up; simpan hasil aktual, checkpoint terbaik, log, dan durasi meskipun peningkatan tidak terjadi.

**Note D06 — Pilih checkpoint**

> **Goal:** Memilih kandidat terbaik dari warm-up dan fine-tuning menggunakan aturan yang sama.
>
> **Target yang diharapkan:** Satu checkpoint terpilih berdasarkan `val_loss` minimum kedua tahap. Tahap, epoch, nilai selection, dan lokasi file tercatat; model dapat dimuat ulang dengan prediksi yang sesuai dalam toleransi numerik yang dicatat.

**Note D07 — Analisis validation**

> **Goal:** Memahami kualitas klasifikasi dan jenis kesalahan checkpoint terpilih.
>
> **Target yang diharapkan:** Seluruh data validation tercakup dalam accuracy, precision/recall/F1 macro, laporan 10 kelas, dan confusion matrix. Kurva training serta analisis menjelaskan kelas yang paling sering tertukar dan gejala overfitting berdasarkan hasil aktual.

**Note D08 — Tutup baseline**

> **Goal:** Menyiapkan hasil baseline yang lengkap untuk ditinjau sebelum eksperimen berikutnya.
>
> **Target yang diharapkan:** Konfigurasi, manifest, urutan kelas, log, checkpoint terpilih, hasil validation, parameter, ukuran file checkpoint, dan waktu training dapat ditemukan dalam satu folder run. Ringkasan mencatat keterbatasan eksperimen dan status test yang belum dievaluasi.

D04 dan D05 masing-masing merupakan satu proses training lengkap. Callbacks disiapkan sebagai bagian tahap tersebut, sehingga tidak memerlukan cell terpisah. Pakai komponen bawaan: ModelCheckpoint, EarlyStopping, CSVLogger, dan TerminateOnNaN. Checkpoint dan log kedua tahap menggunakan nama berbeda. Lihat [ModelCheckpoint](https://keras.io/api/callbacks/model_checkpoint/) dan [EarlyStopping](https://keras.io/api/callbacks/early_stopping/).

Jika fine-tuning tidak membaik, D06 memilih warm-up. Jika suatu tahap gagal atau menghasilkan angka tidak valid, laporkan kegagalannya dan pertahankan checkpoint yang sudah aman; jangan melaporkannya sebagai training sukses.

Seed dan konfigurasi membantu penelusuran eksperimen, tetapi tidak menjamin hasil identik pada semua perangkat. Saat mengulang run, pastikan state awal model, optimizer, shuffle, dan augmentasi benar-benar direset; jumlah seed yang ditulis harus sesuai run yang dijalankan.

## 6. Cara membaca hasil baseline

`val_loss` menjadi aturan pemilihan model. Macro F1, recall per kelas, dan confusion matrix menjelaskan jenis kesalahannya. Accuracy tinggi dapat menyembunyikan satu kelas yang sering tertukar; hasil per kelas tetap diperiksa. Macro berarti setiap kelas mempunyai kontribusi sama pada rata-rata metrik.

Gunakan scikit-learn untuk laporan multiclass dengan prediksi argmax dan urutan label `CLASS_NAMES`. Pertahankan seluruh 10 kelas dalam laporan dan tangani pembagian nol secara eksplisit. Hindari memakai Precision/Recall biner dengan threshold 0,5 sebagai pengganti metrik multiclass tersebut.

Waktu training adalah durasi aktual kedua pemanggilan fit, termasuk validation dan overhead callbacks yang terjadi di dalamnya. Catat durasi tiap tahap dan total; waktu download/preparation tidak dimasukkan. Ukuran file checkpoint dicatat sebagai ukuran artefak, karena isinya dapat mencakup state optimizer. Jangan langsung menyebutnya ukuran model inferensi untuk perbandingan arsitektur.

Kurva train–validation membantu mendeteksi gejala overfitting, tetapi satu kurva atau satu seed tidak cukup untuk menyatakan model pasti mampu bekerja pada foto lapangan. Hasil validation merupakan dasar diagnosis; perubahan konfigurasi berikutnya harus menjadi run baru dengan alasan yang dicatat.

## 7. Handoff dan keterbatasan yang diterima

**Persistensi data.** Notebook bersama belum memasang Drive atau mengekspor manifest/folder hasil preparation. `PREP_DIR` berada di `/content/work/dataset_224`, yang bisa hilang ketika runtime berakhir. D02 menyimpan identitas data dan artefak model; tindakan itu belum otomatis mempersistenkan seluruh gambar. Jika data hilang sebelum training, gunakan handoff folder/manifest versi yang sama dari penanggung jawab preparation. Jangan merekonstruksi split atau menjalankan cell preparation yang menghapus folder hasil tanpa koordinasi.

**Filtering sebelum split.** Cell fisik 17 menggunakan fitur DenseNet121 ImageNet dan prediksi out-of-fold LogisticRegression atas seluruh data untuk membuang 8 kandidat salah label sebelum split. Ini dapat menyeleksi data menurut representasi DenseNet dan memengaruhi independensi test serta keadilan perbandingan. Out-of-fold dan pemeriksaan hash MD5 tidak otomatis menghilangkan risiko metodologis tersebut. Pengguna memilih menerima handoff ini dan mencatat keterbatasannya. Strategi modeling tidak mengubah hasil cleaning atau mengklaim bahwa kandidat tersebut sudah terbukti salah label.

**Batas bukti.** Audit ini membaca notebook tersimpan. Keberadaan file, GPU, ketersediaan RAM/VRAM, durasi, hasil training, dan keberhasilan memuat ulang checkpoint belum diuji pada runtime aktif. Artefak dan angka penelitian hanya boleh diisi setelah cell terkait benar-benar dijalankan.

## 8. Sambungan ke perbandingan final

Setelah baseline DenseNet selesai, tinjau hasil validation dan sepakati protokol pengulangan, arsitektur/adaptasi paper LWDN, serta training yang sebanding. Jangan otomatis memakai cabang pipeline `Lightweight_DenseNet` yang sekarang masih berupa identitas input sebagai bukti model LWDN sudah tersedia.

Bekukan konfigurasi dan checkpoint kedua model sebelum evaluasi test berpasangan satu kali. Gunakan prediksi test tersebut untuk semua metrik dan analisis kesalahan; jangan menggunakannya untuk memilih checkpoint, seed, atau hyperparameter.

Tahap final perlu melaporkan accuracy, precision, recall, macro F1, hasil per kelas, confusion matrix, parameter, ukuran model dengan format/presisi dan cakupan state yang sebanding, waktu training, serta latensi inferensi terukur. Untuk latency, tetapkan perangkat dan precision yang sama, batch 1, warm-up inferensi, jumlah pengulangan, sinkronisasi penyelesaian operasi, serta cakupan waktu preprocessing. Ukur dengan input non-test terlebih dahulu jika masih merancang protokol benchmark; jangan memakai label test untuk keputusan tersebut.

Tampilkan selisih dengan arah dan satuan yang jelas: poin persentase untuk metrik klasifikasi, parameter, MiB, detik/menit training, dan milidetik per gambar. Kesimpulan tentang model yang lebih baik mengikuti hasil terukur dan batasan handoff, bukan asumsi bahwa salah satu arsitektur pasti unggul.

## 9. Hasil review strategi sebelumnya

- Pemeriksaan output model digabung dengan pembangunan model; compile dan callbacks berada dalam tahap training masing-masing.
- Urutan pemilihan lintas warm-up/fine-tuning dibuat eksplisit; hasil warm-up tetap bisa menang.
- Pipeline, augmentasi, dan normalisasi yang sudah tersedia dipakai kembali.
- Kebutuhan Drive/manifest disesuaikan dengan notebook bersama yang belum menyediakannya.
- Class weighting ditentukan dari distribusi train pada output Colab bersama.
- Klaim runtime, ukuran model, independensi test, dan kesiapan deployment dibatasi sesuai bukti yang tersedia.

Dokumen ini merencanakan implementasi berikutnya; tidak memuat kode fine-tuning dan tidak mengubah notebook bersama.
