# Laporan pemulihan LWDN

Tanggal pemeriksaan: 29 Oktober 2026. Sumber utama: kode, output, dan artefak yang dibaca melalui [notebook FGD-6](https://colab.research.google.com/drive/1CKksYUrcdfcZtOxGZq2EDlbh_3Jc2DOL). Pekerjaan ini memperbaiki pemulihan eksperimen, tanpa mengubah arsitektur, split, preprocessing, learning rate, atau batas epoch model.

## Masalah dan penyebabnya

Error yang terlihat pada L04 adalah `NameError: name 'lwdn_run_dir' is not defined`. Cell training membutuhkan beberapa variabel yang hanya hidup di memori runtime. Pemulihan sebelumnya baru mengembalikan handoff data dan baseline DenseNet-121. Konteks eksperimen LWDN belum dipulihkan secara lengkap.

Pemeriksaan berikutnya menemukan run aktif sudah berpindah ke folder baru yang hanya berisi konfigurasi. L02 membuat nama folder baru setiap dijalankan, sedangkan L03 membuat model dengan classifier baru. Menjalankan kedua cell itu kembali tidak memuat hasil training sebelumnya. Masalah yang terverifikasi berada pada state dan pemilihan folder run; hasil validation checkpoint lama cocok dengan pipeline yang dipulihkan.

| Run LWDN | Artefak yang ditemukan | Maknanya |
| --- | --- | --- |
| `20261028T151525_228962Z` | Konfigurasi, urutan kelas, `warmup.csv`, `warmup_best.keras`, `finetune.csv`, `finetune_best.keras`, dan `timing.json` | Run lama yang berisi hasil training dan dipakai untuk pemulihan |
| `20261028T223119_199177Z` | `config.json` dan `classes.json` | Folder baru tanpa checkpoint; tidak dipakai untuk melanjutkan training lama |

Warm-up lama selesai 10 epoch. Log fine-tuning berisi 10 epoch, tetapi checkpoint terbaik berasal dari epoch 8. Checkpoint itu menghasilkan validation accuracy **62,28%** dan loss **1,0660** saat diperiksa kembali pada 342 gambar validation. Ini hasil sementara, bukan hasil test atau hasil akhir perbandingan model.

Lingkungan pemulihan cocok dengan konfigurasi tersimpan: CPU, float32, TensorFlow 2.21.0, dan Keras 3.13.2. State optimizer checkpoint cocok dengan epoch 8, yaitu 1.000 langkah untuk 125 batch per epoch.

## Perubahan di notebook

1. Menambahkan [Pulihkan run LWDN](https://colab.research.google.com/drive/1CKksYUrcdfcZtOxGZq2EDlbh_3Jc2DOL#scrollTo=uq352gQaqzq0) dan cell kodenya setelah L01. Cell ini memakai ID run eksplisit, memeriksa kecocokan handoff, kelas, pengaturan, perangkat, serta versi library, lalu memuat bobot dan optimizer dari checkpoint. Folder terbaru tidak dipilih otomatis.
2. Menambahkan [Verifikasi pemulihan LWDN](https://colab.research.google.com/drive/1CKksYUrcdfcZtOxGZq2EDlbh_3Jc2DOL#scrollTo=jZVowHlQrNsJ). Pemeriksaan mencakup variabel yang diperlukan, file checkpoint, jumlah langkah optimizer, serta kecocokan loss dan accuracy dengan log. Cell ini tidak memperbarui bobot.
3. L02 mempertahankan run yang dipulihkan. Pembuatan eksperimen baru memerlukan flag `START_NEW_LWDN_RUN = True` pada konteks eksperimen baru.
4. L03 melewati pembangunan model baru setelah recovery. L04 melewati warm-up yang sudah selesai. L05 mempertahankan bobot dan optimizer fine-tuning yang sudah dipulihkan.
5. L06 memakai `initial_epoch` dari checkpoint. Untuk run lama ini, kelanjutan menggunakan `finetune_resume.csv` dan `finetune_resume_best.keras`, sehingga log dan checkpoint asli tetap tersedia. CSV memakai mode append dan checkpoint baru harus memperbaiki nilai terbaik sebelumnya.
6. L06 menambahkan `BackupAndRestore` pada folder khusus di Drive, dengan dua backup terakhir dan tanpa menghapus backup setelah selesai. Penanda selesai mencegah training yang sudah selesai dijalankan kembali.
7. L07 memasukkan checkpoint kelanjutan sebagai kandidat dan menolak pemilihan akhir sebelum fine-tuning tercatat selesai. Durasi total lama yang tidak dapat dipulihkan dicatat sebagai tidak tersedia, bukan diisi dengan perkiraan.

Peran checkpoint terbaik dan backup berbeda. Checkpoint terbaik dipakai untuk pemilihan model berdasarkan validation. Backup menyimpan state training agar pekerjaan yang terputus dapat dilanjutkan. Perilaku ini mengikuti dokumentasi [BackupAndRestore](https://keras.io/api/callbacks/backup_and_restore/), [ModelCheckpoint](https://keras.io/api/callbacks/model_checkpoint/), dan [CSVLogger](https://keras.io/api/callbacks/csv_logger/).

## Urutan setelah runtime reset

1. Hubungkan Google Drive melalui cell `FdzCf2API22w`.
2. Pada runtime baru yang belum memiliki folder data, jalankan `K0p5Y4vnJ-dW` dengan `HANDOFF_ACTION = 'restore'`. Jika data dan handoff masih aktif, langkah ini tidak perlu diulang. Jangan menghapus folder hanya untuk melewati pemeriksaan keamanan.
3. Jalankan TRANSFORMASI `e7xe5KJi0Xdf`, kemudian Data pipeline dan handoff `3HknRbVgJDqj`.
4. Pulihkan baseline melalui `dop6y8FwMXVc`, menggunakan `RECOVER_DENSENET_RUN = '20261028T104136_712764Z'`. Tidak perlu menjalankan cell training DenseNet-121.
5. Jalankan L01, kemudian **Pulihkan run LWDN**, dengan `RECOVER_LWDN_RUN = '20261028T151525_228962Z'`.
6. Jalankan **Verifikasi pemulihan LWDN**. Lanjutkan setelah muncul `Pemulihan LWDN LULUS`. Untuk melanjutkan run ini, L02–L05 tidak perlu dijalankan; pengaman juga akan melewatinya bila dijalankan.
7. Jalankan **L06** saat siap melanjutkan training. Batas akhirnya tetap 25 epoch dalam tahap fine-tuning, bukan 25 epoch tambahan. Setelah selesai, jalankan L07. Data test tetap disimpan untuk evaluasi akhir sesuai protokol penelitian.

Jangan memakai Run all untuk memulihkan eksperimen: cell preparation atau pembentukan run baru dapat ikut berjalan. Jangan mengganti perangkat, versi library, atau pengaturan compile/fit diam-diam ketika memakai backup.

## Bukti pemeriksaan

- Pemeriksaan sebelum perbaikan berhasil mendeteksi bahwa run aktif tidak memiliki checkpoint warm-up lama.
- Recovery memuat run lama dan checkpoint fine-tuning epoch 8, termasuk optimizer.
- Verifikasi pada seluruh validation menghasilkan accuracy 62,28% dan loss 1,0660, sesuai log checkpoint.
- L02–L05 dijalankan pada state hasil recovery dan menampilkan status memakai run lama atau melewati langkah; warm-up dan pembangunan model tidak diulang.
- Uji terisolasi memakai model kecil dan data sintetis: training sengaja diinterupsi setelah epoch pertama, kemudian dilanjutkan. Log akhir tetap berisi epoch 0, 1, 2; pemanggilan kedua hanya menjalankan epoch 1 dan 2; optimizer mencapai enam langkah. Pemanggilan ketiga tidak mengubah log atau jumlah langkah optimizer karena training sudah selesai. Model daun tidak dilatih dalam uji ini.
- Uji pemilihan checkpoint pada artefak sintetis memastikan tahap yang belum selesai ditolak, checkpoint kelanjutan dengan loss terendah benar-benar dimuat, dan durasi total yang tidak diketahui tetap kosong.
- Delapan source cell yang ditambahkan atau diubah lolos pemeriksaan sintaks Python. Cell verifikasi di notebook tetap dapat dijalankan kembali.

## Batas pemulihan yang perlu diketahui

Bobot epoch 9–10 pada run lama tidak tersedia. Pemulihan pertama memakai checkpoint terbaik epoch 8, sehingga sebagian pekerjaan setelah checkpoint tersebut harus dihitung ulang. Warm-up dan training DenseNet-121 tetap dapat dipakai kembali.

Backup baru disimpan pada akhir epoch. Gangguan di tengah epoch dapat membuat epoch tersebut diulang. Jika backup terbaru rusak, pemulihan dapat memakai backup sebelumnya. Mekanisme ini tidak mencegah Colab memutus atau menghapus runtime.

Backup tidak menjamin jalur training identik secara bit demi bit: state augmentasi acak dan hitungan tunggu EarlyStopping tidak dipulihkan oleh kode ini. Batas epoch dan pengaturan patience tetap sama. Durasi fine-tuning lama yang terputus tidak tersimpan lengkap, sehingga belum dapat dipakai untuk menghitung waktu training total yang akurat.

Pemeriksaan ini belum menjalankan kelanjutan training model daun, belum menyelesaikan L07, dan belum membuka test set. Langkah berikutnya adalah menjalankan L06 dari state yang sudah dipulihkan.
