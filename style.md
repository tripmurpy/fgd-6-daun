# Style Notebook Modeling FGD

## Sumber dan batas penggunaan

Gunakan [notebook Colab FGD5](https://colab.research.google.com/drive/1ILlfadwl2sftcREvF8B_pRjHAQ3mvO6G) sebagai acuan **susunan cell dan cara penyajian**. Pada Phase 4, FGD5 memakai judul fase Markdown, judul langkah singkat, lalu cell kode yang langsung mengerjakan langkah itu. Cell training final menjalankan training secara langsung, memberi komentar singkat pada parameter, dan membiarkan library menampilkan progres.

Panduan ini berlaku saat menulis cell fine-tuning DenseNet-121 di FGD-6. Terapkan gaya FGD5 pada penyajian; ikuti `AGENTS.md` dan `strategi1.md` untuk keputusan metode yang disetujui. Kode YOLO, algoritma genetika, data, dan parameternya tidak berlaku untuk DenseNet.

## Pola cell yang harus diikuti

1. Pertahankan judul Markdown CRISP-DM Phase 4 yang sudah ada.
2. Sebelum setiap langkah, tulis satu judul Markdown pendek dan, bila perlu, satu kalimat tujuan.
3. Letakkan satu pekerjaan yang jelas dalam setiap code cell. Awali dengan komentar singkat berbahasa Indonesia; beri komentar inline hanya untuk pilihan yang belum jelas dari kodenya.
4. Biarkan output training dari `model.fit()` menunjukkan progres epoch. Tampilkan hanya ringkasan penting yang belum tersedia dari output tersebut, seperti checkpoint terpilih dan `val_loss` terbaik.
5. Tambahkan penjelasan hasil singkat sesudah cell hanya jika output memerlukan konteks.

## Batas cell fine-tuning

Susun cell berurutan untuk:

1. Membuat DenseNet-121 pretrained dengan classifier 10 kelas dan membekukan backbone.
2. Melatih classifier saat warm-up.
3. Membuka blok akhir `conv5_`, membekukan BatchNormalization, dan compile ulang dengan optimizer baru.
4. Melakukan fine-tuning serta menyimpan checkpoint terbaik berdasarkan `val_loss`.
5. Memuat checkpoint dengan `val_loss` validation terendah dari kedua tahap dan menampilkan ringkasan pemilihannya.

Gunakan pipeline, variabel, preprocessing, mount Drive, dan dependency yang sudah tersedia di FGD-6. Setiap cell hanya boleh menambahkan langkah yang diperlukan untuk urutan fine-tuning di atas. Pertahankan pencatatan minimum yang diwajibkan `AGENTS.md`: seed, setelan training, runtime/perangkat, dan checkpoint.

## Hal yang tidak dimasukkan ke cell training

Jangan menaruh konfigurasi metadata bertingkat, hash dan manifest data, audit file ulang, setup pipeline yang sudah tersedia, tabel laporan lengkap, evaluasi test, perbandingan LWDN, benchmark latency, atau deployment di dalam rangkaian cell fine-tuning. Jangan menambah `print()` yang mengulang progres, metrik, atau lokasi yang sudah tampak dari output training dan checkpoint. Simpan evaluasi lengkap untuk tahap Evaluation yang sesuai.

## Selesai jika

Urutan warm-up → fine-tuning → pemilihan checkpoint terlihat jelas saat notebook dibaca dari atas ke bawah, setiap cell fokus pada satu langkah training, dan tidak ada setup atau laporan tambahan di luar batas panduan ini.
