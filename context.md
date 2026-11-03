# Konteks Proyek FGD 6

## Tujuan penelitian

Penelitian mengukur dan menganalisis seberapa besar perbedaan kinerja **DenseNet-121** dan **Lightweight DenseNet (LWDN)** pada klasifikasi citra daun tanaman obat/herbal. Fokusnya adalah perbandingan yang adil pada dataset dan kondisi evaluasi yang sama, lalu menjelaskan trade-off kualitas klasifikasi dan efisiensi komputasi. Jangan menganggap LWDN pasti menyamai DenseNet-121 atau pasti menjadi model terbaik; kesimpulan harus mengikuti hasil eksperimen.

## Ruang lingkup dan sumber acuan

- Ikuti CRISP-DM. Fokus pengerjaan saat ini adalah Modeling dan Evaluation; Deployment dilakukan setelah hasil dibandingkan dan tim memilih model.
- Dataset mentah penelitian adalah Indonesian Herb Leaf Dataset 3500: 3.500 citra pada 10 kelas (350 per kelas). Data cleaning dan preparation dimiliki engineer lain; gunakan handoff yang disepakati dan jangan mengulang atau mengubah langkahnya tanpa koordinasi.
- Lanjutkan notebook bersama **FGD-6 Google Colab** yang sudah dimulai tim. Notebook lokal di `source code/` adalah salinan kerja/referensi; jangan menganggap konfigurasi dan output lokal sebagai hasil eksperimen terkini.
- Pembagian data, preprocessing, augmentasi, seed, dan kondisi runtime harus mengikuti handoff serta konfigurasi notebook yang disepakati. Catat perubahan yang disetujui. Jangan menetapkan ulang split atau membuka test set untuk pemilihan model.
- Paper LWDN asli kini menjadi sumber kandidat untuk review bersama. Adaptasi arsitektur dan setelan eksperimen tetap memerlukan kesepakatan user/tim sebelum implementasi; jangan mengarang desain, pruning, quantization, atau format ekspor.

## Audit notebook tim (snapshot 23 Oktober 2026)

Notebook bersama `FGD-6.ipynb` memiliki 25 sel: 13 sel kode dan 12 sel Markdown. Alur yang sudah ada mencakup setup/konfigurasi (sel 3, 5), inventaris data (6), distribusi dan sampel (8–10), pemeriksaan kualitas (11, 13), cleansing (16), split dan standardisasi (18), definisi augmentasi (20), pipeline input (22), serta validasi (24). Jadi, Colab tim saat ini berakhir di Data Preparation; belum ada sel modeling atau evaluation.

Salinan lokal `source code/FGD_6.ipynb` berbeda: 30 sel, 21 kode, dengan Drive mount dan sel DenseNet-121 tambahan (23–29) untuk warm-up, fine-tuning, validasi/checkpoint, dan ringkasan kurva/artefak. Sel tersebut tidak ada di snapshot Colab tim di atas. Setelan di salinan lokal—10 epoch warm-up, batas 25 epoch fine-tuning, learning rate 1e-3/1e-5, dropout 0,3, membuka `conv5_`, membekukan BatchNorm, mixed precision berbasis GPU, dan satu seed—masih menunggu persetujuan tim; jangan anggap konfigurasi final. Salinan lokal belum memuat evaluasi test berpasangan atau LWDN. Telaah dan rekonsiliasi sel yang ada sebelum menambah fungsi serupa; jangan menyalin atau menulis ulang tanpa memeriksa versi Colab terkini.

Output tersimpan pada snapshot menunjukkan 3.500 citra mentah menjadi 3.414 setelah 78 duplikat dan 8 kandidat salah label dikeluarkan. Kode cleansing otomatis menerapkan penghapusan kandidat ketika proporsinya di bawah batas 3%; kandidat tersebut belum tentu benar-benar salah label. Split tercatat 2.731 train, 341 validation, dan 342 test (80/10/10); jumlah citra per kelas setelah cleansing tidak lagi 350. Sel validasi melaporkan nol error/peringatan, tanpa hash MD5 lintas split, dan bentuk input 224×224×3. Namun execution_count sel-sel tersebut kosong dan notebook bersama terus berubah; periksa ulang handoff/output terkini di Colab sebelum menjadikan angka ini dasar eksperimen. Sel 18 menghapus dan membangun ulang `PREP_DIR`; jangan rerun tanpa memastikan konsekuensi dan handoff-nya.

Notebook contoh FGD5 adalah proyek deteksi/pelacakan kendaraan YOLO. Yang relevan hanya susunan fase, tujuan sel yang jelas, konfigurasi, catatan singkat, serta pemeriksaan output. Jangan menyalin kode, model, preprocessing, algoritma pencarian, metrik, dependency, atau langkah deployment-nya ke proyek klasifikasi daun.

## Review sumber arsitektur LWDN

Pengguna meminta paper LWDN asli ditinjau bersama sebelum implementasi. Paper *LWDN: lightweight DenseNet model for plant disease diagnosis* (Springer, 2024) mendeskripsikan desain berbasis DenseNet121 yang dipangkas, dengan dua cabang LWDN dan penggabungan fitur. Ringkasan paper melaporkan sekitar 1,52 juta parameter, ukuran 13,8 MB, dan akurasi 99,37% pada PlantVillage dengan 39 kelas penyakit; angka tersebut adalah hasil paper pada dataset/protokol berbeda, bukan target atau hasil yang dapat dipindahkan ke klasifikasi 10 jenis daun herbal kita. Paper juga menggunakan transfer learning dan pembekuan sebagian cabang/lapisan.

Sumber untuk diskusi: [halaman resmi Springer](https://link.springer.com/article/10.1007/s41348-024-00915-z) dan [naskah yang dapat dibaca di ResearchGate](https://www.researchgate.net/publication/379758606_LWDN_lightweight_DenseNet_model_for_plant_disease_diagnosis). Sebelum menulis LWDN, tim perlu menyepakati apakah mereplikasi penuh dua cabang dan feature fusion paper dengan head 10 kelas serta bagaimana mengadaptasi preprocessing, transfer learning, freezing, dan hyperparameter ke eksperimen kita. Jangan mengasumsikan pemangkasan tunggal, backbone satu cabang, atau menyamakan preprocessing DenseNet dan LWDN tanpa keputusan metodologis.

## Status konfigurasi DenseNet-121

Konfigurasi DenseNet di notebook lokal (`10` warm-up epoch, hingga `25` fine-tuning epoch, learning rate `1e-3`/`1e-5`, dropout `0,3`, membuka `conv5_`, BatchNorm dibekukan, mixed precision berdasarkan GPU, satu seed) hanya draft. Pengguna belum menyetujuinya. Tinjau dan diskusikan setelan tersebut sebelum menjadikannya konfigurasi eksperimen atau menulis sel yang bergantung padanya.
## Perbandingan dan pelaporan

Latih dan evaluasi kedua model pada partisi data yang sama dengan kondisi yang sebanding. Gunakan validation untuk pemilihan model dan test hanya untuk evaluasi akhir.

Laporkan accuracy, precision, recall, macro F1-score, confusion matrix, jumlah parameter, ukuran model, waktu pelatihan, dan latensi inferensi. Tunjukkan selisih DenseNet-121 dan LWDN secara eksplisit untuk tiap metrik yang dibandingkan; nyatakan arah selisih dan satuannya (misalnya poin persentase untuk metrik persentase, milidetik untuk latensi). Bahas kelas/pasangan kelas yang paling berkontribusi pada perbedaan melalui metrik per kelas dan confusion matrix. Jelaskan juga kondisi pengukuran agar klaim efisiensi dapat ditafsirkan.

Simpan seed, konfigurasi model, preprocessing, checkpoint terpilih, versi/runtime Colab, perangkat, serta hasil yang diperlukan untuk reproduksi. Hasil yang dilaporkan harus berasal dari output notebook yang benar-benar dijalankan; jangan mengisi hasil atau target keberhasilan sebelum pengukuran.

## Rencana fase modeling dan evaluation

1. Konfirmasi handoff data, rekonsiliasi sel DenseNet lokal dengan Colab, lalu cek validasi yang sudah ada. Jangan jalankan ulang cleaning/split/preprocessing kecuali diminta pemilik data.
2. Fine-tune DenseNet-121 memakai implementasi yang sudah ditinjau; gunakan validation untuk pemilihan checkpoint dan catat konfigurasi serta hasil tahap ini.
3. Setelah menyepakati adaptasi paper LWDN dan protokol latihnya, fine-tune dan pilih checkpoint dengan split serta kondisi yang sebanding.
4. Evaluasi checkpoint terpilih dari kedua model pada test set secara berpasangan satu kali, lalu bandingkan selisih metrik dan biaya komputasi.

Untuk setiap increment: satu Markdown singkat menjelaskan tujuan dan keluaran yang diharapkan; setiap code cell punya satu tujuan; periksa variabel dan keluaran yang sudah tersedia agar tidak menduplikasi pekerjaan. Tulis kode profesional, efisien, langsung, dengan Python empat spasi dan `snake_case`. Hindari fitur/abstraksi/dependency yang tidak diperlukan. Bila pilihan penting belum ditentukan (terutama arsitektur LWDN atau protokol evaluasi), tanyakan sebelum menulis kode yang bergantung padanya.

## Deployment

Prototipe Streamlit di `streamlit/` belum menjalankan inferensi. Tunda sambungan model sampai perbandingan selesai dan tim memilih model. Jangan menyatakan prediksi, akurasi aplikasi, atau kesiapan deployment berdasarkan UI prototipe saja.


