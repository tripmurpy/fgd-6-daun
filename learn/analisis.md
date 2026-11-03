# Analisis Reverse Engineering Phase 4: Modeling

## Ruang lingkup

Notebook yang dianalisis adalah [`FGD5_YOLO_V8_V12.ipynb`](https://colab.research.google.com/drive/1ILlfadwl2sftcREvF8B_pRjHAQ3mvO6G#scrollTo=Xy8DvYbSk_5H). Phase 4 dimulai pada sel 27 dan berakhir sebelum Phase 5 pada sel 40. Di dalamnya ada empat sel kode utama: 29, 32, 35, dan 38. Sel lainnya menjadi judul atau penjelasan.

Notebook ini adalah proyek deteksi kendaraan di video, bukan klasifikasi daun herbal FGD-6. Modelnya YOLOv12n dan YOLOv8n, dengan empat kelas kendaraan. Analisis ini membaca kode dan output yang tersimpan di notebook; sel tidak dijalankan ulang.

> **Inti Phase 4:** mencari kombinasi pengaturan training untuk masing-masing model, lalu melatih satu model final per arsitektur menggunakan bobot pretrained. Genetic Algorithm (GA) dimaksudkan untuk mencari hyperparameter. GA tidak melatih bobot jaringan secara langsung; bobot tetap dilatih oleh proses training YOLO.

## Alur besarnya

```text
Data train + validation
        ↓
Uji beberapa kombinasi hyperparameter YOLOv12n (sel 29)
        ↓
Uji beberapa kombinasi hyperparameter YOLOv8n (sel 32)
        ↓
Ambil kombinasi dengan mAP50 tertinggi untuk tiap model
        ↓
Latih ulang YOLOv12n dan YOLOv8n masing-masing 50 epoch (sel 35 dan 38)
```

Data dari tahap sebelumnya yang dipakai oleh sel modeling adalah `data/dataset.yaml`. Output notebook menunjukkan 128 gambar untuk train, 32 gambar untuk validation, dan 45 objek pada validation. Tidak tampak pembagian test terpisah pada alur split tersebut.

## Analisis per sel

| Sel | Isi | Tujuan dan hasil |
|---|---|---|
| 27 | Judul Phase 4 | Menandai awal modeling. Tidak menjalankan kode. |
| 28 | Judul GA YOLOv12 | Membuka bagian pencarian pengaturan YOLOv12. |
| 29 | Kode pencarian YOLOv12 | Membuat kandidat hyperparameter, melatih tiap kandidat selama 12 epoch, menilai mAP50, lalu memilih kandidat yang skornya paling tinggi. Output tersimpan menunjukkan pencarian berjalan dan parameter terpilih sekitar `lr0=0.0063`, `batch=8`, `imgsz=480`. |
| 30 | Penjelasan kode 29 | Menguraikan fungsi evaluasi, populasi, dan pemilihan kandidat. Tidak menambah proses baru. |
| 31 | Judul GA YOLOv8 | Membuka bagian pencarian untuk model YOLOv8. |
| 32 | Kode pencarian YOLOv8 | Mengulangi pola sel 29 untuk YOLOv8. Output tersimpan menunjukkan parameter terpilih sekitar `lr0=0.0060`, `batch=16`, `imgsz=640`. |
| 33 | Penjelasan kode 32 | Menjelaskan fungsi evaluasi, populasi, dan pemilihan kandidat. Tidak menambah proses baru. |
| 34 | Judul training YOLOv12 | Menandai training akhir YOLOv12 setelah pencarian parameter. |
| 35 | Kode training akhir YOLOv12 | Mengambil parameter terpilih, memuat `yolo12n.pt`, lalu training selama 50 epoch. Output menunjukkan training selesai dan bobot `best.pt` serta `last.pt` dibuat di folder run. |
| 36 | Penjelasan kode 35 | Menjelaskan pemakaian parameter terpilih dan model pretrained. Tidak menambah proses baru. |
| 37 | Judul training YOLOv8 | Menandai training akhir YOLOv8. |
| 38 | Kode training akhir YOLOv8 | Mengambil parameter terpilih, memuat `yolov8n.pt`, lalu training selama 50 epoch. Output menunjukkan training selesai dan bobot `best.pt` serta `last.pt` dibuat di folder run. |
| 39 | Penjelasan kode 38 | Menjelaskan pemakaian parameter terpilih dan model pretrained. Tidak menambah proses baru. |

## Cara kerja pencarian parameter

Satu **individu** GA di notebook ini adalah satu paket berisi tiga pengaturan:

- `lr0`: learning rate, yaitu ukuran langkah saat bobot model diperbarui; dicoba dari 0.001 sampai 0.01.
- `batch`: jumlah gambar dalam satu batch; pilihannya 8 atau 16.
- `imgsz`: ukuran gambar masukan; pilihannya 480 atau 640.

Masing-masing kandidat dimuat ke YOLO pretrained (`yolo12n.pt` atau `yolov8n.pt`), kemudian dilatih selama 12 epoch. Setelah training, fungsi mengambil **mAP50** sebagai nilai fitness. Secara sederhana, mAP50 menilai ketepatan deteksi objek berdasarkan kotak deteksi pada ambang IoU 0.50. Skor lebih tinggi dianggap lebih baik. Jika metrik tidak ditemukan atau training gagal, fungsi mengembalikan skor kecil `0.0001`.

Kode membuat populasi awal berisi empat kandidat dan mengulang proses dua generasi. Secara hitungan, setiap model menjalankan sampai delapan training pendek (4 kandidat × 2 generasi), masing-masing 12 epoch. Setelah parameter dipilih, sel 35 atau 38 menjalankan training final 50 epoch. Jadi ada 146 epoch kerja per model yang tersebar di beberapa training terpisah, bukan satu training panjang.

### Mengapa memakai GA?

Dari nama fungsi, komentar, dan metrik fitness di kode, maksud penggunaan GA tampaknya untuk mencoba pengaturan secara otomatis ketika pengaruh gabungan learning rate, batch, dan ukuran gambar tidak mudah diketahui sebelumnya. Ide GA adalah mempertahankan kandidat yang skornya baik, menggabungkan kandidat (crossover), lalu memberi perubahan acak kecil (mutasi), kemudian menguji generasi baru. Nilai tiap kandidat berasal dari hasil training dan validasi, bukan tebakan manual.

### Hal penting: implementasi ini belum menjalankan evolusi GA

Sel 29 dan 32 memang mendaftarkan `mate` (crossover) dan `mutate` (mutasi) ke toolbox. Namun loop yang benar-benar dijalankan hanya:

1. mengevaluasi kandidat yang sudah ada;
2. memberi skor fitness;
3. menjalankan seleksi turnamen untuk membentuk populasi berikutnya.

Tidak ada pemanggilan `toolbox.mate(...)` atau `toolbox.mutate(...)`, juga tidak ada pembuatan offspring. Jadi generasi kedua hanya berisi hasil seleksi ulang dari kandidat awal—bisa dengan kandidat yang sama berulang kali—bukan kombinasi parameter baru. Dalam praktiknya, bagian ini lebih mirip pencarian acak kecil yang diikuti pengulangan kandidat daripada GA lengkap. `tools.selBest` mengambil yang terbaik dari populasi terakhir, bukan menyimpan catatan terbaik dari seluruh perjalanan.

Mutasi yang didaftarkan juga memakai Gaussian untuk ketiga nilai. Jika kelak benar-benar dipakai, batch dan ukuran gambar dapat berubah menjadi angka pecahan atau nilai yang tidak didukung kecuali dibulatkan dan dibatasi. Dalam kode sekarang, masalah itu belum muncul karena mutasi tidak dipanggil.

## Teknik training yang dipakai

Ini adalah **transfer learning/fine-tuning** dari bobot pretrained, bukan inisialisasi bobot secara acak. Setiap evaluasi kandidat membuat objek YOLO dari checkpoint pretrained lagi, lalu memanggil `model.train(...)`. Output Ultralytics juga mencatat bobot pretrained berhasil dipindahkan ke model untuk kelas dataset ini. Sel training akhir mengulang dari checkpoint pretrained dengan hyperparameter pilihan selama 50 epoch; bobot hasil training GA tidak diteruskan ke training final.

Beberapa rincian yang terlihat pada kode dan output:

- Training dijalankan pada GPU `device=0`; output mencatat Tesla T4 dengan VRAM sekitar 15 GB.
- Pencarian memakai `epochs=12`, sedangkan training akhir memakai `epochs=50`.
- Pengaturan lainnya sebagian besar mengikuti default Ultralytics, termasuk `optimizer=auto`, AMP, dan augmentasi bawaan.
- mAP50 menjadi satu-satunya skor fitness pencarian. Waktu inferensi atau skor mAP50-95 tidak ikut menentukan parameter terbaik.
- Output training akhir menilai checkpoint terbaik pada validation. Nilai yang tersimpan: YOLOv12 mAP50 `0.845` dan mAP50-95 `0.703`; YOLOv8 mAP50 `0.878` dan mAP50-95 `0.703`. Ini hasil validation pada notebook, bukan hasil test independen.

## Yang bisa dan belum bisa disimpulkan

Pola yang berguna untuk dipahami adalah pemisahan antara **pencarian konfigurasi** dan **training final**: tetapkan ruang pencarian, ukur kandidat pada validation, pilih parameter, kemudian latih model final. GA dimaksudkan untuk membantu pencarian itu, tetapi implementasi sel 29 dan 32 belum memiliki langkah crossover/mutasi yang membuat generasi baru.

Angka akhir tidak cukup untuk menyimpulkan model mana yang lebih unggul secara umum. Validation hanya berisi 32 gambar dan 45 objek; output per kelas bahkan menunjukkan hanya satu bus dan dua truk. Pencarian parameter dan pelaporan akhir memakai validation yang sama, sehingga skor tersebut juga sudah dipakai untuk memilih konfigurasi. Selain itu, dua model terpilih memakai ukuran gambar dan batch yang berbeda, jadi perbandingan ini mencakup perbedaan konfigurasi sekaligus arsitektur.

Output menyimpan run di `/content/runs/...` dan Phase 4 tidak menyalinnya ke Google Drive. Folder `/content` berada di runtime Colab, sehingga checkpoint berisiko hilang saat runtime di-reset jika tidak disalin atau diunduh.

Ada juga selisih pengaturan seed: sel awal notebook menetapkan seed Python/TensorFlow/PyTorch ke 42, tetapi argumen training Ultralytics yang tercatat di output menunjukkan `seed=0` (default), karena nilai `SEED` tidak diberikan ke `model.train`. Jadi seed 42 mengatur sebagian proses, tetapi belum menjadi seed training Ultralytics.

## Hubungannya dengan modeling FGD-6

Notebook ini dapat dipakai untuk belajar cara membaca alur modeling dan memahami tujuan pencarian hyperparameter. Namun proyek ini berbeda: FGD5 melakukan deteksi objek kendaraan dengan YOLO, sedangkan FGD-6 mengklasifikasikan jenis daun herbal. Karena itu, jangan menyalin langsung GA, parameter, mAP50, pembagian data, atau klaim performanya ke FGD-6.

Untuk pekerjaan FGD-6, ikuti alur yang sudah ditetapkan untuk dataset klasifikasi: mulai dari DenseNet-121, pilih checkpoint memakai validation, lanjutkan ke LWDN setelah spesifikasi arsitekturnya disepakati, lalu gunakan test set satu kali untuk laporan akhir. Metrik klasifikasi dan pelaporan kelas juga harus sesuai tugas daun, bukan metrik deteksi YOLO.
