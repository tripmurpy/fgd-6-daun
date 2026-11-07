# FGD 6 — Analisis Perbedaan Kinerja DenseNet121 dan Lightweight DenseNet (LWDN) pada Klasifikasi Citra Daun Tanaman Obat

Naskah presentasi untuk CRISP-DM Phase 4 dan 5.

Acuan isi: [notebook FGD-6 di Google Colab](https://colab.research.google.com/drive/141G7KbWddlKUCPQK8FyllATNzXQpzJww#scrollTo=Qvwgx-J0SGH9).

# CRISP-DM Phase 4 — Modeling

**Script pembuka**

“Setelah data siap, kita masuk ke Phase 4 untuk membangun dan melatih model. Di sini kita membandingkan DenseNet-121 sebagai model acuan dengan LWDN sebagai kandidat yang lebih kecil, untuk mengenali 10 jenis daun. Keduanya memakai data train yang sama, gambar berukuran 224 × 224, dan alur pelatihan dua tahap.

Model memulai dengan bekal dari ImageNet, yaitu kumpulan gambar yang sebelumnya dipakai untuk melatihnya mengenali berbagai bentuk. Transfer learning berarti kita memakai bekal itu lalu menyesuaikannya untuk mengenali 10 jenis daun kita.

Di tahap pertama, bagian utama model kita bekukan dan kita latih bagian akhir yang bertugas memilih nama kelas daun. Tahap ini berjalan paling lama 15 epoch. Setelah itu, kita lanjut dari hasil terbaik tahap pertama. Di tahap kedua, kita buka sebagian bagian akhir model supaya ciri yang sudah dipelajari bisa disesuaikan lagi dengan daun. Penyesuaiannya dibuat lebih kecil dan lebih hati-hati; tahap ini berjalan paling lama 30 epoch (batas tahap kedua dibuat lebih panjang karena sebagian bagian utama model ikut disesuaikan, sedangkan tahap pertama hanya melatih bagian akhir. Notebook menetapkan angka 15 dan 30, tetapi tidak mencatat alasan khusus pemilihan angkanya). Pada kedua tahap, hasil terbaik dipilih dari data validation. Kalau nilainya tidak membaik selama 4 epoch, pelatihan berhenti dan bobot terbaik dipakai kembali.

Validation kita gunakan untuk memantau pelatihan dan memilih checkpoint. Data test kita simpan untuk evaluasi akhir di Phase 5, jadi hasil test tidak ikut menentukan model mana yang dipilih. Walaupun alur kedua model sama, bagian model yang dibuka saat tahap kedua mengikuti arsitektur masing-masing.”

## 4.1 — Arsitektur DenseNet-121 dan LWDN

**Script untuk menjelaskan kode**

“Pertama, kode membuat fungsi untuk menyiapkan model yang kita pilih. DenseNet-121 memakai bekal ImageNet dan bagian keluaran lamanya dilepas karena sekarang model perlu memilih dari 10 jenis daun. Untuk LWDN, kode mengambil tiga blok awal DenseNet-121 dan melepas blok terakhir. Kode juga memeriksa susunan blok yang tersisa agar bentuk modelnya sesuai dengan rancangan yang sedang dipakai.

Setelah itu, kedua model disiapkan untuk menerima gambar RGB berukuran 224 × 224. Penyesuaian nilai gambar dilakukan di dalam model. Bagian utama model dibekukan dulu, lalu ditambahkan bagian akhir untuk memilih satu dari 10 kelas daun. Di cell ini model baru disiapkan dan diperiksa; pelatihannya dilakukan setelahnya.

Sebelum menghitung parameter, kode mencoba model dengan empat gambar validation. Pemeriksaan ini memastikan model menghasilkan sepuluh skor kelas yang valid dan jumlah skornya satu. Ini hanya pengecekan bahwa alur model berjalan, bukan pengukuran akurasi.

Terakhir, kode menghitung jumlah parameter keseluruhan, bagian utama model, dan bagian yang akan dilatih di tiap tahap. Anggap parameter sebagai nilai-nilai yang dipelajari model dari gambar; angka ini bukan jumlah foto.”

**Script untuk menjelaskan output**

“Bisa kita lihat, DenseNet-121 memiliki 7.047.754 parameter secara keseluruhan, dengan 7.037.504 di bagian utamanya. Pada tahap pertama, yang dilatih hanya 10.250 parameter di bagian pemilih kelas. Saat tahap kedua, jumlah parameter yang dapat dilatih menjadi 2.140.170; dari bagian utama model, 30,3% dibuka untuk penyesuaian.

Pada LWDN, jumlah keseluruhannya 4.333.130 parameter. Artinya, jumlah parameter LWDN 38,52% lebih sedikit daripada DenseNet-121. Bagian utama LWDN memiliki 4.322.880 parameter. Tahap pertama tetap melatih 10.250 parameter di bagian pemilih kelas. Saat tahap kedua, 2.811.914 parameter dapat dilatih, dan 64,8% dari bagian utamanya dibuka.

Jadi, LWDN memang memiliki jumlah parameter keseluruhan yang lebih sedikit. Namun, persentase bagian utama yang dibuka saat penyesuaian justru lebih besar daripada DenseNet-121.

![Screenshot output tabel parameter pada bagian 4.1](phase4-4_1-output.png)

## 4.2 — Prosedur training bersama

**Script untuk menjelaskan kode**

“Di bagian sebelumnya, aku sudah menjelaskan bentuk model DenseNet-121 dan LWDN. Sebelum mulai melatihnya, cell ini memastikan pemeriksaan mutu data di bagian 3.8 sudah lulus. Setelah itu, model memakai data train untuk belajar dan data validation untuk mengecek perkembangannya. Persiapan ini nyambung dengan 4.1 karena cell ini memakai model yang sudah dirancang di sana, serta data yang sudah disiapkan sebelumnya. 

Aku menjalankan dua tahap yang sama untuk kedua model. Di tahap pertama, bagian utama model dibekukan, jadi yang belajar hanya bagian akhir yang memilih nama kelas daun. Batasnya 15 epoch dengan learning rate 0,001. 

Setelah tahap pertama, kode memuat bobot dari hasil terbaik, lalu membuka sebagian bagian utama model untuk penyesuaian tahap kedua. Lapisan BatchNormalization tetap dibekukan. Tahap ini memakai learning rate yang lebih kecil, yaitu 0,00001, dengan batas maksimal 30 epoch. Selama kedua tahap, validation dipakai untuk memilih bobot terbaik. Kalau hasil validation tidak membaik selama empat epoch, pelatihan berhenti dan bobot terbaik dipulihkan. Jadi, 15 dan 30 adalah batas maksimal, bukan jumlah epoch yang harus selalu dijalankan. Data test disimpan untuk evaluasi akhir di Phase 5.”

**Script untuk menjelaskan output**

“Sekarang aku jelaskan hasil pelatihan DenseNet-121 dulu. Pada tahap pertama, bobot terbaik ada di epoch 13 dengan akurasi validation 93,18%. Di tahap kedua, pelatihan berhenti pada epoch 5 karena hasil validation tidak membaik selama empat epoch. Bobot terbaik tahap kedua berasal dari epoch 1, dengan akurasi validation 93,77%. Jadi, pada hasil ini akurasinya naik 0,59 poin persentase dari tahap pertama.

Untuk LWDN, tahap pertamanya juga mendapat bobot terbaik di epoch 13, dengan akurasi validation 78,93%. Di tahap kedua, pelatihan berhenti pada epoch 20 dan bobot terbaik tahap kedua berasal dari epoch 16. Akurasi validation terbaiknya menjadi 95,25%, atau naik 16,32 poin persentase dari tahap pertama.

Angka-angka ini berasal dari data validation yang dipakai untuk memantau pelatihan. Jadi, ini belum hasil evaluasi akhir pada data test. Hasil test kedua model baru dibandingkan di Phase 5.”

“Nanti di bagian 4.3 aku jelaskan kenapa checkpoint terbaik DenseNet-121 tercatat di epoch 1, sementara LWDN di epoch 16, walaupun proses latihannya berhenti di epoch 5 dan 20.”

![Screenshot output pelatihan DenseNet-121 dan LWDN pada bagian 4.2](phase4-4_2-output.png)

## 4.3 — Pemilihan checkpoint melalui validation

**Script untuk menjelaskan aturan dan kode**

“Di bagian ini, aku memilih satu hasil terbaik untuk setiap model berdasarkan data validation. Kode membuka kembali checkpoint dari tahap pertama dan kedua, lalu mengecek hasil prediksinya pada gambar validation. Hasil akurasinya dicocokkan lagi dengan catatan training agar angkanya sesuai. Setelah itu, kode memilih checkpoint dengan validation accuracy tertinggi. Kalau hasilnya sama, yang dipilih adalah checkpoint dari tahap pertama, atau warm-up.

Kode juga menghitung macro F1 untuk merangkum seberapa seimbang model mengenali tiap jenis daun. Lalu, semua hasil dimasukkan ke tabel dan dibuat grafik accuracy serta loss dari awal sampai akhir training. Data test tidak dipakai untuk memilih checkpoint; evaluasi test tetap dilakukan di Phase 5.”

**Script untuk menjelaskan output**

“Sekarang kita lihat bagaimana kedua model berkembang selama dilatih. Warm-up adalah tahap awal, saat model belajar memilih jenis daun. Setelah itu, fine-tuning menjadi tahap penyesuaian lanjutan. Accuracy menunjukkan seberapa tepat model mengenali gambar.

Pada DenseNet-121, hasil awal di data train sekitar pertengahan 40 persen, sementara di validation sekitar 70 persen. Keduanya naik cukup cepat, lalu menjadi lebih berdekatan dan stabil di atas 90 persen. Saat masuk fine-tuning, hasilnya tetap stabil dan hanya berubah sedikit. Nilai validation terbaiknya, 93,77%, muncul pada epoch pertama fine-tuning. Kita tetap melanjutkan pelatihan sampai epoch 5 untuk melihat apakah hasilnya bisa membaik. Karena empat epoch berikutnya tidak memberi peningkatan, kita memakai hasil terbaik dari epoch pertama.

LWDN menunjukkan pola yang berbeda di tahap awal. Hasil train mulai sekitar 22 persen dan validation sekitar 46 persen. Keduanya meningkat, tetapi hasil validation lebih naik-turun daripada DenseNet-121. Setelah masuk fine-tuning, hasil kedua bagian naik tajam dan kemudian bergerak di sekitar 90 persen, dengan beberapa perubahan kecil. Nilai validation terbaiknya mencapai 95,25% pada epoch ke-16 fine-tuning. Pelatihan berlanjut sampai epoch 20 untuk menunggu kemungkinan perbaikan. Karena empat epoch setelah epoch terbaik tidak meningkatkan hasil, kita menggunakan hasil dari epoch ke-16.

Jadi, dari visualisasi ini kita melihat DenseNet-121 lebih cepat mencapai hasil yang stabil, sedangkan LWDN meningkat lebih besar saat memasuki fine-tuning. Pada kedua model, kita memilih hasil terbaik yang tercatat selama proses pelatihan.”

Tabel di atas grafik merangkum pilihan itu. DenseNet-121 mendapat macro F1 93,72% dan total waktu training dua tahap 5,66 menit. LWDN mendapat macro F1 95,30% dan waktu 8 menit. Jadi, pada data validation ini, nilai accuracy LWDN lebih tinggi 1,48 poin persentase dan macro F1 lebih tinggi 1,58 poin, dengan waktu training 2,34 menit lebih lama. Ini membantu aku memilih checkpoint untuk evaluasi berikutnya; kesimpulan akhir tetap menunggu hasil pada data test di Phase 5.”

![Screenshot tabel dan grafik pemilihan checkpoint pada bagian 4.3](phase4-4_3-output.png)

# CRISP-DM Phase 5 — Evaluation

**Script pembuka**

“Setelah kita melatih kedua model dan memilih checkpoint terbaiknya di Phase 4, sekarang kita lanjut ke evaluasi akhir. Di Phase 4, data validation membantu kita memantau latihan dan memilih checkpoint. Di Phase 5, kita memakai data test yang sejak awal disisihkan dan belum dipakai untuk melatih atau memilih checkpoint. Tujuannya untuk melihat seberapa baik kedua model mengenali foto daun yang belum mereka gunakan selama proses tersebut.”

## 5.1 — Evaluasi test dan perbandingan dua model

**Script untuk menjelaskan kode**

“Di cell ini, kode memastikan checkpoint kedua model sudah tersedia, lalu membuka masing-masing model untuk mencoba gambar yang sama dari data test. Setelah setiap tebakan dibandingkan dengan nama daun yang sebenarnya, kode merangkum berapa banyak tebakan yang benar, hasil untuk setiap jenis daun, dan jenis kesalahan yang muncul. Kode juga membuat confusion matrix agar kita bisa melihat daun yang tertukar.

Untuk menunjukkan perkiraan rentang hasilnya, kode mengambil ulang kelompok foto berkali-kali. Itulah asal kisaran IK 95% yang muncul pada output. Di tahap ini model hanya dinilai; bobotnya tidak dilatih atau diubah lagi.”

**Pembuka singkat evaluasi test**

“Sekarang kita lihat hasil DenseNet-121 pada 337 foto test yang disisihkan.”

**Script untuk menjelaskan output evaluasi test**

“Pada output ini, DenseNet-121 berhasil menebak dengan benar 314 dari 337 foto. Jadi, akurasi test-nya 93,18%, sementara 23 foto lainnya belum dikenali dengan benar. Sebelumnya, akurasi validation-nya 93,77%; hasil test ini sedikit lebih rendah, tetapi masih berada di kisaran yang sama.

Kisaran IK 95% pada output, yaitu 90,50% sampai 95,85%, memberi gambaran bahwa hasil dapat sedikit berubah jika kelompok foto yang diuji berbeda. Test loss-nya 0,2027 adalah ukuran lain untuk melihat kesalahan tebakan model; angka ini bukan persentase.

Di tabel per jenis daun, support menunjukkan jumlah foto sebenarnya untuk kelas itu. Kemangi memiliki support 34 foto. Recall-nya 79,41%, artinya sekitar 27 dari 34 foto Kemangi berhasil dikenali. Ini recall terendah di antara kelas daun, jadi Kemangi paling sering terlewat. Precision Kemangi 93,10% berarti sebagian besar tebakan “Kemangi” memang benar. F1-nya 85,71%, yaitu ringkasan keseimbangan precision dan recall. Di confusion matrix, kita juga melihat 6 foto Kemangi ditebak sebagai Sirih.

Sirih memiliki support 34 foto, recall 85,29%, precision 76,32%, dan F1 80,56%—F1 terendah di antara kelas daun. Artinya, model masih mengenali sebagian besar foto Sirih, tetapi dari semua foto yang ditebak sebagai Sirih, hanya sekitar tiga per empat yang benar. Salah satu kekeliruan yang tampak di confusion matrix adalah 4 foto Sirih ditebak sebagai Nangka.

Sebaliknya, ada tiga jenis daun yang seluruh fotonya dikenali dengan benar: Lidah Buaya (35 foto), Pandan (34 foto), dan Seledri (34 foto). Ketiganya memiliki recall dan F1 sebesar 100%. Jadi, pada data test ini, ketiga kelas tersebut menjadi contoh hasil pengenalan yang paling tepat.

Untuk macro average, kita menghitung rata-rata dengan memberi bobot yang sama pada tiap jenis daun; macro F1 DenseNet-121 adalah 93,15%. Weighted average juga menghitung rata-rata, tetapi memberi bobot lebih besar pada jenis daun yang jumlah fotonya lebih banyak; weighted F1-nya 93,16%. Karena kedua nilai F1 hampir sama, ringkasan hasilnya tidak banyak berubah dengan kedua cara hitung ini. Notebook menjalankan evaluasi yang sama pada LWDN dengan 337 foto test yang sama; hasil tersimpan menunjukkan 309 tebakan benar atau akurasi 91,69%, dengan macro F1 91,75%.”

![Screenshot output evaluasi test DenseNet-121](phase5-5_1-densenet-report.png)

**Script untuk menjelaskan confusion matrix test**

“Sekarang kita lihat confusion matrix DenseNet-121. Kotak hijau pekat di garis diagonal menunjukkan foto yang berhasil ditebak sesuai jenis daunnya; semakin pekat warnanya, semakin banyak foto yang masuk ke hitungan itu. Kotak di luar garis diagonal menunjukkan foto yang tertukar. Baris menunjukkan jenis daun sebenarnya, sedangkan kolom menunjukkan tebakan model.

Di sini kita fokus pada dua kesalahan yang terlihat mulai dari 34 foto berlabel Kemangi, model densenet 121 menebak 6 foto sebagai Sirih;

dan juga seperti yang bisa kita lihat , ada gambar sebanya 34 berlabel sirih yang menebak nangka sebanyak 4 kali 

Kenapa bisa begitu? Kemungkinan, pada beberapa foto, bentuk dan pola tulang daun Kemangi terlihat mirip dengan Sirih. Tekstur daun, pencahayaan, atau kualitas foto juga bisa membuat ciri visualnya tampak serupa sehingga model keliru mengenalinya. Ini masih dugaan, karena confusion matrix hanya menunjukkan jumlah dan arah kesalahan, bukan penyebabnya.”


![Confusion matrix test DenseNet-121](phase5-5_1-densenet-confusion.png)

**Script untuk menjelaskan output evaluasi test LWDN**

“Setelah kita melihat hasil DenseNet-121, sekarang kita lanjut ke LWDN. Keduanya diuji dengan 337 foto test yang sama, jadi kita bisa membandingkan hasilnya dari kumpulan foto yang setara.

LWDN menebak benar 309 dari 337 foto, sehingga akurasi test-nya 91,69%; 28 foto lainnya belum dikenali dengan benar. Akurasi validation sebelumnya 95,25%, jadi hasil pada test lebih rendah 3,56 poin persentase. Kisaran IK 95% pada output adalah 88,72% sampai 94,37%. Test loss-nya 0,2435, yaitu ukuran kesalahan lain dan bukan persentase.

Angka akurasi ini memberi gambaran umum. Berikutnya kita lihat tabel per daun untuk tahu jenis mana yang paling mudah dan paling sulit dikenali. Recall menunjukkan berapa banyak foto dari jenis daun sebenarnya yang berhasil dikenali. Precision menunjukkan seberapa sering tebakan untuk suatu jenis daun benar, sedangkan F1 merangkum keduanya.

Kalau kita melihat berapa banyak foto asli tiap kelas yang berhasil dikenali, Nangka, Pandan, dan Seledri sama-sama mencapai recall 100%. Dari sisi keseimbangan semua ukuran, Seledri paling konsisten: precision, recall, dan F1-nya 100%, dengan 34 dari 34 foto dikenali benar. Pandan memiliki precision 97,14%, recall 100%, dan F1 98,55%. Lidah Buaya seri dengan Pandan pada F1 98,55%, dengan precision 100% dan recall 97,14%. Nangka juga mengenali seluruh foto Nangka, tetapi precision-nya 79,07% dan F1-nya 88,31%, karena beberapa foto dari daun lain ikut ditebak sebagai Nangka.

Di sisi yang lebih sulit, F1 Sirih paling rendah, yaitu 78,87%. Recall-nya 82,35%, artinya sekitar 28 dari 34 foto Sirih dikenali benar; precision-nya 75,68%, jadi beberapa tebakan berlabel Sirih sebenarnya berasal dari jenis daun lain. Ini sejalan dengan confusion matrix yang menunjukkan ada foto Kemangi yang tertukar sebagai Sirih.

Pada baris macro average, setiap jenis daun diberi bobot yang sama: precision 92,29%, recall 91,66%, dan F1 91,75%. Weighted average memperhitungkan jumlah foto tiap jenis daun: precision 92,23%, recall 91,69%, dan F1 91,75%. Karena jumlah foto per kelas pada data ini hampir sama, kedua ringkasan tersebut juga berdekatan.”

![Screenshot output evaluasi test LWDN](phase5-5_1-lwdn-report.png)

**Script untuk menjelaskan confusion matrix test LWDN**

“Di confusion matrix LWDN, kotak hijau pekat pada garis diagonal menunjukkan foto yang dikenali sesuai labelnya; kotak di luar diagonal menunjukkan foto yang tertukar. Baris menunjukkan daun sebenarnya dan kolom menunjukkan tebakan model.

Kesalahan terbanyak dalam satu pasangan adalah 4 foto Kemangi yang diprediksi sebagai Sirih. Pada DenseNet-121, pasangan ini muncul 6 kali, jadi kedua model sama-sama sesekali tertukar pada pasangan daun ini. Kemungkinan, beberapa foto Kemangi memiliki tampilan yang mirip dengan Sirih. Namun, itu masih dugaan; confusion matrix menunjukkan jumlah dan arah salah tebak, bukan penyebab visualnya.”

![Confusion matrix test LWDN](phase5-5_1-lwdn-confusion.png)

## 5.1.1 — Perbandingan hasil kedua model

**Script untuk menjelaskan kode**

“Setelah kita melihat hasil DenseNet-121 dan LWDN satu per satu, sekarang kita rangkum perbandingannya di bagian 5.1.1. Cell ini tidak melatih ulang model. Kode memakai hasil prediksi dan catatan evaluasi test yang sudah dibuat sebelumnya, lalu memastikan kedua model sudah punya hasil sebelum perbandingan dijalankan.

Pertama, kode menyusun tabel yang membandingkan akurasi, nilai F1, jumlah parameter, dan lama training. Untuk memberi gambaran seberapa besar selisihnya bisa berubah, kode juga menghitung rentang perkiraan akurasi dan macro F1 dengan mengambil ulang kelompok foto test. Setelah itu, kode membandingkan tebakan kedua model pada foto test yang sama: berapa foto yang benar ditebak keduanya, hanya benar oleh DenseNet-121, hanya benar oleh LWDN, atau salah oleh keduanya.

Terakhir, kode mencari sampai tiga pasangan jenis daun dan tebakan yang paling sering tertukar pada masing-masing model. Kode menghitung jumlah serta persentasenya dari foto kelas sebenarnya, lalu menampilkan satu contoh foto untuk setiap pasangan. Jadi, bagian ini membantu kita melihat gambaran perbandingan sekaligus contoh kesalahan yang terjadi.”

**Script untuk menjelaskan tabel perbandingan**

“Di tabel paling atas, kita membandingkan kedua model pada 337 foto test yang sama. DenseNet-121 menebak benar 314 foto atau 93,18%, sedangkan LWDN menebak benar 309 foto atau 91,69%. Jadi, pada pengujian ini DenseNet-121 mendapat lima tebakan benar lebih banyak, dengan selisih akurasi 1,48 poin persentase.

Kolom `ci95_low` dan `ci95_high` menunjukkan kisaran perkiraan akurasi: 90,50%–95,85% untuk DenseNet-121 dan 88,72%–94,37% untuk LWDN. Macro F1 memberi bobot yang sama pada tiap jenis daun, sedangkan weighted F1 ikut menghitung jumlah fotonya. Nilai keduanya hampir sama: 93,15% dan 93,16% untuk DenseNet-121, serta 91,75% untuk LWDN. Test loss juga lebih rendah pada DenseNet-121 (0,2027) dibanding LWDN (0,2435); angka ini menunjukkan tingkat kesalahan, bukan persentase.

Tabel selisih menunjukkan hasil LWDN dikurangi DenseNet-121. Angka negatif berarti hasil LWDN lebih rendah: selisih akurasi −1,48 poin dan macro F1 −1,39 poin. Rentang perkiraannya masih mencakup kemungkinan hasil lebih tinggi maupun lebih rendah, jadi dari hasil ini saja kita belum bisa memastikan satu model pasti lebih unggul.”

Jumlah parameter DenseNet-121 adalah 7.047.754, sedangkan LWDN 4.333.130. Jadi, LWDN memiliki parameter lebih sedikit. Namun, waktu training yang tercatat pada run ini sekitar 339,5 detik atau 5,66 menit untuk DenseNet-121, dan 479,7 detik atau 8 menit untuk LWDN. Pada run ini, LWDN membutuhkan waktu training lebih lama meskipun jumlah parameternya lebih sedikit.

Di ringkasan berpasangan, 299 foto ditebak benar oleh kedua model. Ada 15 foto yang hanya benar oleh DenseNet-121, 10 foto yang hanya benar oleh LWDN, dan 13 foto yang salah ditebak keduanya. Jadi, sebagian besar tebakan kedua model sama-sama benar. Pada foto yang hasilnya berbeda, DenseNet-121 benar lima kali lebih banyak.”

**Script untuk menjelaskan tabel salah tebak DenseNet-121**

“Sekarang kita lihat tabel pasangan salah tebak DenseNet-121. Kolom label menunjukkan daun sebenarnya, sedangkan kolom prediksi menunjukkan tebakan model. Kolom contoh menampilkan nama satu foto yang mewakili pasangan itu, sementara jumlah citra kelas menunjukkan banyaknya foto test dari jenis daun tersebut. Angka jumlah dan persentase menunjukkan seberapa sering pasangan itu muncul di antara foto dari jenis daun sebenarnya.

Kesalahan yang paling sering adalah 6 dari 34 foto Kemangi ditebak sebagai Sirih, atau 17,65%. Berikutnya, 4 dari 34 foto Sirih ditebak sebagai Nangka (11,76%). Lalu, 3 dari 32 foto Jeruk Nipis ditebak sebagai Jambu Biji (9,38%). Tabel ini hanya menampilkan tiga pasangan yang paling sering tertukar, supaya kita bisa melihat pola kesalahan yang paling menonjol.”

![Screenshot tabel perbandingan dan salah tebak DenseNet-121](phase5-5_1_1-comparison.png)

**Script untuk menjelaskan contoh gambar dan tabel salah tebak LWDN**

“Setelah melihat tabel DenseNet-121, kita lanjut ke contoh foto yang salah ditebak dan hasil pasangan salah tebak LWDN. Untuk LWDN, pasangan yang paling sering tertukar adalah 4 dari 34 foto Kemangi yang diprediksi sebagai Sirih (11,76%). Setelah itu, 3 dari 35 foto Belimbing Wuluh diprediksi sebagai Sirih (8,57%), dan 3 dari 32 foto Jeruk Nipis diprediksi sebagai Nangka (9,38%).

Foto-foto yang ditampilkan di bawah tabel adalah satu contoh untuk setiap pasangan yang tertukar. Judul di atas foto menjelaskan daun sebenarnya, tebakan model, jumlah kejadian, dan persentasenya. Jadi, kita dapat melihat contoh gambarnya, sementara jumlah pada tabel tetap merangkum semua foto yang masuk ke pasangan tersebut.”

![Screenshot contoh salah tebak DenseNet-121 dan tabel serta contoh LWDN](phase5-5_1_1-error-examples.png)

## 5.2 — Ukuran model dan kecepatan inferensi

**Script untuk menjelaskan kode**

“Setelah membandingkan tebakan kedua model, sekarang kita lihat kebutuhan model saat digunakan. Di bagian ini, kode menyimpan checkpoint terpilih ke dalam berkas model, lalu membukanya kembali dan memastikan hasilnya tetap sama. Setelah itu, kode mengukur waktu model saat memberi prediksi.

Kode memakai gambar dari satu kelompok validation hanya untuk mengukur kecepatan. Model mencoba prediksi untuk satu gambar dan untuk sekumpulan 32 gambar. Sebelum waktu dicatat, model diberi 10 kali pemanasan, lalu pengukuran diulang 100 kali. Waktu satu gambar menunjukkan cepat atau lambatnya satu prediksi; pengukuran 32 gambar dipakai untuk menghitung banyaknya gambar yang bisa diproses per detik. Pengukuran ikut menghitung penyesuaian gambar di dalam model, tetapi tidak menghitung waktu membuka berkas atau memulai aplikasi.”

**Script untuk menjelaskan output**

“Di tabel ini, jumlah parameter DenseNet-121 adalah sekitar 7,05 juta, sedangkan LWDN sekitar 4,33 juta. Ukuran berkas DenseNet-121 sekitar 29,69 MB dan LWDN 18,43 MB. Jadi, berkas LWDN sekitar 37,9% lebih kecil dan lebih ringan untuk disimpan.

Lalu kita lihat waktu prediksi satu gambar. Median DenseNet-121 adalah 22,40 milidetik, sementara LWDN 16,56 milidetik. Artinya, pada perangkat pengujian ini, LWDN memberi prediksi sekitar 1,35 kali lebih cepat. Hasil p95 juga searah: 23,49 milidetik untuk DenseNet-121 dan 17,60 milidetik untuk LWDN. Jadi, bahkan pada pengukuran yang lebih lambat, waktu LWDN masih lebih rendah.

Untuk sekumpulan gambar, throughput DenseNet-121 adalah sekitar 228 gambar per detik, sedangkan LWDN sekitar 242 gambar per detik. Angka ini menunjukkan bahwa pada pengujian batch tersebut, LWDN memproses lebih banyak gambar dalam satu detik.

Alasan kita memilih LWDN adalah ukurannya lebih kecil dan waktu prediksinya lebih cepat, sehingga lebih praktis jika aplikasi perlu merespons dengan cepat dan memakai model yang lebih ringkas. Tapi kita tetap perlu menyampaikan tradeoff-nya: LWDN membutuhkan waktu training lebih lama, sekitar 8 menit dibanding DenseNet-121 sekitar 5,66 menit. Hasil akurasi dan macro F1 LWDN di data test juga sedikit lebih rendah. Jadi, pilihan LWDN di sini didasarkan pada efisiensi penggunaan, bukan karena nilainya paling tinggi di semua ukuran. Catatan notebook di bagian 5.5 juga menyebutkan bahwa bukti yang ada belum cukup untuk memastikan penurunan macro F1 tetap berada dalam batas 2 poin yang ditetapkan.”

![Screenshot output ukuran model dan kecepatan inferensi](phase5-5_2-efficiency-output.png)

**Output lanjutan: Ringkasan hasil run**

**Script untuk menjelaskan kode**

“Setelah kita melihat hasil test dan benchmark efisiensi secara terpisah, cell ini menyatukannya dalam satu ringkasan. Kode mengambil hasil kualitas dari evaluasi test di 5.1 dan hasil ukuran serta kecepatan dari benchmark di 5.2. Lalu, kode menghitung selisih LWDN dikurangi DenseNet-121, supaya kita bisa melihat arah dan besar perbedaannya.

Di bagian akhir, kode memeriksa batas penurunan macro F1 sebesar 2 poin persentase yang sudah ditetapkan. Kode membandingkan selisih macro F1 dengan rentang perkiraannya, lalu mencetak kesimpulan apakah bukti yang tersedia cukup untuk memenuhi batas tersebut. Cell ini hanya merangkum hasil yang sudah ada; tidak melatih ulang model.”

**Script untuk menjelaskan output**

“Bagian paling atas menunjukkan ringkasan run yang sedang kita bahas. ID `20261105T181155Z_cf66db3d` adalah penanda percobaan ini. Data dibagi menjadi 2.698 foto train, 337 foto validation, dan 337 foto test. Ada 2.172 grup untuk 3.372 foto; foto yang mirip dikelompokkan agar tidak tersebar ke bagian data yang berbeda.

Tabel ini menggabungkan kualitas pada data test dengan efisiensi dari benchmark. Di bagian kualitas, DenseNet-121 mendapat accuracy 93,18%, macro precision 93,51%, macro recall 93,12%, dan macro F1 93,15%. LWDN mendapat 91,69%, 92,29%, 91,66%, dan 91,75%. Jadi, angka kualitas DenseNet-121 sedikit lebih tinggi. Hasilnya juga terlihat pada jumlah tebakan benar: 314 foto untuk DenseNet-121 dan 309 untuk LWDN, dari 337 foto yang sama.

Di bagian efisiensi, LWDN memiliki 4,33 juta parameter dan ukuran 18,43 MB, sedangkan DenseNet-121 memiliki 7,05 juta parameter dan ukuran 29,69 MB. Waktu prediksi LWDN lebih rendah: latency median 16,56 ms dan p95 17,59 ms, dibanding 22,40 ms dan 23,49 ms pada DenseNet-121. Throughput LWDN juga lebih tinggi, sekitar 242 foto per detik dibanding 228. Namun waktu training LWDN lebih lama, yaitu 8 menit dibanding 5,66 menit.

Kolom selisih menghitung LWDN dikurangi DenseNet-121. Jadi, angka minus pada accuracy berarti hasil LWDN lebih rendah; angka minus pada ukuran dan latency berarti berkas serta waktu prediksi LWDN lebih kecil; sedangkan angka plus pada throughput berarti LWDN memproses lebih banyak foto per detik. Untuk macro F1, penurunan yang terukur adalah 1,39 poin—masih di bawah batas 2 poin yang disepakati.

Namun, rentang perkiraan 95% untuk selisih macro F1 adalah −4,28 sampai +1,60 poin. Karena rentang ini masih mencakup kemungkinan penurunan yang melewati batas 2 poin, output menyimpulkan bahwa bukti belum cukup untuk memastikan LWDN memenuhi batas tersebut. Jadi, kita bisa menjelaskan LWDN sebagai pilihan yang lebih ringkas dan cepat untuk penggunaan, sambil jujur bahwa keunggulan efisiensinya belum memastikan kualitasnya setara dengan DenseNet-121.”

![Screenshot ringkasan hasil run, kualitas test, dan efisiensi](phase5-run-summary-output.png)

# CRISP-DM Phase 6 — Deployment

**Kalimat jembatan dari Phase 5 ke Phase 6**

“Setelah kita mengevaluasi hasil kedua model dan membandingkan kecepatannya, sekarang kita masuk ke Phase 6, yaitu deployment. Di tahap ini kita menentukan model yang paling sesuai untuk dibawa ke aplikasi dengan mempertimbangkan hasil pengenalan dan kebutuhan penggunaan.”

## Pemilihan LWDN

**Script untuk menjelaskan tabel dan alasan pemilihan**

“Di tabel ini kita merangkum kelebihan dan kekurangan kedua model. Pada data test, accuracy DenseNet-121 adalah 93,18%, sedangkan LWDN 91,69%. Macro F1-nya 93,15% untuk DenseNet-121 dan 91,75% untuk LWDN. Jadi, berdasarkan angka yang tercatat, DenseNet-121 memang sedikit lebih tinggi pada hasil pengenalan.

Kolom selisih pada tabel dihitung relatif terhadap DenseNet-121. Karena itu, angka −1,6% untuk accuracy dan −1,5% untuk Macro F1 menunjukkan penurunan relatif pada LWDN. Jika kita menyebut selisihnya dalam poin persentase, accuracy LWDN sekitar 1,49 poin lebih rendah, sedangkan Macro F1 sekitar 1,39 poin lebih rendah.

Namun, rentang perkiraan 95% untuk selisih Macro F1 adalah −4,28 sampai +1,60 poin persentase. Rentang ini masih memungkinkan LWDN mendapat hasil beberapa poin lebih rendah atau sedikit lebih tinggi. Jadi, bukti yang ada belum cukup untuk memastikan perbedaan kedua model. Secara angka pada run ini, DenseNet-121 skornya lebih tinggi, tetapi kita tidak menyebutnya terbukti unggul secara statistik.

LWDN punya kelebihan pada kebutuhan aplikasi. Jumlah parameternya 38,5% lebih sedikit, ukuran berkasnya 37,9% lebih kecil, dan latency median-nya 26,1% lebih rendah pada benchmark Kaggle. Latency yang lebih rendah berarti waktu untuk memberi satu prediksi lebih singkat. Inilah nilai pengimbang LWDN: walaupun skor test-nya sedikit di bawah DenseNet-121, modelnya lebih ringkas dan lebih cepat pada benchmark tersebut.

Karena aplikasi kita mengutamakan model yang ringkas dan respons prediksi yang cepat, kita memilih LWDN untuk deployment. Pilihan ini adalah kompromi yang mengutamakan efisiensi, bukan klaim bahwa LWDN mendapat skor tertinggi di semua metrik. Kita juga perlu jujur bahwa ringkasan Phase 5 belum memastikan penurunan Macro F1 LWDN tetap dalam batas 2 poin yang disepakati. Hasil kecepatan yang ditampilkan berasal dari benchmark Kaggle, jadi performa pada perangkat ringan yang sebenarnya tetap perlu diuji langsung.”

![Screenshot bagian Pemilihan LWDN pada Phase 6](phase6-lwdn-selection.png)

**Kalimat jembatan ke demo HerbleafAI**

“Nah, teman-teman di sini pasti penasaran, kan, bagaimana implementasi model LWDN ini? Untuk itu, saya serahkan kepada Kak Rahma untuk memperlihatkan implementasi LWDN yang kami beri nama HerbleafAI. Kak Rahma, waktu dan tempat saya persilakan.”

