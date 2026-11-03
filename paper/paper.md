# Overview FGD Papers

Dokumen ini merangkum lima paper tentang klasifikasi tumbuhan herbal/medisinal menggunakan citra daun dan deep learning. Fokus umum kelima paper adalah membantu identifikasi tanaman secara otomatis ketika bentuk, warna, tekstur, dan pola tulang daun antarspesies terlihat mirip.

## FGD Paper 1

**Judul:** A Hybrid Deep Learning Model for Aromatic and Medicinal Plant Species Classification Using a Curated Leaf Image Dataset

### Apa yang diteliti

Penelitian ini meneliti klasifikasi 39 spesies tanaman aromatik dan medisinal berdasarkan citra daun. Peneliti juga membuat dataset khusus dari Aromatic and Medicinal Plant Research Station, Odakkali, Kerala, India.

### Tujuan

Tujuannya adalah meningkatkan akurasi klasifikasi tanaman yang memiliki kemiripan morfologi tinggi, sekaligus menyediakan dataset yang lebih sesuai untuk penelitian klasifikasi tanaman aromatik dan medisinal.

### Metode

- Membandingkan beberapa model deep learning secara bertahap.
- Memulai dari custom CNN sebagai baseline.
- Menggunakan transfer learning dengan VGG16.
- Menerapkan data augmentation dan fine-tuning pada layer konvolusi yang lebih dalam.
- Menambahkan Squeeze-and-Excitation (SE) block untuk memberi perhatian lebih pada fitur penting seperti tekstur dan urat daun.
- Membentuk model hybrid dengan Batch Normalization, GRU, Transformer Encoder, dan Dilated Convolution.

### Algoritma/model yang dipakai

Custom CNN, VGG16, VGG19, Xception, InceptionV3, MobileNetV2, Squeeze-and-Excitation, GRU, Transformer Encoder, dan Dilated Convolution.

### Hasil

Model baseline custom CNN hanya mencapai akurasi validasi sekitar 52,53%. Performa meningkat setelah menggunakan VGG16, augmentation, fine-tuning, dan SE block. Model hybrid menjadi model terbaik dengan akurasi validasi 96,70% pada bagian diskusi/kesimpulan. Namun, abstrak paper mencantumkan angka 95,24%, sehingga angka tersebut perlu dianggap sebagai perbedaan pelaporan internal paper.

### Inti kontribusi

Kontribusi utamanya adalah dataset 39 kelas dan arsitektur hybrid yang menggabungkan fitur spasial, perhatian kanal, konteks global, serta pola sekuensial untuk klasifikasi daun yang mirip.

## FGD Paper 2

**Judul:** Medicinal Plant Leaf Classification using Deep Learning and Vision Transformers

### Apa yang diteliti

Penelitian ini meneliti klasifikasi citra daun tanaman medisinal ke dalam 10 kelas menggunakan dataset berisi 38.066 citra.

### Tujuan

Tujuannya adalah membuat sistem identifikasi daun yang otomatis, cepat, akurat, dan relatif ringan sehingga berpotensi digunakan pada perangkat dengan sumber daya terbatas.

### Metode

- Menggunakan transfer learning berbasis VGG16.
- Membandingkan CNN custom dengan model berbasis Vision Transformer.
- Menggunakan ConvMixer dan Compact Convolutional Transformer (CCT).
- Melakukan denoising menggunakan Wavelet Thresholding dan Gaussian Blurring.
- Melatih dan mengevaluasi model selama 30 epoch.

### Algoritma/model yang dipakai

VGG16, custom CNN, ConvMixer, Compact Convolutional Transformer (CCT), dan hybrid CNN-ViT.

### Hasil

Nilai AUC yang dilaporkan adalah:

- VGG16: 0,50
- ConvMixer: 0,50
- CCT: 0,79
- Custom CNN: 0,83
- Hybrid CNN-ViT: 0,74

Custom CNN memperoleh AUC tertinggi, sedangkan CCT menunjukkan performa lebih baik dibandingkan VGG16 dan ConvMixer. Model yang diusulkan memiliki jumlah parameter lebih sedikit sehingga lebih ringan secara komputasi.

### Inti kontribusi

Paper ini menunjukkan bahwa kombinasi CNN dan Vision Transformer dapat digunakan untuk klasifikasi daun medisinal, tetapi model ringan dan preprocessing denoising tetap penting untuk menjaga efisiensi.

## FGD Paper 3

**Judul:** Klasifikasi Jenis Daun Tumbuhan Herbal Berdasarkan Lontar Usada Taru Pramana Menggunakan CNN

### Apa yang diteliti

Penelitian ini meneliti klasifikasi 50 jenis daun tumbuhan herbal yang berkaitan dengan pengetahuan tradisional Bali dalam Lontar Usada Taru Pramana. Dataset yang digunakan bernama TPHerbleaf.

### Tujuan

Tujuannya adalah mengidentifikasi jenis daun herbal secara otomatis dan mendukung pemahaman serta pelestarian pengetahuan pengobatan tradisional Bali melalui teknologi klasifikasi citra.

### Metode

- Menggunakan dataset citra daun TPHerbleaf.
- Membandingkan tiga arsitektur CNN.
- Menguji konfigurasi learning rate dan dropout.
- Mengukur performa pada data training, validation, dan testing.

### Algoritma/model yang dipakai

MobileNet, Inception-ResNetV2, dan EfficientNet-B2.

### Hasil

Model terbaik adalah MobileNet dengan learning rate 0,0001 dan dropout 20%. Model tersebut mencapai:

- Akurasi training: 100%
- Akurasi validation: 79%
- Akurasi testing: 82%

Perbedaan yang cukup besar antara training dan validation menunjukkan kemungkinan overfitting atau keterbatasan generalisasi model.

### Inti kontribusi

Paper ini menghubungkan klasifikasi citra modern dengan dokumentasi pengetahuan lokal Bali, khususnya daftar tanaman herbal yang tercatat dalam Lontar Usada Taru Pramana.

## FGD Paper 4

**Judul:** Fusing Appearance and Vein Morphology Using Dual-Branch Deep Networks for Accurate Medicinal Plant Identification

### Apa yang diteliti

Penelitian ini meneliti identifikasi tujuh spesies tanaman medisinal dari pasangan citra RGB dan citra skeleton tulang daun. Dataset terdiri dari 14.344 pasangan citra.

### Tujuan

Tujuannya adalah menggabungkan informasi tampilan daun dan struktur venasi agar model lebih tahan terhadap perubahan pencahayaan, background clutter, occlusion, serta kemiripan antarspesies.

### Metode

- Membuat dua cabang model yang berjalan terpisah.
- Cabang pertama memproses citra RGB untuk menangkap warna, tekstur, dan bentuk global.
- Cabang kedua memproses citra skeleton untuk menangkap topologi dan pola tulang daun.
- Menggabungkan embedding kedua cabang pada tahap late feature-level fusion.
- Menggunakan augmentation berbasis frekuensi kelas agar kelas minoritas mendapat transformasi lebih kuat.
- Menerapkan transfer learning, class weighting, dan regularization.
- Membandingkan model dual-branch dengan RGB-only, skeleton-only, dan baseline fused-image.

### Algoritma/model yang dipakai

MobileNetV2 untuk fitur tampilan RGB dan DenseNet121 untuk fitur venasi/skeleton. Metode tambahan yang dipakai adalah late feature fusion, transfer learning, class weighting, regularization, dan class-frequency aware augmentation.

### Hasil

Model dual-branch mencapai akurasi keseluruhan 97%, dengan macro precision, recall, dan F1-score masing-masing sekitar 0,96. Kelas Centella asiatica dan Neem memperoleh recall 1,00. Kesalahan terbesar terjadi pada kelas yang memiliki kemiripan visual tinggi, seperti Arjun dan Guava.

### Inti kontribusi

Paper ini menunjukkan bahwa informasi warna/tekstur dan venasi sebaiknya dipelajari secara terpisah sebelum digabungkan. Pemisahan ini mengurangi gangguan antar-modalitas dan membuat fitur yang dihasilkan lebih saling melengkapi.

## FGD Paper 5

**Judul:** Biomedical Image Processing for Enhanced Classification of Medicinal Plants Using Voting Ensemble

### Apa yang diteliti

Penelitian ini meneliti klasifikasi 10 jenis tanaman medisinal Indonesia, yaitu guava, curry, basil, turmeric, mint, papaya, betel, soursop, aloe vera, dan green tea.

### Tujuan

Tujuannya adalah membangun klasifikasi yang akurat dan stabil dengan menggabungkan beberapa model CNN, sehingga identifikasi tanaman medisinal lebih dapat diandalkan dalam kondisi citra yang bervariasi.

### Metode

- Melatih tiga model CNN secara terpisah.
- Menggunakan held-out test set berisi 200 citra.
- Menggabungkan probabilitas kelas dari ketiga model menggunakan average voting ensemble.
- Mengevaluasi accuracy, precision, recall, F1-score, dan confusion matrix.

### Algoritma/model yang dipakai

ResNet18, VGG16, EfficientNet-B0, dan average voting ensemble.

### Hasil

| Model | Accuracy | Precision | Recall | F1-score |
|---|---:|---:|---:|---:|
| ResNet18 | 87,00% | 91,16% | 87,00% | 87,35% |
| VGG16 | 63,50% | 63,89% | 63,50% | 61,10% |
| EfficientNet-B0 | 94,00% | 94,49% | 94,00% | 94,01% |
| Average voting ensemble | 93,00% | 94,99% | 93,00% | 93,22% |

EfficientNet-B0 menjadi model tunggal terbaik dalam accuracy. Ensemble menghasilkan accuracy sedikit lebih rendah, tetapi precision tertinggi dan prediksi yang lebih stabil. Kesalahan terutama terjadi pada tanaman yang tampak mirip, misalnya Aloe Vera dengan Green Tea dan Basil Leaf dengan Green Tea.

### Inti kontribusi

Paper ini menunjukkan bahwa ensemble voting dapat meningkatkan kestabilan dan precision klasifikasi, walaupun tidak selalu mengalahkan model tunggal terbaik dalam accuracy.

## Perbandingan singkat

| Paper | Fokus utama | Dataset/kelas | Model utama | Hasil utama |
|---|---|---:|---|---|
| 1 | Hybrid deep learning dan dataset tanaman aromatik/medisinal | 39 kelas | VGG16 + SE + GRU + Transformer + Dilated Conv | 96,70% validasi menurut kesimpulan |
| 2 | CNN ringan dan Vision Transformer | 38.066 citra, 10 kelas | Custom CNN, CCT, ConvMixer, hybrid CNN-ViT | AUC tertinggi 0,83 pada custom CNN |
| 3 | Klasifikasi herbal berbasis Lontar Usada Taru Pramana | 50 kelas | MobileNet, Inception-ResNetV2, EfficientNet-B2 | 82% testing dengan MobileNet |
| 4 | Penggabungan tampilan dan venasi daun | 14.344 pasangan citra, 7 kelas | MobileNetV2 + DenseNet121 dual-branch | 97% accuracy |
| 5 | Ensemble untuk tanaman medisinal Indonesia | 200 citra uji, 10 kelas | ResNet18, VGG16, EfficientNet-B0 + voting | 94% accuracy terbaik pada EfficientNet-B0 |

## Kesimpulan umum

Kelima paper sama-sama menunjukkan bahwa CNN efektif untuk klasifikasi tanaman dari citra daun. Perbedaannya terletak pada strategi peningkatan performa:

1. Paper 1 meningkatkan representasi fitur melalui hybrid architecture yang kompleks.
2. Paper 2 menekankan model ringan dan penggunaan Vision Transformer.
3. Paper 3 menerapkan model CNN pada konteks pelestarian pengetahuan lokal Bali.
4. Paper 4 menggabungkan dua sumber informasi, yaitu tampilan dan venasi daun.
5. Paper 5 menggabungkan prediksi beberapa CNN dengan average voting.

Secara umum, model terbaik tidak hanya ditentukan oleh algoritma, tetapi juga oleh kualitas dataset, jumlah kelas, kemiripan antarspesies, preprocessing, augmentasi, dan cara evaluasi. Akurasi training yang sangat tinggi, seperti pada Paper 3, perlu dibaca bersama akurasi validation dan testing untuk mendeteksi overfitting.
