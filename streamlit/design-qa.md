# Pemeriksaan UI

## Pembaruan 27 Oktober 2026: struktur hasil

final result: passed

- Acuan terbaru: `C:/Users/ASUS/AppData/Local/Temp/codex-clipboard-aa6662d4-47bc-4689-8290-a14fc60216ae.png`.
- Bukti browser terbaru: `preview-results.png` (informasi daun) dan `preview-rejected.png` (tidak dikenali), viewport desktop bawaan 1224 × 852, tanpa normalisasi density.
- Foto berada dalam kotak persegi di kiri, penjelasan dalam persegi panjang di kanan. Foto dipertahankan utuh dengan `object-fit: contain`.
- Penjelasan hanya memuat jenis daun, nama latin, khasiat; keadaan tidak dikenali hanya memuat pesan permintaan pengguna. Status, nama file, skor keyakinan, dan catatan tambahan di baris hasil dihapus.
- Jenis daun, nama latin, serta khasiat belum diisi. Radio “Simulasi tampilan hasil” di area unggah memperagakan kedua keadaan secara manual; tidak ada keputusan model atau klaim khasiat.
- Kedua keadaan diperiksa melalui browser dan self-check. Self-check juga memeriksa gambar tetap ada pada keadaan tidak dikenali, escaping nama file pada atribut HTML, dan tidak adanya kolom Keyakinan. Tidak ada console error pada sesi pemeriksaan.
- Tata letak dan label dibandingkan dengan sketsa. Font sans serif serta palet hijau sebelumnya dipertahankan. Tidak ada dekorasi atau aset gambar tambahan. Tidak ada temuan P0/P1/P2.
- Breakpoint ponsel menumpuk penjelasan di bawah gambar; tampilan ponsel belum diperiksa ulang pada pembaruan ini. Bukti desktop lama `preview.png` dan bukti ponsel lama `preview-mobile.png` bukan acuan versi terbaru.

Bagian berikut adalah catatan pemeriksaan versi awal, sebelum perubahan judul dan struktur hasil.

final result: passed

## Acuan dan bukti

- Sketsa pengguna: `C:/Users/ASUS/AppData/Local/Temp/codex-clipboard-b1728f1c-43c3-4833-9b74-b6a419ca6b82.png` (962 × 980 piksel; mencakup toolbar aplikasi gambar).
- Implementasi: `app.py`, preview lokal `http://localhost:8501`.
- `preview.png`: keadaan awal, viewport desktop 1280 × 900.
- `preview-mobile.png`: keadaan awal, viewport ponsel 390 × 844.
- `preview-results.png`: bukti interaksi folder berisi gambar valid dan file rusak, desktop 1280 × 900; diambil sebelum perapian jarak/kontras akhir.
- Screenshot diambil dari browser aplikasi. Ukuran CSS dan piksel screenshot sama; tidak ada normalisasi density. Sketsa adalah wireframe, bukan referensi tipografi atau ukuran CSS. Perbandingan menilai susunan dan alur, bukan kesamaan piksel.

## Hasil perbandingan

Sketsa dan screenshot implementasi dibuka bersama dalam satu pemeriksaan visual. Dua area utama dipertahankan: unggah dan tombol di atas, daftar gambar serta informasi di bawah. Pilihan radio menggantikan dua input yang tampil bersamaan agar sumber unggahan jelas. Warna hijau mengikuti permintaan pengguna, menggantikan warna kertas sketsa. Dekorasi sudut tidak ditambahkan agar tetap sederhana.

- Tipografi: sans serif bawaan Streamlit, satu judul utama dan dua judul bagian. Teks Indonesia ringkas; judul dan label tidak terpotong pada lebar 390 piksel.
- Jarak: panel mengikuti satu kolom. Pemeriksaan awal menemukan margin heading HTML terlalu besar; margin dikurangi dan screenshot desktop/ponsel diambil ulang setelah perbaikan. Tidak ada temuan P0/P1/P2 tersisa.
- Warna: hijau daun `#527d32`, latar `#f6f8f3`, panel putih, teks hijau gelap. Teks sekunder diperkuat pada pemeriksaan akhir. Tanpa gradien atau bayangan dekoratif.
- Gambar: foto valid ditampilkan sesuai proporsi dan orientasi. Ikon memakai Material bawaan Streamlit. Gambar sketsa hanya dipakai sebagai file uji, bukan sebagai prediksi tanaman atau aset produk.
- Isi: jenis daun, nama latin, dan keyakinan memiliki tempat pada baris hasil. Semua hasil sekarang diberi status belum dianalisis; tidak ada nama tanaman atau skor yang direkayasa. Keadaan “tidak dikenali” menunggu model serta aturan penolakan yang diuji.
- Pemeriksaan fokus: uploader dan tombol diuji langsung; baris hasil diperiksa dari screenshot hasil dan accessibility tree. Label, nama file, pesan kesalahan, serta ketiga informasi hasil terbaca.

## Interaksi dan pemeriksaan teknis

- Browser: unggah satu gambar, tekan Prediksi, lihat pratinjau; unggah folder dua file, periksa pesan file rusak dan keberhasilan file valid; ganti mode dan pastikan hasil lama hilang; buka/tutup tips.
- Desktop 1280 × 900 dan ponsel 390 × 844 diperiksa. Lebar konten ponsel 390 piksel sama dengan viewport; hasil bagian bawah diakses dengan scroll vertikal.
- Tidak ada console error pada sesi yang diperiksa.
- `python self_check.py`: lulus untuk PNG/JPEG, orientasi EXIF, thumbnail, data rusak/terpotong, format tidak didukung, batas byte/piksel, render awal, dan pilihan folder.

## Batas pemeriksaan

Ini prototipe UI lokal. Inferensi, ketepatan identifikasi, penolakan gambar non-daun, dan hosting publik belum diuji atau diimplementasikan. Teks tombol unggah bawaan Streamlit masih berbahasa Inggris; label dan panduan di sekitarnya berbahasa Indonesia. Pengujian dengan pembaca layar dan perangkat ponsel fisik belum dilakukan.
