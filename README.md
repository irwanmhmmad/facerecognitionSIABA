# ABSEN Face Recognition Service

Self-hosted API untuk registrasi template biometrik, persetujuan admin, dan verifikasi wajah 1:1 bagi ABSENMOBILE dan ABSENWEB.

## Status implementasi

Sudah tersedia:

- sesi enrollment dengan challenge acak dan masa berlaku;
- validasi token login ABSENWEB untuk endpoint mobile tanpa menanam service key di aplikasi;
- token upload JWT sekali sesi agar aplikasi tidak menyimpan service key;
- upload tepat 6 sampel melalui `multipart/form-data`;
- penyimpanan sementara enam foto enrollment dalam bentuk terenkripsi untuk pemeriksaan admin;
- validasi format, ukuran, jumlah wajah, kualitas, dan konsistensi sampel;
- pemeriksaan berjenjang oleh KTU dan Admin BKD sebelum agregasi embedding menjadi template;
- enkripsi embedding sebelum disimpan;
- status `collecting`, `pending_ktu_review`, `pending_bkd_review`, `active`, `rejected`, `needs_recapture`, dan `revoked`;
- daftar pengajuan untuk master ABSENWEB;
- aktivasi, penolakan, permintaan rekam ulang, dan pencabutan profil;
- sesi verifikasi 1:1 dengan challenge dan token berumur pendek;
- idempotency key untuk menghadapi retry pada jaringan buruk;
- audit log tanpa menyimpan gambar mentah;
- SQLite untuk development dan PostgreSQL melalui Docker Compose;
- adapter engine development dan adapter OpenCV YuNet + SFace dengan verifikasi checksum model;
- health/readiness endpoint dan dokumentasi OpenAPI otomatis.
- laporan enrollment dan verifikasi dalam format CSV untuk ABSENWEB.

> **Penting:** `FACE_ENGINE=development` hanya menguji workflow API. Engine tersebut bukan detektor wajah atau face recognition produksi. Production mode menolak berjalan dengan engine demo atau secret bawaan.

## Arsitektur

```text
ABSENMOBILE
  | 1. meminta sesi melalui Laravel
  | 2. upload frame memakai JWT singkat
  v
ABSEN Face Recognition Service
  |- quality check / face embedding
  |- template terenkripsi
  |- verifikasi 1:1
  v
PostgreSQL

ABSENWEB/Laravel
  |- identitas pegawai dan NIP
  |- master persetujuan admin
  |- aturan absensi
  |- token login menentukan NIP pegawai
```

Service biometrik menggunakan `nip-<18 digit NIP baru>` sebagai `subject_id`. Nama, jabatan, OPD,
dan data kepegawaian lain tetap hanya dikelola ABSENWEB.

Instalasi lama yang masih menyimpan `employee-<id>` harus dimigrasikan satu kali sebelum service
baru digunakan:

```powershell
python scripts/migrate_subject_to_nip.py `
  --database .\face_service.db `
  --old-subject employee-759 `
  --nip 198001012005011001
```

Skrip membuat cadangan database terlebih dahulu dan mempertahankan status serta template aktif.

## Menjalankan development

Gunakan Python 3.11 atau lebih baru.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
Copy-Item .env.example .env
uvicorn app.main:app --reload --port 8001
```

Dokumentasi API tersedia di `http://127.0.0.1:8001/docs`.

Jalankan pemeriksaan:

```powershell
ruff check .
pytest
```

## Menjalankan dengan Docker dan PostgreSQL

Salin `.env.example` menjadi `.env`, lalu ganti seluruh secret dan password.

```powershell
docker compose up --build
```

Port default hanya diikat ke `127.0.0.1:8001`. Untuk menerima upload langsung dari aplikasi, tempatkan reverse proxy HTTPS di depan endpoint `/v1/*`. Endpoint `/internal/*` harus tetap privat dan hanya dapat dipanggil Laravel.

## Kunci enkripsi

Buat kunci Fernet satu kali dan simpan di secret manager/server environment:

```powershell
python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
```

Masukkan hasilnya ke `BIOMETRIC_ENCRYPTION_KEY`. Kehilangan kunci membuat seluruh template tidak dapat dibaca. Jangan menyimpan kunci di Git atau database biometrik.

## Alur enrollment

### 1. Aplikasi membuat sesi dengan token login ABSENWEB

```http
POST /mobile/v1/enrollments
Authorization: Bearer <token-login-absenweb>
Idempotency-Key: mobile-nip-198001012005011001-<timestamp>
Content-Type: application/json

{"struktur_org":"1,2,949,950,"}
```

Face Service memvalidasi token ke `GET /api/get-my-profile`, mengambil `nip_baru`, menormalisasinya
menjadi 18 digit, lalu menyimpan snapshot `struktur_org` dari payload enrollment. Nilai struktur
disimpan tanpa wildcard; contoh pola SQL `LIKE '1,2,949,950,%'` disimpan sebagai
`1,2,949,950,`. Service kemudian
menerbitkan token upload yang hanya berlaku untuk satu sesi. `INTERNAL_SERVICE_KEY` tidak pernah
dikirim ke aplikasi.

Untuk membaca status registrasi:

```http
GET /mobile/v1/profile
Authorization: Bearer <token-login-absenweb>
```

Respons menyertakan `face_trained`. Nilai ini hanya `true` ketika profil berstatus `active` dan
memiliki template biometrik aktif. Aplikasi wajib memeriksa flag ini sebelum memulai absensi.

Endpoint internal di bawah ini tetap tersedia bagi integrasi server-to-server ABSENWEB.

### 1b. Laravel membuat sesi (server-to-server)

```http
POST /internal/v1/enrollments
X-Service-Key: <internal-key>
Idempotency-Key: enrollment-nip-198001012005011001-v1
Content-Type: application/json

{"subject_id":"nip-198001012005011001"}
```

Respons berisi `id`, challenge acak, `upload_token`, dan `expires_at`.

### 2. Aplikasi mengirim tepat 6 sampel

```http
POST /v1/enrollments/{id}/captures
Authorization: Bearer <upload-token>
Content-Type: multipart/form-data

sample_type=frontal
capture_metadata={"captured_at":"2026-10-26T09:15:00+09:00","device_id":"device-installation-id","platform":"android","app_version":"2.3.0","camera":"front","sequence":1,"challenge_action":"blink"}
image=<jpeg/png/webp>
```

Urutan yang direkomendasikan adalah dua foto `frontal`, satu `slight_left`, satu `slight_right`,
serta dua `additional` untuk sudut sedikit menengadah dan sedikit menunduk. Keenam foto harus
berbeda, menampilkan tepat satu wajah, dan lolos pemeriksaan kualitas.

`sample_type` yang diterima: `frontal`, `slight_left`, `slight_right`, dan `additional`.
`capture_metadata` bersifat opsional, berupa JSON maksimal 4 KB, dan hanya menerima field yang
ditunjukkan pada contoh. Foto disimpan sebagai blob terenkripsi di tabel
`enrollment_sample_images`; foto tidak disimpan sebagai berkas publik di web server.

### 3. Aplikasi menyelesaikan enrollment

```http
POST /v1/enrollments/{id}/complete
Authorization: Bearer <upload-token>
Content-Type: application/json

{
  "liveness_passed": true,
  "liveness_method": "on_device_blink_and_head_turn_v1",
  "completed_challenges": ["blink", "turn-left", "turn-right"],
  "challenge_duration_ms": 4200
}
```

Service memastikan bukti challenge sesuai urutan sesi, ketiga sudut tersedia, dan keenam frame tidak
identik. Status menjadi `pending_ktu_review`. Pada tahap ini template belum dibentuk; embedding sampel
tetap terenkripsi sampai persetujuan akhir Admin BKD.

### 4. ABSENWEB menampilkan pegawai yang sudah upload

```http
GET /internal/v1/enrollments/uploaded?status=pending_ktu_review&struktur_org=1,2,949,950,&nip=198001012005011001
X-Service-Key: <internal-key>
```

Endpoint ini mendukung status `pending_ktu_review`, `pending_bkd_review`, `active`, `rejected`, dan
`needs_recapture`, filter prefix `struktur_org` (dengan atau tanpa `%`), serta pagination `limit` dan
`offset`. Parameter `nip` opsional dan harus berupa tepat 18 digit tanpa awalan `nip-`.

Untuk modal pemeriksaan, backend ABSENWEB mengambil metadata dan URL enam foto melalui:

```http
GET /internal/v1/enrollments/{id}/samples
X-Service-Key: <internal-key>
```

Jika ABSENWEB hanya memiliki NIP, enam foto pengajuan terbaru yang masih menunggu pemeriksaan dapat
diambil langsung sekaligus tetap dibatasi pada struktur organisasi admin:

```http
GET /internal/v1/enrollments/by-nip/198001012005011001/samples?struktur_org=1,2,949,950,
X-Service-Key: <internal-key>
```

Setiap `image_url` pada respons adalah endpoint internal yang mengembalikan foto setelah dekripsi.
Backend Laravel harus mem-proxy foto itu ke browser setelah memeriksa role dan OPD/unit admin;
`X-Service-Key` tidak boleh dikirim ke JavaScript/browser. Respons foto memakai `Cache-Control:
no-store` dan hanya tersedia selama status masih `pending_ktu_review` atau `pending_bkd_review`.

### 5. KTU menyetujui

```http
POST /internal/v1/enrollments/{id}/ktu-approve
X-Service-Key: <internal-key>
Content-Type: application/json

{"actor":"ktu-opd"}
```

Status berubah menjadi `pending_bkd_review`. KTU atau Admin BKD dapat memakai endpoint `/reject` atau
`/request-recapture` selama pengajuan masih berada pada tahap pemeriksaan.

### 6. Admin BKD memverifikasi dan mengaktifkan

```http
POST /internal/v1/enrollments/{id}/bkd-approve
X-Service-Key: <internal-key>
Content-Type: application/json

{"actor":"admin-bkd"}
```

Setelah persetujuan Admin BKD, service baru mengagregasi enam embedding menjadi template biometrik.
Template lama, jika ada, dicabut sebelum template baru diaktifkan. Endpoint lama `/activate` tetap
tersedia sebagai alias untuk integrasi yang sudah ada. Foto sementara dan embedding per foto
langsung dihapus setelah aktivasi, penolakan, permintaan rekam ulang, atau sesi collecting digantikan.

### Menghapus template untuk pendaftaran ulang

Administrator dapat menghapus seluruh template biometrik seorang pegawai dan membuka kembali alur
pendaftaran enam foto menggunakan NIP 18 digit:

```http
DELETE /internal/v1/profiles/by-nip/199412022025041005/template
X-Service-Key: <internal-key>
Content-Type: application/json

{
  "actor": "admin-bkd",
  "reason": "Pegawai diminta melakukan registrasi wajah ulang"
}
```

Endpoint ini benar-benar menghapus embedding template, membersihkan bukti sementara dari enrollment
yang belum selesai, mengubah profil menjadi `needs_recapture`, dan mencatat audit. Riwayat enrollment
tetap dipertahankan. Pemanggilan ulang setelah reset berhasil bersifat aman dan menghasilkan jumlah
template terhapus `0`. Setelah itu aplikasi mobile akan mengizinkan pegawai memulai pendaftaran baru.

## Alur verifikasi absensi

Aplikasi dapat membuat sesi menggunakan token login AbsenWeb. Face Service mengambil NIP dari token,
sehingga aplikasi tidak dapat memilih identitas pegawai lain:

```http
POST /mobile/v1/verifications
Authorization: Bearer <token-login-absenweb>
Idempotency-Key: attendance-198001012005011001-20261019-am
Content-Type: application/json

{}
```

Laravel membuat sesi:

```http
POST /internal/v1/verifications
X-Service-Key: <internal-key>
Idempotency-Key: attendance-198001012005011001-20261019-am
Content-Type: application/json

{"subject_id":"nip-198001012005011001"}
```

Aplikasi menjalankan challenge lokal lalu mengunggah frame terbaik:

```http
POST /v1/verifications/{id}/capture
Authorization: Bearer <upload-token>
Content-Type: multipart/form-data

liveness_passed=true
image=<jpeg/png/webp>
```

Face Service mengembalikan `match`, `score`, `threshold`, `liveness_passed`, dan `reason_code`. Laravel tetap menjadi pihak yang memutuskan dan menyimpan absensi.

## Unduh laporan

Laporan hanya tersedia melalui endpoint internal dan memerlukan `X-Service-Key`:

```http
GET /internal/v1/reports/enrollments.csv?status=active&created_from=2026-10-01T00:00:00Z
X-Service-Key: <internal-key>
```

```http
GET /internal/v1/reports/verifications.csv?subject_id=nip-198001012005011001&created_to=2026-10-31T23:59:59Z
X-Service-Key: <internal-key>
```

Filter opsional yang tersedia adalah `status`, `subject_id`, `created_from`, dan `created_to`.
File memakai UTF-8 dengan BOM agar dapat dibuka langsung di Microsoft Excel. Nilai teks juga
diamankan dari evaluasi formula spreadsheet. Laporan tidak memuat gambar, embedding, token, atau challenge.

## Memasang model produksi OpenCV

Adapter produksi menggunakan OpenCV DNN dengan YuNet 2023mar sebagai detector dan SFace 2021dec
sebagai recognizer. Server tidak mengunduh model saat startup. Model diambil secara eksplisit dan
diverifikasi menggunakan SHA-256:

```powershell
python scripts/fetch_opencv_models.py
```

Struktur file:

```text
models/
  opencv/
    face_detection_yunet_2023mar.onnx
    face_recognition_sface_2021dec.onnx
    LICENSE-YUNET.txt
    LICENSE-SFACE.txt
    manifest.json
```

Instal dependency inference:

```powershell
python -m pip install -e ".[dev]"
```

Atur environment:

```dotenv
APP_ENV=production
FACE_ENGINE=opencv
FACE_MODELS_ROOT=./models
OPENCV_MODEL_LICENSE_CONFIRMED=true
VERIFICATION_THRESHOLD=0.363
```

Nilai `0.363` adalah baseline publik SFace,
bukan threshold final; kalibrasikan menggunakan data uji sah sebelum produksi.
`OPENCV_MODEL_LICENSE_CONFIRMED=true` adalah pernyataan operator bahwa teks lisensi, provenance,
dan penggunaan model telah ditinjau. Simpan file lisensi dan `manifest.json` bersama deployment.

## Paket wajah offline dan sinkronisasi

Paket offline hanya diterbitkan untuk profil dan template aktif yang memakai model produksi saat ini.
Perangkat wajib mengirim `X-Device-Id` dan public key RSA X.509 DER Base64 milik Android Keystore:

```http
GET /mobile/v1/offline-package
Authorization: Bearer <token-login>
X-Device-Id: <device-terdaftar>
X-Device-Public-Key: <base64-x509-der-rsa>
```

Embedding database tetap terenkripsi. Saat penerbitan, embedding hanya didekripsi di memori, lalu
dienkripsi ulang memakai AES-256-GCM dengan kunci sekali pakai yang dibungkus RSA-OAEP. Paket berlaku
maksimal tujuh hari. Reset, pencabutan, dan penggantian template mencabut paket lama serta menaikkan
`templateVersion`.

Buat pasangan kunci signing terpisah dari kunci enkripsi database:

```powershell
python scripts/generate_offline_signing_key.py
```

Simpan nilai `OFFLINE_SIGNING_PRIVATE_KEY_B64` hanya di secret manager server. Salin public key yang
dihasilkan ke environment mobile sebagai `EXPO_PUBLIC_FACE_OFFLINE_PUBLIC_KEY`. Nilai tersebut adalah
X.509 DER Base64 dan aman didistribusikan. Rotasi kunci dilakukan dengan mengganti
`OFFLINE_SIGNING_KEY_ID`; mobile perlu mempertahankan public key lama selama paket lama masih berlaku.

AbsenWeb menyinkronkan bukti offline untuk verifikasi ulang server melalui:

```http
POST /internal/v1/offline-verifications
X-Service-Key: <internal-key>
Content-Type: multipart/form-data
```

Field wajib: `subject_id`, `model_id`, `face_package_id`, `captured_at` (dengan zona waktu),
`liveness_method`, `evidence_hash` (SHA-256 foto), dan `image`. Skor dari perangkat tidak diterima.
Foto hanya diproses di memori dan tidak disimpan.

## Batasan keamanan yang masih harus diselesaikan sebelum produksi

- active liveness (kedip dan gerakan kepala) saat ini dinilai pada aplikasi dan buktinya divalidasi terhadap challenge acak dari service;
- passive liveness di service saat ini berupa pemeriksaan urutan, tipe pose, dan keunikan frame. Ini bukan PAD bersertifikasi; sebelum produksi gunakan PAD pasif yang teruji atau SDK liveness native dengan hasil bertanda tangan;
- ganti pembuatan tabel otomatis dengan migration terkontrol sebelum upgrade produksi;
- simpan seluruh secret pada Vault/KMS atau secret manager;
- aktifkan TLS/mTLS, rate limit, request signing, firewall, dan rotasi kunci;
- kalibrasi threshold memakai data uji yang sah dan representatif;
- lakukan pengujian presentasi foto, layar, video replay, mask, dan injection;
- tetapkan retensi, penghapusan, audit, serta prosedur re-enrollment bersama BKD.

## Prinsip privasi

- foto enrollment disimpan sementara sebagai blob terenkripsi agar KTU/Admin BKD dapat memeriksa
  enam foto; foto dihapus setelah keputusan akhir dan tidak pernah diletakkan pada direktori publik;
- foto verifikasi absensi hanya diproses di memori dan tidak ditulis ke disk/database;
- database menyimpan template/embedding terenkripsi dan metadata minimum;
- tidak ada endpoint untuk mengunduh embedding;
- API menggunakan verifikasi 1:1, bukan pencarian 1:N;
- audit tidak mencatat gambar, embedding, token, service key, atau nama pegawai; `subject_id`
  berbasis NIP dicatat untuk penelusuran aktivitas biometrik per pegawai.

