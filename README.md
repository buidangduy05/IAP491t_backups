# Phishing Email Detection (IAP491)

Phát hiện email phishing từ **raw email** (`.eml`, mbox, CSV): trích 134 feature kỹ thuật cùng một vector
Doc2Vec 100 chiều, train **9 classifier** và chọn **1 model tốt nhất trên tập validation**.
Tập test chỉ được dùng để đánh giá sau cùng, không dùng để chọn model hay ngưỡng.

## Cài đặt

Python 3.11.

```powershell
python -m venv venv
./venv/Scripts/Activate.ps1          # Linux/macOS: source venv/bin/activate
pip install -r requirements.txt
```

## Dữ liệu

Dataset **không có trong repo**. Đặt dữ liệu vào `Dataset/` theo cấu trúc sau:

| Thư mục / file | Nội dung | Nhãn |
|---|---|---|
| `Dataset/enron/maildir/` | Enron, khoảng 517k email nội bộ | Hợp lệ |
| `Dataset/spamassassin/` | Các thư mục `*_easy_ham`, `*_hard_ham`, `*_spam` | Hợp lệ / spam theo tên thư mục |
| `Dataset/phishing/` | Nazario (`phishing-2015` … `phishing-2025`, `phishing0-2.mbox`, `20051114.mbox`) | Phishing |
| `Dataset/phishing/*.csv` | CEAS-08, Enron, Ling, SpamAssassin, TREC-05/06/07 (Zenodo 8339691, kiểm tra MD5) | Hợp lệ / `spam_or_phishing` |
| `Dataset/EPVME_extracted/` | 49k email tấn công giả lập (spoofing, XSS, MIME) | `malicious_synthetic` |
| `Dataset/phishing_pot/email/*.eml` | Phishing thật từ honeypot ([phishing_pot](https://github.com/rf-peixoto/phishing_pot), CC BY-NC 4.0) | Phishing |
| `Dataset/malspam/<năm>/*.zip` | Email malspam thật kèm file độc từ [malware-traffic-analysis.net](https://www.malware-traffic-analysis.net/), giữ nguyên zip có mật khẩu | `malspam` (lớp dương) |
| `Dataset/StealthPhisher2025.csv` | Corpus URL, không phải email | Không dùng để train email |
| `Dataset/df.csv` | Nhãn 0/1/2 không có tài liệu | Bị loại (vẫn được ghi lại) |

Tải Nazario: `python download_nazario.py`. Tải malspam: `python download_malspam.py` (đọc phần Sandbox trước).

Quy tắc nhãn:
- Nhãn lấy theo nguồn dữ liệu, **không bao giờ suy từ nội dung**.
- Chỉ những file Nazario có tên đã đăng ký mới được gán nhãn phishing.
- File hoặc dòng không rõ nhãn được giữ lại với nhãn `unknown` để xem xét, không đưa vào train.

## Chạy

```powershell
# (Tùy chọn) Đọc và đối soát toàn bộ dataset, không train.
python train.py --audit-only --output runs/audit

# Train 9 model và chọn 1 model trên validation.
python train.py --workers 4 --output runs/baseline_v2

# Tính metrics trên test, vẽ ROC/confusion matrix, phân tích lỗi theo nguồn.
python evaluate.py --run runs/baseline_v2

# (Tùy chọn) Feature importance của các model dạng cây.
python analyze_feature_importance.py --model-dir runs/baseline_v2/dumped_models/classification
```

| Tùy chọn | Ý nghĩa |
|---|---|
| `--target-mode phishing` (mặc định) | Phishing + EPVME / hợp lệ. Spam và `spam_or_phishing` bị loại |
| `--target-mode broad` | Mọi thư độc hại (kể cả spam) / hợp lệ: tầng 1 của lọc hai tầng |
| `--target-mode phishing_vs_spam` | Phishing / spam, chỉ dùng nguồn ghi rõ loại: tầng 2 |
| `--workers N` | Số process tiền xử lý (mặc định tối đa 4) |
| `--cache-dir` | Thư mục cache tiền xử lý (mặc định `.pipeline_cache/`) |

Mỗi lần chạy cần một thư mục `--output` mới. Run cũ không bao giờ bị ghi đè.

> ⚠️ **RAM và thời gian.** Toàn bộ dataset có khoảng 800k email. Máy 16 GB nên đóng bớt ứng dụng khác
> trước khi chạy. SVM (`SVC(probability=True)`) và KNN chậm khi tập train có vài trăm nghìn dòng.

## Sandbox (Docker)

Malspam, EPVME và Phishing Pot chứa malware thật. Nên chạy pipeline trong sandbox phân tích **tĩnh**:
email và attachment chỉ được đọc, **không bao giờ được chạy**.

```powershell
./sandbox/sandbox.ps1 build              # build image distroless (Python 3.11, đúng phiên bản thư viện của venv)
./sandbox/sandbox.ps1 import-dataset     # copy Dataset/ vào Docker volume (chạy lại khi thêm dữ liệu)
./sandbox/sandbox.ps1 verify             # 16 kiểm tra cách ly, chạy từ bên trong container
./sandbox/sandbox.ps1 test
./sandbox/sandbox.ps1 features --workers 4 --output runs/features_v4   # đọc email + trích feature TRONG sandbox
python train.py --from-features runs/features_v4 --output runs/baseline_v4   # train 9 model trên MÁY THẬT
python evaluate.py --run runs/baseline_v4                                   # đánh giá trên máy thật
./sandbox/sandbox.ps1 inspect D:\mail\suspicious.eml   # báo cáo attachment + luật cho 1 email
```

| Lớp cách ly | Cách làm |
|---|---|
| Không mạng | `--network none`: chỉ có card `lo`; không tới được internet, LAN, máy host (`host.docker.internal`, 192.168.65.254) hay container khác; không có DNS. Chặt hơn cả "mạng nội bộ" |
| Filesystem | Root chỉ đọc; dataset mount **chỉ đọc**; thư mục duy nhất trên máy thật ghi được là `runs/<tên>` của lần chạy đó; `/tmp` là tmpfs `noexec` |
| Quyền | User không phải root (uid 10001), `--cap-drop ALL` (kể cả khi import), `no-new-privileges`, seccomp mặc định; không mount Docker socket; daemon Docker không mở cổng TCP |
| Tài nguyên | RAM 7 GB (vượt thì container bị dừng, Windows không bị treo), 6 CPU, tối đa 512 process |
| Hệ điều hành | Image Linux: malware Windows không chạy được; lỗi parser bị nhốt trong container |
| Defender | Dataset nằm trong volume của máy ảo WSL2, nên không bị cách ly mẫu giữa chừng |

**Chia việc:** chỉ bước đọc email và trích feature (header, URL, HTML/script, text, attachment) chạm vào nội dung
độc hại, nên chạy trong sandbox. Kết quả được xuất ra dạng **không chạy được code**: vector `.npy` (đọc với
`allow_pickle=False`), text sạch `.jsonl.gz`, metadata CSV, cùng SHA-256 của từng file trong `features_manifest.json`.
Máy thật kiểm tra hash rồi train Doc2Vec + 9 model bằng RAM đầy đủ, và không bao giờ đọc email thô. Train toàn bộ
trong sandbox (`sandbox.ps1 train`) vẫn có, nhưng các classifier vượt giới hạn 7 GB với dữ liệu đầy đủ.

Image cuối là **distroless** (`gcr.io/distroless/python3-debian12:nonroot`, Python 3.11.2 của Debian 12): chỉ có
Python và thư viện của nó, không có shell, `apt`, `tar`, `perl`, `coreutils` hay `pip`. Thư viện được cài ở tầng build
(`python:3.11-slim-bookworm`, chỉ dùng wheel). Cả hai digest đều được pin.
`docker scout quickview iap491-sandbox` (2026-10-26): **0 critical, 0 high, 0 medium, 0 low** (đã đọc 47 gói Python,
55 gói Debian). `nltk`/`textblob` đã bị gỡ khỏi dự án vì CVE-2026-81726 chưa có bản vá: danh sách stopword được chép
nguyên văn vào `preprocess.NLTK_ENGLISH_STOPWORDS`, còn lemmatization động từ đã tắt (`lemmatization: False`).
Thỉnh thoảng làm mới digest để nhận bản vá mới.

Giới hạn còn lại: container dùng chung kernel với máy ảo WSL2 của Docker Desktop. Một lỗ hổng thoát container
(rất hiếm, và đã bị seccomp + bỏ capability + non-root làm khó thêm) sẽ rơi vào máy ảo đó, không phải thẳng vào
Windows. Cách ly mạnh hơn nữa cần một máy ảo riêng (VirtualBox) hoặc Enhanced Container Isolation (bản Docker Business).

## Cách hoạt động

```
raw email ─► load_data.py ─► khử trùng + gom nhóm gần trùng ─► chia train/validation/test 60/20/20 theo nhóm
                                     │
                                     ▼
          preprocess.py: text hiển thị đã khử obfuscation ─► Doc2Vec (100 chiều)
                         134 feature kỹ thuật ─► StandardScaler (fit trên train)
                                     │
                                     ▼
          train.py: 9 classifier ─► chọn ngưỡng trên validation ─► chọn 1 model ─► evaluate.py (test)
```

**Chống rò rỉ dữ liệu**
- Email trùng nội dung (SHA-256) chỉ giữ một bản. Cùng nội dung nhưng nhãn mâu thuẫn thì loại cả hai, ghi vào `label_conflicts.csv`.
- Email gần trùng (SimHash, Hamming ≤ 3) luôn nằm chung một tập, không bị chia giữa train và test.
- Vocabulary Doc2Vec và StandardScaler chỉ fit trên tập train.

**Tiền xử lý text**
- Bỏ phần HTML bị ẩn (`display:none`, chữ cùng màu nền, `font-size:0`…).
- Ghép chữ bị tách bằng tag (`Pa<span>ss</span>word`).
- Đổi homoglyph Cyrillic/Greek, chữ "font lạ" (math bold, fullwidth, small caps) về Latin.
- Xóa ký tự vô hình.
- Bỏ các chuỗi chỉ lộ nguồn dữ liệu: subject mẫu của EPVME, địa chỉ `phishing@pot`.

### 134 feature (`preprocess.BASELINE_FEATURE_NAMES`)

| Nhóm | Số | Ví dụ |
|---|---:|---|
| Nội dung | 7 | Độ dài, tỷ lệ chữ hoa, dấu `!`, ký hiệu tiền |
| URL | 11 | Số URL, IP, rút gọn link, TLD đáng ngờ, anchor/href lệch |
| Header | 6 | Freemail, Reply-To khác người gửi, thương hiệu trong display name |
| HTML | 4 | Form, script, phần tử ẩn, email chỉ có ảnh. `is_html` và `html_to_text_ratio` **không** vào model: `is_html` chỉ là điều kiện để chạy các kiểm tra HTML (email không có phần HTML nhận 0); `html_to_text_ratio` > 0 đúng khi có HTML nên cũng chỉ lộ "có HTML hay không" |
| Attachment | 19 | File thực thi ngụy trang, macro, double extension, archive mã hóa, PDF active content, nội dung active (HTML/script/binary) mang tên tài liệu/ảnh theo Magika |
| Social engineering | 5 | Từ khóa khẩn cấp, thông tin nhạy cảm, thương hiệu, lời chào chung chung, trúng thưởng |
| Header security / giả mạo | 21 | Nhiều `From`, From injection, envelope sai cú pháp, Return-Path lệch, domain giả dạng/ghép thương hiệu, domain người gửi giả dạng domain người nhận |
| Link | 14 | URL chứa email người nhận, open redirect, hosting miễn phí, anchor nêu thương hiệu sai domain, link giả dạng domain người gửi, tỷ lệ link ra ngoài domain người gửi, hình dạng host lạ nhất (độ dài, chữ số, gạch nối, entropy, subdomain, punycode) |
| Payload attachment | 11 | olevba (AutoExec, VBA stomping, XLM), HTML smuggling, ISO, LNK, OneNote, PDF `/URI`, `.mobileconfig` |
| HTML/script | 13 | `eval`, `atob`, `document.write`, iframe ẩn, ô nhập mật khẩu, chuỗi mã hóa dài |
| HTML payload | 23 | **URL trong thuộc tính HTML**: scheme nguy hiểm (`javascript:`, `data:text/html`…), URL bị obfuscate (entity/tab trong scheme, `user@host`, IP hex/thập phân, `%` trong host, `\`), form gửi ra ngoài, `<meta refresh>`, `<base href>`. **Phần tử ẩn chứa payload**: link ẩn tới domain không có trong phần hiển thị, chuỗi mã hóa (base64/hex/`%xx`/`\x`), script/markup, chuỗi entropy cao. **JavaScript bị obfuscate**: escape `\x`/`\u`, mảng char-code, cắt nối chuỗi, chữ ký packer/JSFuck/aaencode/jjencode/obfuscator.io, chuỗi mã hóa entropy cao, giải mã rồi thực thi; redirect, ghi DOM. **HTML injection**: payload markup/script trong query của link, markup active trong giá trị thuộc tính, markup bị mã hóa (`&lt;script`, `%3Cscript`), thẻ nguy hiểm (`object`, `embed`, `iframe srcdoc`, `svg onload`), CSS `expression()` |

Mọi phân tích attachment đều **tĩnh**: không thực thi, không giải nén ra đĩa, không mở URL.

**Loại nội dung file (Magika).** Ngoài magic byte, mỗi attachment (kể cả file trong ZIP) được Google Magika
(`magika==1.0.3`, model deep learning chạy offline trên CPU) gán nhãn nội dung (`content_type` trong
`inspect_email_attachments`). Magika mạnh ở định dạng text mà magic byte không phân biệt được: HTML, JavaScript,
VBScript, PowerShell, batch, script UTF-16. Tên/MIME là tài liệu hoặc ảnh (`.pdf`, `.docx`, `.jpg`, `image/*`…)
nhưng nội dung active → `attachment_disguised_active_count`. Magika chỉ cho biết **loại** file, không phán file độc.

**URL bị gateway bọc.** Link do gateway phía người nhận viết lại (Outlook Safe Links, Proofpoint URL Defense v1–v3,
Barracuda LinkProtect, Trend Micro Click-Time) được gỡ về URL gốc (`unwrap_gateway_url`) trước mọi feature URL/link.
Redirect phía người gửi (`google.com/url`, `l.facebook.com`) giữ nguyên, vì đó là dấu hiệu open redirect.

**File gửi qua link cloud** (Drive, OneDrive/SharePoint, Dropbox, iCloud Mail Drop, Proton Drive, WeTransfer, Box,
MEGA, Zoho WorkDrive, Adobe) và email thông báo chia sẻ của nhà cung cấp: `inspect_cloud_file_links`,
`cloud_file_link_count`, `cloud_share_notification` trong `_content_indicators`. Chỉ là **bằng chứng**, không vào
model: dữ liệu legit (Enron, SpamAssassin ham) gần như không có link loại này, nên model sẽ học "email hiện đại = phishing".

### Luật chắc chắn (`preprocess.ATTACHMENT_RULES`)

Gần như không email nào trong tập train mang macro tự chạy, LNK hay HTML smuggling, nên model học được trọng
số gần 0 cho các feature này. Vì vậy 15 dấu hiệu dưới đây **tự gắn cờ email**, chạy song song với model:
file thực thi ngụy trang, exe nhúng trong file khác, double extension, tên file chứa ký tự đảo chiều (RLO),
macro AutoExec, VBA stomping, macro XLM, LNK chứa lệnh, HTML smuggling, ISO/IMG, OneNote nhúng file,
archive có mật khẩu kèm mật khẩu trong body, Apple `.mobileconfig`, nội dung active mang tên tài liệu/ảnh (Magika),
và **JavaScript bị obfuscate trong HTML body** (`obfuscated_script`).

- Mỗi luật đã được đo: **0 lần kích hoạt** trên email hợp lệ có attachment (119 email SpamAssassin ham;
  Enron maildir và các CSV không có attachment).
- Luật Magika (`disguised_active_content`, đo 2026-10-31): 0/556 email SpamAssassin ham có phần đính kèm
  (tính cả phần inline); bắt 10 spam, 6 Phishing Pot (HTML đặt tên `.pdf`), 35 malspam (JS/PE đặt tên `.png`/`.jpg`).
- Luật `obfuscated_script` (`html_js_obfuscated`, đo 2026-11-02): kích hoạt khi có chữ ký obfuscator (packer,
  JSFuck, aaencode, jjencode, obfuscator.io) hoặc ≥ 2 trong 6 dấu hiệu (escape, mảng char-code, cắt nối chuỗi,
  chuỗi mã hóa, giải mã rồi thực thi). 0/1.008 email HTML hợp lệ (SpamAssassin ham + CSV Zenodo), dù 129 email
  trong số đó có JavaScript thật. JSON-LD (`application/ld+json`) không bị tính là script.
- Không dùng làm luật: "có file thực thi" (5/119 email hợp lệ), script/form và PDF active content (phổ biến ở
  file hợp lệ, và model có hơn 200 ví dụ phishing để tự học).
- `evaluate.py` báo thêm dòng `<model> + attachment rules` (model **hoặc** luật, cùng ngưỡng) và file
  `attachment_rule_hits.csv`. Web demo gọi `preprocess.assess_attachment_rules(raw)`.

Domain được tách theo **Public Suffix List** (bản đóng gói trong `tldextract==5.4.0`, không cần mạng):
`a.b.example.co.uk` → `example.co.uk`, `user.github.io` là một site riêng. Phát hiện giả dạng có hai lớp:
so với `BRAND_DOMAINS` (khoảng 28 thương hiệu), và so link/người gửi/người nhận **với nhau**
(`domains_look_alike`: homoglyph, `rn`→`m`, sửa 1–2 ký tự, đổi đuôi tên dài, ghép bằng gạch nối).
Lớp thứ hai dùng được cho mọi domain, không cần danh sách.

Những dấu hiệu được trích nhưng **cố ý không đưa vào model**, vì trong các corpus này chúng phản ánh nguồn
chứ không phải phishing: Authentication-Results/SPF/DKIM, số hop `Received`, Message-ID, X-Mailer, số đếm
obfuscation, dấu hiệu BEC, độ phổ biến domain theo Tranco (`enron.com` không có trong danh sách 2026), host trên
suffix private của PSL (chỉ phổ biến ở phishing sau 2015).

### 9 classifier và cách chọn

- **9 classifier:** XGBoost, Random Forest, MLP, Logistic Regression, SVM, KNN, Decision Tree, AdaBoost, Gradient Boosting.
  Naive Bayes đã bị loại (2026-11-02): GaussianNB bão hòa xác suất về 0/1 trên vector Doc2Vec + feature đã chuẩn hóa (baseline_v5: ngưỡng 1e-45, recall 0,65).
- **Tune:** XGBoost, Random Forest và MLP có nhiều bộ tham số ứng viên, được so trên validation.
- **Ngưỡng mỗi model:** trên validation, lấy recall cao nhất với điều kiện precision ≥ 0,9 (`MIN_VALIDATION_PRECISION`).
- **Model được chọn:** model tốt nhất theo quy tắc trên, ghi vào `run_manifest.json` (`selected_model`).

## Kết quả của một run

| File | Nội dung |
|---|---|
| `dumped_models/classification/*.pkl` | 9 model, `model_dbow.pkl` (Doc2Vec), `feature_scaler.pkl` |
| `dumped_models/classification/run_manifest.json` | Model được chọn, ngưỡng, tên feature, chữ ký tiền xử lý, hash các file |
| `validation_results.csv`, `ablation_validation.csv` | Kết quả mọi ứng viên trên validation; so sánh Doc2Vec / feature / kết hợp |
| `evaluation_results.csv`, `evaluation_by_source.csv`, `error_analysis.csv` | Metrics trên test (sau khi chạy `evaluate.py`), kể cả dòng `+ attachment rules` |
| `attachment_rule_hits.csv` | Số email trong test mà mỗi luật attachment bắt được, và bao nhiêu trong số đó thật sự là phishing |
| `ROCs.png`, `confusion-matrix.png` | Biểu đồ trên test |
| `model-metrics.png` | Biểu đồ cột Precision, Recall, F1, ROC-AUC, AP, FP của từng model trên test |
| `load_summary.json`, `ingestion.sqlite` | Đối soát từng file/bản ghi đã đọc |
| `source_labels.csv`, `label_conflicts.csv`, `label_scope_excluded.csv`, `training_dispositions.csv` | Nhãn theo nguồn, mâu thuẫn, bị loại, và lý do cho từng email |

## Giới hạn cần biết

- **Mỗi nguồn chỉ chứa một lớp.** Ví dụ Enron toàn email hợp lệ, Nazario toàn phishing. Model có thể học "nguồn" thay vì "phishing", nên hãy xem `evaluation_by_source.csv` chứ đừng chỉ nhìn F1 tổng.
- **Spam** có nhãn rõ chỉ đến từ SpamAssassin (khoảng năm 2002), nên tầng `phishing_vs_spam` dễ học theo thời kỳ.
- **Windows Defender** có thể cách ly mẫu EPVME/Phishing Pot chứa malware khi chúng được mở. Loader ghi lại các file bị chặn (`files_blocked_by_antivirus`) thay vì dừng, nên số email đọc được có thể khác nhau giữa các máy.
- `Dataset/phishing_pot/email/sample-3989.csv` là danh sách dữ liệu cá nhân bị rò rỉ, không phải email. Loader không đọc nó. Không commit thư mục này.
- **Web demo / suy luận** phải dùng `preprocess.extract_baseline_features()` và `preprocess.infer_document_vector()` để vector khớp với lúc train.

## Cấu trúc mã nguồn

| File | Vai trò |
|---|---|
| `load_data.py` | Đọc mọi nguồn, gán nhãn theo nguồn, đối soát |
| `preprocess.py` | Làm sạch text, trích feature, khử trùng, gom nhóm, chia tập |
| `pipeline_cache.py` | Cache tiền xử lý (SQLite) và xử lý song song |
| `train.py` | Doc2Vec + 9 classifier, chọn model trên validation |
| `evaluate.py` | Đánh giá trên test, không chỉnh ngưỡng |
| `analyze_feature_importance.py` | Feature importance, đánh dấu feature nghi là shortcut theo nguồn |
| `specialists.py`, `predict_specialists.py` | Chế độ cũ: 4 nhánh chuyên trách (`--mode specialists`) và lọc hai tầng |
| `dataset_audit.py`, `benchmark_pipeline.py`, `download_nazario.py`, `download_malspam.py` | Đối soát, benchmark tiền xử lý, tải Nazario và malspam |
| `sandbox/` | Dockerfile, danh sách thư viện pin phiên bản, `sandbox.ps1` |
| `tests/test_pipeline.py` | `python -m unittest discover -s tests -v` |

Tài liệu chi tiết hơn: [SPECIALIST_PIPELINE.md](SPECIALIST_PIPELINE.md), [PIPELINE_VALIDATION.md](PIPELINE_VALIDATION.md).
