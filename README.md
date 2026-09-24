# PhishingEmailDetection (bản mở rộng đầy đủ kỹ thuật)

Cấu trúc thư mục — bám theo đúng cách tổ chức của repo gốc:

```
PhishingEmailDetection/
├── Dataset/
│   ├── enron/              <- bỏ dữ liệu Enron (raw maildir) vào đây
│   └── phishing/           <- bỏ dữ liệu Nazario (.mbox) vào đây
├── dumped_models/
│   └── classification/     <- tự tạo khi train.py chạy, chứa mọi model
├── model/                  <- model bạn chọn để deploy (copy tay từ trên)
├── load_data.py
├── preprocess.py
├── train.py
├── evaluate.py
└── requirements.txt
```

## Cài đặt

```bash
cd PhishingEmailDetection
pip install -r requirements.txt
python -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('omw-1.4'); nltk.download('punkt'); nltk.download('punkt_tab')"
```

## Chạy

```bash
# 1. Copy dữ liệu vào Dataset/enron/ và Dataset/phishing/ (xem README trong mỗi thư mục)

# 2. Train (tạo dumped_models/classification/*.pkl + data_X_test.npy + data_y_test.npy)
python train.py

# 3. Đánh giá (in bảng Accuracy/F1/Precision/Recall/ROC-AUC + vẽ ROC/confusion matrix)
python evaluate.py
```

## 6 nhóm kỹ thuật phát hiện phishing đã tích hợp (trong `preprocess.py`)

1. Content/NLP  2. URL-based  3. Header-based  4. HTML/structure
5. Attachment  6. Social-engineering

→ 42 feature (`preprocess.FEATURE_NAMES`), ghép cùng vector Doc2Vec(100
chiều) = 142 chiều đưa vào 10 classifier trong `train.py`.

Chi tiết từng thay đổi so với repo gốc: xem docstring/comment đầu mỗi file
(`load_data.py`, `preprocess.py`, `train.py`, `evaluate.py`).
