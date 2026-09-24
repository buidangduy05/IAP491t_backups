# #### Cleaning + feature engineering
#
# Giữ nguyên các hàm gốc của repo (cleanText, spelling_correction,
# remove_stopwords, tokenize_text) để unit-test.py không bị vỡ.
#
# Bổ sung thêm các hàm trích xuất đặc trưng (feature engineering) cho ĐẦY ĐỦ
# các nhóm kỹ thuật phát hiện phishing, không chỉ dựa vào nội dung văn bản:
#   - extract_url_features            (nhóm URL)
#   - extract_header_features         (nhóm Header)
#   - extract_html_features           (nhóm HTML/structure)
#   - extract_attachment_features     (nhóm Attachment)
#   - extract_social_engineering_features (nhóm Social-engineering)
#   - extract_content_stats_features  (nhóm Content/NLP thống kê)
#   - extract_engineered_features(raw_text) -> list[float]  (hàm tổng, dùng
#     trong train.py, PHẢI được gọi trên text GỐC, trước khi cleanText() xoá
#     mất URL/HTML/header)

from bs4 import BeautifulSoup
import re
import hashlib
from textblob import TextBlob
import nltk
from nltk.corpus import wordnet
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

import email
from email.utils import parseaddr, getaddresses
from urllib.parse import urlparse


# --------------------------------------------------------------------------
# Các hàm GỐC của repo (không đổi logic, giữ tương thích unit-test.py)
# --------------------------------------------------------------------------

def cleanText(text):
    try:
        text = BeautifulSoup(text, "html.parser").text
    except Exception:
        # Một số email phishing thật chứa markup cố tình làm dị dạng (SGML
        # marked-section kiểu <![if IE]>...<![endif]>, thẻ hỏng...) khiến
        # bs4/html.parser reject thẳng (ParserRejectedMarkup) thay vì trả
        # về text như bình thường -- fallback sang regex strip-tag đơn giản
        # (_strip_html, định nghĩa bên dưới) để không mất cả email đó.
        text = _strip_html(str(text))
    # remove all special characters
    text = re.sub('[^A-Za-z0-9 ]+', r'', text)
    text = re.sub(r'\|\|\|', r' ', text)
    # remove http(s) links with <URL>
    text = re.sub(r'http\S+', r'<URL>', text)
    # remove unwanted lines starting from special charcters
    text = re.sub(r'\n: \'\'.*', '', text)
    text = re.sub(r'\n!.*', '', text)
    text = re.sub(r'^:\'\'.*', '', text)
    # remove non-breaking new line characters
    text = re.sub(r'\n', ' ', text)
    text = text.lower()
    text = text.replace('x', '')
    return text


def spelling_correction(text):
    text = TextBlob(text)
    text = text.correct()
    return text


def remove_stopwords(text):
    stop_words = set(stopwords.words('english'))
    word_tokens = nltk.word_tokenize(text)
    filtered_sentence = [w for w in word_tokens if not w in stop_words]
    return ' '.join(filtered_sentence)


def tokenize_text(text):
    tokens = []
    for sent in nltk.sent_tokenize(text):
        for word in nltk.word_tokenize(sent):
            if len(word) < 2:
                continue
            tokens.append(word.lower())
    return tokens


# --------------------------------------------------------------------------
# Danh sách từ khoá / domain tham chiếu cho các nhóm kỹ thuật bổ sung
# --------------------------------------------------------------------------

URL_REGEX = re.compile(r'(https?://[^\s"\'<>\)\]]+|www\.[^\s"\'<>\)\]]+)', re.IGNORECASE)

# Giới hạn an toàn khi xử lý dữ liệu phishing thật (nội dung độc hại, có thể
# bị cố tình làm bất thường để "bẻ" parser): cắt bớt text quá dài và giới
# hạn số URL/anchor xử lý mỗi email, để 1 email dị thường không làm cả batch
# chạy chậm bất thường (không ảnh hưởng gì tới email bình thường).
_MAX_TEXT_LEN = 20000
_MAX_URLS_PER_EMAIL = 200
_MAX_ANCHORS_PER_EMAIL = 200

SHORTENER_DOMAINS = {
    "bit.ly", "tinyurl.com", "goo.gl", "t.co", "ow.ly", "is.gd", "buff.ly",
    "adf.ly", "bit.do", "cutt.ly", "rb.gy", "shorte.st", "s.id", "tiny.cc",
}

SUSPICIOUS_TLDS = {
    ".zip", ".top", ".xyz", ".click", ".country", ".gq", ".tk", ".ml",
    ".cf", ".ga", ".work", ".support", ".loan", ".men", ".review", ".win",
}

FREEMAIL_DOMAINS = {
    "gmail.com", "yahoo.com", "hotmail.com", "outlook.com", "aol.com",
    "mail.com", "protonmail.com", "yandex.com", "gmx.com",
}

EXECUTABLE_EXTENSIONS = {
    ".exe", ".scr", ".bat", ".cmd", ".com", ".pif", ".js", ".jse", ".vbs",
    ".vbe", ".jar", ".msi", ".ps1", ".wsf", ".hta",
}
MACRO_EXTENSIONS = {".docm", ".xlsm", ".pptm", ".dotm", ".xltm"}
ARCHIVE_EXTENSIONS = {".zip", ".rar", ".7z", ".ace", ".iso"}

URGENCY_KEYWORDS = [
    "urgent", "immediately", "act now", "act fast", "expire", "expires",
    "expiring", "suspend", "suspended", "suspension", "verify your account",
    "verify now", "confirm your account", "within 24 hours",
    "limited time", "final notice", "last warning", "account will be closed",
    "unauthorized access", "unusual activity", "security alert",
]
SENSITIVE_INFO_KEYWORDS = [
    "password", "social security", "ssn", "credit card", "card number",
    "cvv", "pin number", "bank account", "routing number", "login credentials",
    "update your billing", "confirm your identity", "date of birth",
]
BRAND_KEYWORDS = [
    "paypal", "amazon", "apple", "microsoft", "google", "netflix", "irs",
    "bank of america", "wells fargo", "chase", "dhl", "fedex", "usps",
    "linkedin", "facebook", "instagram", "docusign", "office365",
]
GENERIC_GREETINGS = [
    "dear customer", "dear user", "dear valued customer", "dear member",
    "dear account holder", "dear sir/madam", "dear sir", "dear madam",
    "dear client", "hello customer",
]
REWARD_KEYWORDS = [
    "congratulations", "you have won", "you've won", "claim your prize",
    "free gift", "winner", "lottery", "selected to receive",
]


# --------------------------------------------------------------------------
# Helpers nội bộ để tách raw email -> plain/html body
# --------------------------------------------------------------------------

def _safe_decode(part):
    try:
        payload = part.get_payload(decode=True)
        if payload is None:
            payload_str = part.get_payload()
            return payload_str if isinstance(payload_str, str) else ""
        charset = part.get_content_charset() or "utf-8"
        return payload.decode(charset, errors="replace")
    except Exception:
        return ""


def _get_email_body(msg):
    plain_parts, html_parts = [], []
    if msg.is_multipart():
        for part in msg.walk():
            disp = str(part.get("Content-Disposition") or "")
            if "attachment" in disp:
                continue
            ctype = part.get_content_type()
            if ctype == "text/plain":
                plain_parts.append(_safe_decode(part))
            elif ctype == "text/html":
                html_parts.append(_safe_decode(part))
    else:
        if msg.get_content_type() == "text/html":
            html_parts.append(_safe_decode(msg))
        else:
            plain_parts.append(_safe_decode(msg))
    return "\n".join(p for p in plain_parts if p), "\n".join(h for h in html_parts if h)


def _strip_html(html):
    text = re.sub(r"<(script|style)[^>]*>.*?</\1>", " ", html, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"&nbsp;|&amp;|&lt;|&gt;|&quot;|&#39;", " ", text)
    return re.sub(r"\s+", " ", text).strip()


# --------------------------------------------------------------------------
# 1. Content/NLP thống kê (bổ sung thêm cạnh cleanText/tokenize gốc)
# --------------------------------------------------------------------------

def extract_content_stats_features(plain_text, html_text, subject):
    body_text = plain_text if plain_text.strip() else _strip_html(html_text)
    full_text = f"{subject}\n{body_text}"
    words = re.findall(r"[A-Za-z']+", full_text)
    n_words = max(len(words), 1)
    uppercase_words = sum(1 for w in words if len(w) > 2 and w.isupper())
    return {
        "char_length": len(body_text),
        "word_count": len(words),
        "uppercase_word_ratio": uppercase_words / n_words,
        "exclamation_count": full_text.count("!"),
        "exclamation_in_subject": subject.count("!"),
        "money_symbol_count": len(re.findall(r"[$\u20ac\u00a3\u00a5]", full_text)),
        "digit_ratio": sum(c.isdigit() for c in full_text) / max(len(full_text), 1),
    }, full_text.lower()


# --------------------------------------------------------------------------
# 2. URL-based
# --------------------------------------------------------------------------

def _safe_urlparse(url):
    """
    urlparse() có thể raise ValueError trên URL rác/malformed trong email
    thật (vd chứa dấu '[' khiến parser tưởng nhầm là IPv6 host và bị Python
    reject). Bọc lại để 1 URL lỗi không làm crash toàn bộ pipeline trích
    xuất feature -- chỉ đơn giản coi URL đó là không xác định được domain.
    """
    try:
        return urlparse(url)
    except ValueError:
        return urlparse("")


def extract_url_features(plain_text, html_text):
    combined = f"{plain_text}\n{_strip_html(html_text)}\n{html_text}"
    urls = [u.rstrip('.,);') for u in URL_REGEX.findall(combined)]
    total_urls_found = len(urls)
    urls = urls[:_MAX_URLS_PER_EMAIL]  # tránh email "URL-bombing" làm chậm parser

    ip_urls = shortened = at_symbol = https_count = http_count = suspicious_tld = 0
    lengths, domains = [], set()
    ip_pattern = re.compile(r"^(\d{1,3}\.){3}\d{1,3}")

    for u in urls:
        lengths.append(len(u))
        if "@" in u:
            at_symbol += 1
        if u.lower().startswith("https://"):
            https_count += 1
        elif u.lower().startswith("http://"):
            http_count += 1
        netloc = _safe_urlparse(u if "://" in u else "http://" + u).netloc.lower()
        host = netloc.split(":")[0]
        if host:
            domains.add(host)
            if ip_pattern.match(host):
                ip_urls += 1
            if host in SHORTENER_DOMAINS:
                shortened += 1
            if any(host.endswith(tld) for tld in SUSPICIOUS_TLDS):
                suspicious_tld += 1

    mismatch = 0
    if html_text:
        for i, m in enumerate(re.finditer(
                r'<a[^>]+href=["\']([^"\']+)["\'][^>]*>(.*?)</a>',
                html_text, flags=re.IGNORECASE | re.DOTALL)):
            if i >= _MAX_ANCHORS_PER_EMAIL:
                break
            href, anchor_html = m.group(1), m.group(2)
            anchor_urls = URL_REGEX.findall(_strip_html(anchor_html))
            if anchor_urls:
                href_domain = _safe_urlparse(href if "://" in href else "http://" + href).netloc.lower()
                anchor_domain = _safe_urlparse(
                    anchor_urls[0] if "://" in anchor_urls[0] else "http://" + anchor_urls[0]
                ).netloc.lower()
                if href_domain and anchor_domain and href_domain != anchor_domain:
                    mismatch += 1

    return {
        "num_urls": total_urls_found,
        "num_unique_domains": len(domains),
        "num_ip_urls": ip_urls,
        "num_shortened_urls": shortened,
        "num_at_symbol_urls": at_symbol,
        "num_https_urls": https_count,
        "num_http_urls": http_count,
        "num_suspicious_tld_urls": suspicious_tld,
        "avg_url_length": (sum(lengths) / len(lengths)) if lengths else 0.0,
        "max_url_length": max(lengths) if lengths else 0,
        "num_anchor_href_mismatch": mismatch,
    }


# --------------------------------------------------------------------------
# 3. Header-based
# --------------------------------------------------------------------------

def extract_header_features(msg):
    from_header = msg.get("From", "") or ""
    reply_to = msg.get("Reply-To", "") or ""
    to_header = msg.get("To", "") or ""
    subject = msg.get("Subject", "") or ""
    date_header = msg.get("Date", "")
    received_headers = msg.get_all("Received", []) or []

    display_name, from_addr = parseaddr(from_header)
    from_domain = from_addr.split("@")[-1].lower() if "@" in from_addr else ""
    reply_addr = parseaddr(reply_to)[1]
    reply_domain = reply_addr.split("@")[-1].lower() if "@" in reply_addr else ""
    reply_to_mismatch = bool(reply_domain and from_domain and reply_domain != from_domain)

    display_lower = display_name.lower()
    brand_in_display = any(b in display_lower for b in BRAND_KEYWORDS)
    sender_is_freemail = from_domain in FREEMAIL_DOMAINS
    brand_freemail_mismatch = bool(brand_in_display and sender_is_freemail)

    recipients = getaddresses([to_header])
    num_recipients = len([a for _, a in recipients if a])

    return {
        "from_domain_is_freemail": int(sender_is_freemail),
        "reply_to_differs_from_sender": int(reply_to_mismatch),
        "display_name_brand_freemail_mismatch": int(brand_freemail_mismatch),
        "num_recipients": num_recipients,
        "num_received_hops": len(received_headers),
        "has_date_header": int(bool(date_header)),
        "subject_is_reply_or_fwd": int(bool(re.match(r"^\s*(re|fw|fwd)\s*:", subject, re.IGNORECASE))),
        "from_display_name_length": len(display_name),
    }


# --------------------------------------------------------------------------
# 4. HTML/structure
# --------------------------------------------------------------------------

def extract_html_features(html_text):
    if not html_text:
        return {
            "is_html": 0, "has_form_tag": 0, "has_script_tag": 0,
            "has_hidden_elements": 0, "is_image_only": 0, "html_to_text_ratio": 0.0,
        }
    stripped = _strip_html(html_text)
    has_form = int(bool(re.search(r"<form[\s>]", html_text, re.IGNORECASE)))
    has_script = int(bool(re.search(r"<script[\s>]", html_text, re.IGNORECASE)))
    has_hidden = int(bool(re.search(
        r'display\s*:\s*none|visibility\s*:\s*hidden|opacity\s*:\s*0|hidden=["\']?hidden',
        html_text, re.IGNORECASE)))
    num_images = len(re.findall(r"<img[\s>]", html_text, re.IGNORECASE))
    is_image_only = int(num_images > 0 and len(stripped.split()) < 15)
    ratio = (len(stripped) / len(html_text)) if html_text else 0.0
    return {
        "is_html": 1, "has_form_tag": has_form, "has_script_tag": has_script,
        "has_hidden_elements": has_hidden, "is_image_only": is_image_only,
        "html_to_text_ratio": ratio,
    }


# --------------------------------------------------------------------------
# 5. Attachment
# --------------------------------------------------------------------------

def extract_attachment_features(msg):
    num_attachments = has_executable = has_macro = has_archive = has_double_ext = 0
    if msg.is_multipart():
        for part in msg.walk():
            disp = str(part.get("Content-Disposition") or "")
            filename = part.get_filename()
            if "attachment" in disp or filename:
                num_attachments += 1
                if filename:
                    fname = filename.lower()
                    ext_matches = re.findall(r"\.[a-z0-9]+", fname)
                    if any(fname.endswith(e) for e in EXECUTABLE_EXTENSIONS):
                        has_executable = 1
                    if any(fname.endswith(e) for e in MACRO_EXTENSIONS):
                        has_macro = 1
                    if any(fname.endswith(e) for e in ARCHIVE_EXTENSIONS):
                        has_archive = 1
                    if len(ext_matches) >= 2:
                        has_double_ext = 1
    return {
        "num_attachments": num_attachments,
        "has_executable_attachment": has_executable,
        "has_macro_attachment": has_macro,
        "has_archive_attachment": has_archive,
        "has_double_extension_attachment": has_double_ext,
    }


# --------------------------------------------------------------------------
# 6. Social-engineering
# --------------------------------------------------------------------------

def extract_social_engineering_features(lowered_text):
    def _count(keywords):
        return sum(1 for kw in keywords if kw in lowered_text)
    return {
        "urgency_keyword_count": _count(URGENCY_KEYWORDS),
        "sensitive_info_keyword_count": _count(SENSITIVE_INFO_KEYWORDS),
        "brand_keyword_count": _count(BRAND_KEYWORDS),
        "generic_greeting": int(_count(GENERIC_GREETINGS) > 0),
        "reward_scam_keyword_count": _count(REWARD_KEYWORDS),
    }


# --------------------------------------------------------------------------
# HÀM TỔNG — dùng trong train.py / model_deployment.py
# --------------------------------------------------------------------------

FEATURE_NAMES = [
    "char_length", "word_count", "uppercase_word_ratio", "exclamation_count",
    "exclamation_in_subject", "money_symbol_count", "digit_ratio",
    "num_urls", "num_unique_domains", "num_ip_urls", "num_shortened_urls",
    "num_at_symbol_urls", "num_https_urls", "num_http_urls",
    "num_suspicious_tld_urls", "avg_url_length", "max_url_length",
    "num_anchor_href_mismatch",
    "from_domain_is_freemail", "reply_to_differs_from_sender",
    "display_name_brand_freemail_mismatch", "num_recipients",
    # "num_received_hops" và "has_date_header" ĐÃ LOẠI KHỎI feature set:
    # phân tích feature importance (analyze_feature_importance.py) trên
    # dataset Enron+Nazario+SpamAssassin cho thấy num_received_hops chiếm
    # tới 97-98% importance ở 3/5 model dạng cây -- gần như chắc chắn là
    # shortcut phân biệt "email nội bộ Enron" (hạ tầng mail-relay cố định,
    # 1 công ty, 1 giai đoạn 2000-2002) vs "email được archive lại nhiều
    # năm từ nhiều nguồn khác nhau" (Nazario/SpamAssassin), KHÔNG phải tín
    # hiệu phishing thật. has_date_header cũng nghi vấn tương tự (top 4/5
    # model) vì lý do gần giống: độ đầy đủ của header phụ thuộc cách thu
    # thập/archive dữ liệu, không phải bản chất email.
    "subject_is_reply_or_fwd",
    "from_display_name_length",
    "is_html", "has_form_tag", "has_script_tag", "has_hidden_elements",
    "is_image_only", "html_to_text_ratio",
    "num_attachments", "has_executable_attachment", "has_macro_attachment",
    "has_archive_attachment", "has_double_extension_attachment",
    "urgency_keyword_count", "sensitive_info_keyword_count",
    "brand_keyword_count", "generic_greeting", "reward_scam_keyword_count",
]

# Bộ đếm để theo dõi có bao nhiêu email bị fallback về vector 0 (lỗi trích
# xuất) trên tổng số email đã xử lý -- vì except Exception ở dưới im lặng
# nuốt lỗi (bắt buộc, để 1 email hỏng không sập cả batch train), nếu không
# đếm lại thì sẽ không biết được tỉ lệ email bị "mất" feature là bao nhiêu.
# train.py gọi get_extraction_stats() sau khi .apply() xong để in ra.
_extraction_stats = {"total": 0, "failed": 0}


def get_extraction_stats():
    return dict(_extraction_stats)


def reset_extraction_stats():
    _extraction_stats["total"] = 0
    _extraction_stats["failed"] = 0


def extract_engineered_features(raw_text):
    """
    Nhận vào 1 email RFC-822 THÔ (chưa qua cleanText) dạng str, trả về
    list[float] đúng thứ tự FEATURE_NAMES -- 42 đặc trưng trên cả 6 nhóm
    kỹ thuật. Nếu raw_text không parse được như 1 email hợp lệ (vd dữ liệu
    trong CSV chỉ còn phần content thuần), hàm vẫn không lỗi: các trường
    header/attachment sẽ về 0, chỉ còn content/social-engineering hoạt động
    trên toàn bộ text.

    Vì dữ liệu phishing (Nazario) là nội dung ĐỘC HẠI THẬT, có thể có email
    cố tình chứa HTML/URL bất thường để "bẻ" parser -- plain_text/html_text
    được CẮT BỚT (xem _MAX_TEXT_LEN) trước khi đưa vào các bước regex nặng
    (URL, anchor-mismatch) để tránh 1 email dị thường làm cả batch chạy rất
    chậm; phần bị cắt gần như không ảnh hưởng tới feature vì các đặc điểm
    phishing (URL, form, script...) thường nằm ngay đầu email.
    """
    _extraction_stats["total"] += 1
    try:
        msg = email.message_from_string(str(raw_text))
    except Exception:
        msg = email.message_from_string("")

    try:
        subject = msg.get("Subject", "") or ""
        plain_text, html_text = _get_email_body(msg)

        # Nếu email module không tách được body (vd text đã bị làm sạch từ trước,
        # không có header) thì coi toàn bộ raw_text là plain_text.
        if not plain_text and not html_text:
            plain_text = str(raw_text)

        plain_text = plain_text[:_MAX_TEXT_LEN]
        html_text = html_text[:_MAX_TEXT_LEN]

        content_feats, lowered = extract_content_stats_features(plain_text, html_text, subject)
        url_feats = extract_url_features(plain_text, html_text)
        header_feats = extract_header_features(msg)
        html_feats = extract_html_features(html_text)
        attach_feats = extract_attachment_features(msg)
        social_feats = extract_social_engineering_features(lowered)

        merged = {}
        for d in (content_feats, url_feats, header_feats, html_feats, attach_feats, social_feats):
            merged.update(d)

        return [float(merged[name]) for name in FEATURE_NAMES]
    except Exception:
        # 1 email lỗi bất thường (encoding, cấu trúc hỏng...) không được
        # phép làm crash cả batch train trên hàng chục nghìn email -- trả
        # về vector 0, coi như "không trích được đặc trưng" cho email này.
        _extraction_stats["failed"] += 1
        return [0.0] * len(FEATURE_NAMES)


# --------------------------------------------------------------------------
# Khử trùng lặp / gần trùng (chống data leakage)
# --------------------------------------------------------------------------

def compute_dedup_key(raw_text):
    """
    Sinh 1 khoá để phát hiện email TRÙNG LẶP/GẦN TRÙNG -- vấn đề rất phổ
    biến trong Enron corpus: 1 email được forward cho nhiều người, nằm rải
    rác ở nhiều mailbox khác nhau (allen-p/, arora-h/...), chỉ khác header
    (To, Message-ID, Date, X-Folder...) còn nội dung BODY giống hệt hoặc
    gần hệt nhau.

    Nếu không xử lý, train_test_split ngẫu nhiên rất dễ chia 2 bản gần-trùng
    của CÙNG 1 email vào cả train lẫn test -- model coi như đã "nhìn thấy"
    gần như nguyên văn đáp án của tập test ngay trong lúc train (data
    leakage), khiến điểm test bị thổi phồng giả tạo, không phản ánh đúng
    khả năng tổng quát hoá thật của model.

    Cách làm: chỉ lấy phần BODY (bỏ qua header, vì header luôn khác nhau
    giữa các bản forward), chuẩn hoá khoảng trắng + chữ thường, rồi hash
    MD5 lại để so sánh nhanh trên tập dữ liệu lớn (500K+ email) mà không
    cần so khớp string trực tiếp (chậm).
    """
    try:
        msg = email.message_from_string(str(raw_text))
    except Exception:
        msg = email.message_from_string("")

    try:
        plain_text, html_text = _get_email_body(msg)
        body = plain_text if plain_text.strip() else _strip_html(html_text)
        if not body.strip():
            body = str(raw_text)
    except Exception:
        body = str(raw_text)

    normalized = re.sub(r'\s+', ' ', body).strip().lower()
    return hashlib.md5(normalized.encode('utf-8', errors='replace')).hexdigest()
