# #### Loading data from corpus
#
# Giữ nguyên tên hàm/tham số như repo gốc (load_data_phishing, load_data_enron)
# để train.py / unit-test.py gọi theo đúng API cũ. Điểm khác so với bản gốc:
#   - Bản gốc đọc từ ./Dataset/phishing/*.csv và ./Dataset/enron/*.csv với
#     cột có sẵn 'content' (dữ liệu đã được ai đó tiền xử lý ra CSV).
#   - Ở đây đọc thẳng dữ liệu THÔ mà bạn đang có: CẢ Enron lẫn Nazario đều là
#     raw email KHÔNG ĐUÔI FILE (mỗi file = 1 email, đủ header), nên cả 2 hàm
#     load_data_phishing / load_data_enron cùng dùng chung 1 cách đọc: quét
#     đệ quy (os.walk) toàn bộ cây thư mục, coi MỌI file là 1 raw email.
#     (Nếu sau này bạn có thêm data dạng .mbox gộp nhiều email trong 1 file,
#     hàm _read_raw_emails bên dưới tự nhận diện và tách ra, không cần đổi gì.)
#   - `data.append(...)` (bản gốc) đã bị gỡ khỏi pandas >= 2.0 -> đổi sang
#     pd.concat để chạy được trên môi trường hiện tại.

import os
import mailbox
import pandas as pd
from collections import Counter


def _win_long_path(path):
    """
    Windows mặc định giới hạn đường dẫn 260 ký tự (MAX_PATH) -- cấu trúc
    Enron maildir lồng khá sâu rất dễ vượt qua giới hạn này. Dùng prefix
    "\\\\?\\" để bypass MAX_PATH trên Windows.

    QUAN TRỌNG: KHÔNG dùng os.path.abspath() ở đây để lấy đường dẫn tuyệt
    đối -- trên Windows, os.path.abspath() gọi thẳng Win32 API
    GetFullPathNameW, và API này tự động STRIP dấu chấm/khoảng trắng ở
    CUỐI TÊN FILE (behavior "legacy DOS path" của Windows) -- đúng loại
    file mà Enron corpus đang dùng (tên file "1.", "10.", "100."... có
    dấu chấm cuối!). Nếu dùng abspath(), dấu chấm sẽ mất TRƯỚC KHI kịp
    thêm "\\?\\" vào, khiến "\\?\\" trở nên vô nghĩa (path đúng, tên file
    sai). Ở đây tự ghép cwd + path rồi chuẩn hoá bằng os.path.normpath()
    (hàm string thuần, KHÔNG gọi Win32 API) để giữ nguyên dấu chấm cuối.
    """
    if os.name != 'nt':
        return path
    if not os.path.isabs(path):
        path = os.path.join(os.getcwd(), path)
    normalized = os.path.normpath(path)
    if normalized.startswith('\\\\?\\'):
        return normalized
    if normalized.startswith('\\\\'):  # UNC path (\\server\share\...)
        return '\\\\?\\UNC\\' + normalized.lstrip('\\')
    return '\\\\?\\' + normalized


def _open_text_robust(path):
    """
    Thử mở file theo đường dẫn BÌNH THƯỜNG trước (nhanh, hoạt động đúng
    trong đa số trường hợp); nếu lỗi mới fallback sang extended-length path
    (\\\\?\\...) -- tăng khả năng đọc được file trên nhiều cấu hình ổ đĩa/
    Windows khác nhau thay vì áp \\\\?\\ cứng cho mọi file (1 số kiểu ổ đĩa
    mạng/USB xử lý \\\\?\\ không tốt, và \\\\?\\ đòi hỏi path khớp CHÍNH XÁC
    100%, không tự chuẩn hoá dấu / hay dấu chấm cuối tên file như path
    thường).
    Trả về (content, error_type, error_message) -- error_type=None nếu đọc
    thành công (dù bằng cách nào); nếu cả 2 cách đều lỗi, error_type lấy
    theo lỗi của lần thử BÌNH THƯỜNG (đại diện chính), error_message gộp cả
    2 message để chẩn đoán được chính xác nguyên nhân.
    """
    try:
        with open(path, 'r', encoding='utf-8', errors='replace') as f:
            return f.read(), None, None
    except Exception as e_normal:
        if os.name != 'nt':
            return None, type(e_normal).__name__, str(e_normal)
        try:
            with open(_win_long_path(path), 'r', encoding='utf-8', errors='replace') as f:
                return f.read(), None, None
        except Exception as e_long:
            combined = f"normal: {e_normal} | long-path: {e_long}"
            return None, type(e_normal).__name__, combined


def _read_raw_emails(root_dir, verbose=True):
    """
    Quét đệ quy `root_dir`. Với mỗi file:
      - nếu là 1 mbox gộp nhiều email (nhận diện qua đuôi .mbox) -> tách ra
        từng email con.
      - ngược lại -> coi cả file là 1 raw email (đúng trường hợp Enron/
        Nazario bạn đang dùng: mỗi file không đuôi = 1 email đầy đủ header).
    Trả về list[str] raw email text.

    In kèm thống kê (tổng số file tìm thấy vs. số file đọc lỗi, theo từng
    loại lỗi) để dễ tự chẩn đoán khi số email đọc được thấp bất thường --
    nguyên nhân thường gặp trên Windows là đường dẫn quá dài (xem
    _win_long_path ở trên).
    """
    emails = []
    if not os.path.isdir(root_dir):
        if verbose:
            print(f"  [!] Thư mục '{root_dir}' không tồn tại.")
        return emails

    total_files = 0
    error_counter = Counter()
    error_samples = []  # (path, path_len, error_message) -- vài mẫu đầu để chẩn đoán

    if verbose:
        print(f"  Bắt đầu quét '{root_dir}' ...", flush=True)

    for dirpath, _dirnames, filenames in os.walk(root_dir):
        for fname in filenames:
            if fname.startswith('.'):
                continue
            total_files += 1

            # Progress heartbeat -- Enron thường có 500K+ file, nếu không in
            # gì trong lúc chạy sẽ rất dễ tưởng nhầm là bị treo / không đọc
            # thư mục này. In mỗi 5000 file, flush=True để hiện ngay trên
            # console Windows (không bị buffer).
            if verbose and total_files % 5000 == 0:
                print(f"    ... đã quét {total_files} file, đọc được {len(emails)} email "
                      f"({sum(error_counter.values())} lỗi)", flush=True)

            path = os.path.join(dirpath, fname)

            if fname.lower().endswith('.mbox'):
                try:
                    box = mailbox.mbox(_win_long_path(path))
                    emails.extend(msg.as_string() for msg in box)
                    continue
                except Exception:
                    pass  # rơi xuống đọc như 1 file raw email bình thường

            resolved_path = path
            content, err_type, err_msg = _open_text_robust(path)
            if err_type is not None:
                error_counter[err_type] += 1
                if len(error_samples) < 5:
                    error_samples.append((resolved_path, len(resolved_path), err_msg))
                continue

            if content.strip():
                emails.append(content)

    if verbose:
        print(f"  -> Quét thấy {total_files} file trong '{root_dir}', "
              f"đọc thành công {len(emails)} email.")
        if error_counter:
            detail = ", ".join(f"{k}: {v}" for k, v in error_counter.most_common())
            print(f"  [!] {sum(error_counter.values())} file đọc lỗi ({detail}).")
            print("      Mẫu vài file lỗi đầu tiên (đường dẫn / độ dài / message lỗi gốc):")
            for p, plen, msg in error_samples:
                print(f"        - len={plen}: {p}")
                print(f"          -> {msg}")
            if "OSError" in error_counter or "FileNotFoundError" in error_counter:
                print("      -> Nếu 'len' ở trên < 260 mà vẫn lỗi 'not found', nhiều khả năng KHÔNG "
                      "phải do MAX_PATH -- kiểm tra xem Dataset/ có đang nằm trong thư mục đồng bộ "
                      "OneDrive/Google Drive/Dropbox không (file có thể là 'placeholder' cloud-only, "
                      "chưa thực sự tải về máy). Nếu 'len' > 260, đúng là do MAX_PATH -- thử copy "
                      "Dataset/ ra ổ đĩa gốc (vd C:\\data\\enron) để đường dẫn ngắn lại, hoặc bật "
                      "'Enable Win32 long paths' trong Group Policy / Registry của Windows.")

    return emails


def load_data_generic(data, root_dir, label):
    """
    Hàm tổng quát để nạp thêm BẤT KỲ nguồn dữ liệu nào có cùng định dạng
    (thư mục chứa raw email không đuôi, hoặc file .mbox), khi bạn biết
    CHẮC toàn bộ thư mục chỉ thuộc 1 nhãn duy nhất.

    Với SpamAssassin (có cả ham lẫn spam trộn trong cùng 1 corpus, cần tự
    phân loại theo tên thư mục con), dùng load_data_spamassassin() thay vì
    hàm này -- xem hàm đó bên dưới.
    """
    emails = _read_raw_emails(root_dir)
    print(f"[load_data_generic] Đọc được {len(emails)} email (label={label}) từ '{root_dir}'")

    frames = [data] if not data.empty else []
    if emails:
        frames.append(pd.DataFrame({'Text': emails, 'Class': [label] * len(emails)}))

    data = pd.concat(frames, ignore_index=True) if frames else data
    return data.dropna()


def _classify_spamassassin_label(dirpath, spamassassin_root):
    """
    SpamAssassin public corpus (https://spamassassin.apache.org/old/publiccorpus/)
    tổ chức theo tên thư mục chuẩn: easy_ham, easy_ham_2, hard_ham (email hợp
    lệ) và spam, spam_2 (email rác/độc hại). Tự nhận diện legit (0) hay
    phishing (1) dựa theo tên thư mục con chứa file, không cần khai báo tay
    từng thư mục.
    Trả về 0 (ham/legit), 1 (spam/phishing), hoặc None nếu không nhận diện
    được (vd file nằm ngay dưới root, không có thư mục con nào chứa
    "ham"/"spam" trong tên).
    """
    rel = os.path.relpath(dirpath, spamassassin_root)
    parts = [] if rel == '.' else rel.split(os.sep)
    for part in parts:
        low = part.lower()
        if 'spam' in low:
            return 1
        if 'ham' in low:
            return 0
    return None


def load_data_spamassassin(data_phishing, data_legit, spamassassin_dir='./Dataset/spamassassin/'):
    """
    Đọc toàn bộ SpamAssassin public corpus trong `spamassassin_dir`, TỰ
    PHÂN LOẠI theo tên thư mục con (xem _classify_spamassassin_label):
      - easy_ham/, easy_ham_2/, hard_ham/...  -> gộp vào data_legit  (Class=0)
      - spam/, spam_2/...                      -> gộp vào data_phishing (Class=1)

    Cách dùng trong train.py (gọi SAU load_data_phishing, TRƯỚC
    load_data_enron, để n_phishing truyền cho load_data_enron tính đúng cả
    phần spam của SpamAssassin):

        data = load_data_phishing(data)                      # Nazario
        data, data_enron = load_data_spamassassin(data, data_enron)
        data_enron = load_data_enron(data_enron, BALANCED_MODE, n_phishing=len(data))

    Lưu ý: SpamAssassin gán nhãn "spam" theo nghĩa rác/quảng cáo nói chung,
    không hẳn 100% là "phishing" (lừa đảo đánh cắp thông tin) -- coi spam=1
    theo đúng yêu cầu của bạn, nhưng nghĩa là model sẽ học rộng hơn thành
    "phát hiện email độc hại/không mong muốn nói chung", không thuần chỉ
    phishing như khi chỉ dùng Nazario.
    """
    if not os.path.isdir(spamassassin_dir):
        print(f"  [SpamAssassin] Thư mục '{spamassassin_dir}' không tồn tại, bỏ qua.")
        return data_phishing, data_legit

    print(f"  Bắt đầu quét SpamAssassin '{spamassassin_dir}' ...", flush=True)

    ham_emails, spam_emails = [], []
    total_files = 0
    unknown_count = 0
    error_counter = Counter()

    for dirpath, _dirnames, filenames in os.walk(spamassassin_dir):
        label = _classify_spamassassin_label(dirpath, spamassassin_dir)
        for fname in filenames:
            if fname.startswith('.') or fname.lower() in ('cmds',):
                continue  # SpamAssassin corpus có kèm file 'cmds' không phải email
            total_files += 1

            if total_files % 5000 == 0:
                print(f"    ... đã quét {total_files} file (ham={len(ham_emails)}, "
                      f"spam={len(spam_emails)})", flush=True)

            if label is None:
                unknown_count += 1
                continue

            path = os.path.join(dirpath, fname)
            content, err_type, err_msg = _open_text_robust(path)
            if err_type is not None:
                error_counter[err_type] += 1
                continue
            if not content.strip():
                continue

            (spam_emails if label == 1 else ham_emails).append(content)

    print(f"  -> SpamAssassin: {len(ham_emails)} email ham (legit) + {len(spam_emails)} "
          f"email spam (phishing) / {total_files} file quét được.")
    if unknown_count:
        print(f"  [!] {unknown_count} file bỏ qua vì không xác định được nằm trong thư mục "
              f"ham hay spam (tên thư mục không chứa 'ham'/'spam').")
    if error_counter:
        detail = ", ".join(f"{k}: {v}" for k, v in error_counter.most_common())
        print(f"  [!] {sum(error_counter.values())} file đọc lỗi ({detail}).")

    frames_legit = [data_legit] if not data_legit.empty else []
    if ham_emails:
        frames_legit.append(pd.DataFrame({'Text': ham_emails, 'Class': [0] * len(ham_emails)}))
    data_legit = pd.concat(frames_legit, ignore_index=True) if frames_legit else data_legit
    data_legit = data_legit.dropna()

    frames_phish = [data_phishing] if not data_phishing.empty else []
    if spam_emails:
        frames_phish.append(pd.DataFrame({'Text': spam_emails, 'Class': [1] * len(spam_emails)}))
    data_phishing = pd.concat(frames_phish, ignore_index=True) if frames_phish else data_phishing
    data_phishing = data_phishing.dropna()

    return data_phishing, data_legit


def load_data_phishing(data, phishing_dir='./Dataset/phishing/'):
    """
    Đọc mọi email phishing (Nazario) trong `phishing_dir`, đệ quy, mỗi file
    không đuôi = 1 raw email -> Class=1.
    """
    emails = _read_raw_emails(phishing_dir)
    print(f"[load_data_phishing] Đọc được {len(emails)} email từ '{phishing_dir}'")

    frames = [data] if not data.empty else []
    if emails:
        frames.append(pd.DataFrame({'Text': emails, 'Class': [1] * len(emails)}))

    data = pd.concat(frames, ignore_index=True) if frames else data
    data = data.dropna()
    return data


def load_data_enron(data_enron, type_of_data, enron_dir='./Dataset/enron/', n_phishing=None, seed=42):
    """
    Đọc đệ quy toàn bộ file trong cây thư mục maildir của Enron (mỗi file =
    1 email thô, không đuôi) -> Class=0.

    type_of_data == 0: IMBALANCED -- giữ nguyên toàn bộ (giống Enron thật,
    số email hợp lệ >> số email phishing).
    type_of_data == 1: BALANCED -- downsample email hợp lệ xuống bằng đúng
    `n_phishing` (số email phishing đã load được từ load_data_phishing).
    Khác bản gốc (cắt cứng data_enron.index[1:678]), ở đây lấy mẫu NGẪU
    NHIÊN và tự khớp theo số lượng phishing thật, vì số này phụ thuộc dataset
    Nazario bạn tải về, không cố định.
    """
    emails = _read_raw_emails(enron_dir)
    print(f"[load_data_enron] Đọc được {len(emails)} email từ '{enron_dir}'")

    frames = [data_enron] if not data_enron.empty else []
    if emails:
        frames.append(pd.DataFrame({'Text': emails, 'Class': [0] * len(emails)}))

    data_enron = pd.concat(frames, ignore_index=True) if frames else data_enron
    data_enron = data_enron.dropna()

    if type_of_data == 1 and n_phishing is not None and len(data_enron) > n_phishing:
        data_enron = data_enron.sample(n=n_phishing, random_state=seed).reset_index(drop=True)

    return data_enron
