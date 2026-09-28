"""Read raw emails, extensionless mboxes and labelled CSVs.

Keep the existing loader API. Text is bytes for raw messages so MIME decoding
can use the original charset; CSV rows are rebuilt as valid MIME messages.
CSV numeric labels are preserved (0/1); they must match the dataset's label
convention. A spam label is not proof of phishing.
"""
import csv
import mailbox
import os
import re
import sys
from collections import Counter
from email.mime.text import MIMEText
from email import policy
from pathlib import Path

import pandas as pd

DATASET_DIR = Path(__file__).resolve().parent / "Dataset"
SOURCE_COUNTS = Counter()
SOURCE_ERRORS = Counter()
SOURCE_ROW_ISSUES = []
# Preserve the project's existing broad target: class 1 includes spam.
# Set False if training specifically on verified phishing rather than spam.
INCLUDE_SPAM = True
LABELS = {"0": 0, "1": 1, "ham": 0, "legit": 0, "legitimate": 0,
          "safe email": 0, "phishing": 1, "phishing email": 1}


def reset_source_stats():
    SOURCE_COUNTS.clear()
    SOURCE_ERRORS.clear()
    SOURCE_ROW_ISSUES.clear()


def _win_long_path(path):
    path = os.fspath(path)
    if os.name != "nt":
        return path
    if not os.path.isabs(path):
        path = os.path.join(os.getcwd(), path)
    path = os.path.normpath(path)
    prefix = chr(92) * 2 + "?" + chr(92)
    if path.startswith(prefix):
        return path
    if path.startswith(chr(92) * 2):
        return prefix + "UNC" + chr(92) + path[2:]
    return prefix + path


def _open_text_robust(path):
    # Open the exact Windows filename first, preserving Enron's trailing dots.
    try:
        with open(_win_long_path(path), "rb") as handle:
            return handle.read(), None, None
    except OSError as exc:
        return None, type(exc).__name__, str(exc)


def _looks_like_mbox(path):
    with open(_win_long_path(path), "rb") as handle:
        return handle.read(5) == b"From "


def _try_split_mbox(path):
    if not (_looks_like_mbox(path) or str(path).lower().endswith(".mbox")):
        return []
    box = mailbox.mbox(_win_long_path(path), create=False)
    try:
        messages = [box.get_bytes(key) for key in box.iterkeys()]
    finally:
        box.close()
    if not messages:
        raise ValueError("Declared mbox contains no messages")
    return messages


def _classify_spamassassin_label(dirpath, spamassassin_root):
    parts = os.path.relpath(dirpath, spamassassin_root).lower().split(os.sep)
    for part in reversed(parts):
        if part in {"easy_ham", "easy_ham_2", "hard_ham", "ham", "legitimate"}:
            return 0
        if part in {"spam", "spam_2", "phishing"}:
            return 1
    return None


def _read_csv_emails(path):
    limit = sys.maxsize
    while True:
        try:
            csv.field_size_limit(limit)
            break
        except OverflowError:
            limit //= 10
    with open(_win_long_path(path), encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        names = reader.fieldnames or []
        columns = {name.strip().lower(): name for name in names}
        if len(columns) != len(names):
            raise ValueError("Duplicate CSV column names")
        def column(*aliases):
            return next((columns[a] for a in aliases if a in columns), None)
        label_col = column("label", "class", "email type")
        raw_col = column("raw_email", "email text")
        body_col = column("body", "text", "content")
        subject_col = column("subject")
        if not label_col or not (raw_col or body_col or subject_col):
            SOURCE_COUNTS["csv_vector_or_unsupported"] += 1
            print(f"  Skip CSV without labelled email content: {path}")
            return
        for number, row in enumerate(reader, 2):
            def quarantine(reason):
                SOURCE_COUNTS["csv_rows_quarantined"] += 1
                SOURCE_ROW_ISSUES.append({"file": os.fspath(path), "record": number,
                                          "ending_line": reader.line_num, "reason": reason})

            if None in row or any(value is None for value in row.values()):
                quarantine("CSV field count does not match header; cannot safely infer missing fields")
                continue
            value = (row.get(label_col) or "").strip().lower()
            label = LABELS.get(value)
            if value == "spam" and INCLUDE_SPAM:
                label = 1
            if label is None:
                quarantine("Missing or unsupported label")
                continue
            if raw_col:
                text = row.get(raw_col) or ""
                if text.strip():
                    yield text, label
                else:
                    SOURCE_COUNTS["empty"] += 1
                continue
            body = row.get(body_col, "") or ""
            subject = row.get(subject_col, "") or ""
            if not (body.strip() or subject.strip()):
                SOURCE_COUNTS["empty"] += 1
                continue
            # compat32 stores source headers without the strict address registry.
            # Corpus addresses may be malformed or have non-ASCII local parts.
            # MIMEText encodes the body and Unicode headers without losing rows.
            subtype = "html" if re.search(r"<\s*(html|body|a|p|div|table|form|img)\b", body, re.I) else "plain"
            msg = MIMEText(body, _subtype=subtype, _charset="utf-8", policy=policy.compat32)
            for header, aliases in (("Subject", ("subject",)), ("From", ("sender", "from")),
                                    ("To", ("receiver", "to")), ("Date", ("date",)),
                                    ("Reply-To", ("reply_to", "reply-to"))):
                value = row.get(column(*aliases), "") or ""
                if value:
                    msg[header] = " ".join(value.splitlines())
            yield msg.as_bytes(), label


def _read_rows(root_dir, default_label, classify=False, optional=False):
    root = _win_long_path(root_dir)
    rows = []
    if not os.path.isdir(root):
        print(f"  Missing dataset directory: {root_dir}")
        if not optional:
            SOURCE_ERRORS["missing_directory"] += 1
        return rows
    def onerror(exc):
        SOURCE_ERRORS[type(exc).__name__] += 1
        print(f"  Directory read failed: {exc}")
    total = 0
    for directory, dirs, files in os.walk(root, onerror=onerror):
        dirs.sort()
        for name in sorted(files):
            low = name.lower()
            if low.startswith(".") or low.startswith("readme") or low == "cmds" or low.endswith(".md"):
                SOURCE_COUNTS["metadata_skipped"] += 1
                continue
            path = os.path.join(directory, name)
            total += 1
            if total % 5000 == 0:
                print(f"  {root_dir}: scanned {total} files; loaded {len(rows)} emails", flush=True)
            try:
                if low.endswith((".zip", ".gz", ".bz2", ".xz", ".tar", ".7z")):
                    raise ValueError("Extract archive before training")
                if low.endswith(".csv"):
                    found = list(_read_csv_emails(path))
                else:
                    nested_label = _classify_spamassassin_label(directory, root)
                    label = nested_label if nested_label is not None else default_label
                    if label is None:
                        SOURCE_ERRORS["unknown_folder_label"] += 1
                        continue
                    if classify and label == 1 and not INCLUDE_SPAM:
                        SOURCE_COUNTS["spam_excluded"] += 1
                        continue
                    messages = _try_split_mbox(path)
                    if not messages:
                        content, error, detail = _open_text_robust(path)
                        if error:
                            raise OSError(detail)
                        messages = [content]
                    found = [(message, label) for message in messages if message.strip()]
                    SOURCE_COUNTS["empty"] += len(messages) - len(found)
                SOURCE_COUNTS[str(root_dir)] += len(found)
                rows.extend(found)
            except (OSError, UnicodeError, csv.Error, ValueError, mailbox.Error) as exc:
                SOURCE_ERRORS[type(exc).__name__] += 1
                print(f"  Read failed: {path}: {exc}")
    print(f"  {root_dir}: {total} files -> {len(rows)} labelled emails", flush=True)
    return rows


def _append_rows(data, rows):
    frames = [data] if not data.empty else []
    if rows:
        frames.append(pd.DataFrame(rows, columns=["Text", "Class"]))
    return (pd.concat(frames, ignore_index=True).dropna(subset=["Text", "Class"])
            if frames else pd.DataFrame(columns=["Text", "Class"]))


def _read_raw_emails(root_dir, verbose=True):
    return [text for text, _ in _read_rows(root_dir, 0)]


def load_data_generic(data, root_dir, label):
    if label not in (0, 1):
        raise ValueError("label must be 0 or 1")
    return _append_rows(data, _read_rows(root_dir, label))


def load_data_phishing(data, phishing_dir=None):
    return _append_rows(data, _read_rows(phishing_dir or DATASET_DIR / "phishing", 1))


def load_data_spamassassin(data_phishing, data_legit, spamassassin_dir=None):
    rows = _read_rows(spamassassin_dir or DATASET_DIR / "spamassassin", None,
                      classify=True, optional=True)
    return (_append_rows(data_phishing, [(text, label) for text, label in rows if label == 1]),
            _append_rows(data_legit, [(text, label) for text, label in rows if label == 0]))


def load_data_enron(data_enron, type_of_data, enron_dir=None, n_phishing=None, seed=42):
    data = _append_rows(data_enron, _read_rows(enron_dir or DATASET_DIR / "enron", 0))
    if type_of_data == 1 and n_phishing is not None:
        legit, other = data[data.Class == 0], data[data.Class != 0]
        if len(legit) > n_phishing:
            legit = legit.sample(n=n_phishing, random_state=seed)
        data = pd.concat([legit, other], ignore_index=True)
    return data
