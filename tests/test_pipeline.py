import csv
import hashlib
import json
import mailbox
from pathlib import Path
import sqlite3
import tempfile
import unittest
from contextlib import closing
from unittest.mock import patch

import numpy as np
import pandas as pd
from email.message import EmailMessage

import load_data
from dataset_audit import record_id
from pipeline_cache import PreprocessingCache
from preprocess import cleanText, remove_stopwords, group_email_variants, compute_dedup_key, _simhash, URL_REGEX
from specialists import prepare_specialists, train_specialists, run_specialists, evaluate_specialists, BRANCHES
from train import load_training_emails, select_label_scope, prepare_data


def email_bytes(index, positive=False):
    msg = EmailMessage()
    msg['Subject'] = f'Notice {index}'
    msg['From'] = ('Account <support@gmail.com>' if positive else 'Colleague <colleague@example.org>')
    msg['To'] = 'recipient@example.org'
    msg['Reply-To'] = 'other@example.invalid' if positive else 'colleague@example.org'
    # Short unique bodies avoid accidentally forming one giant near-duplicate group.
    msg.set_content(f'Message identifier {index}: please read https://example.invalid/{index}')
    msg.add_alternative(f'<html><body>Message identifier {index} '
                        f'<a href="https://example.invalid/{index}">Open</a>'
                        + ('<script>const marker=1;</script>' if index % 3 == 0 else '')
                        + '</body></html>', subtype='html')
    msg.add_attachment(f'Harmless test fixture {index}'.encode(), maintype='application',
                       subtype='octet-stream', filename=f'note-{index}.txt')
    return msg.as_bytes()


def training_frame(count=180):
    rows = []
    for index in range(count):
        location = f'synthetic:message:{index}'
        positive = bool(index % 2)
        rows.append({'Text': email_bytes(index, positive), 'Class': int(positive),
                     'Source': 'synthetic', 'Location': location,
                     'LabelKind': 'phishing' if positive else 'legitimate',
                     'OriginalLabel': str(int(positive)), 'LabelBasis': 'synthetic test fixture',
                     'RecordId': record_id(location)})
    return pd.DataFrame(rows)


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def test_ingestion_accounts_multiline_malformed_unknown_hidden_and_mbox(self):
        dataset = self.root / 'Dataset'
        (dataset / 'phishing').mkdir(parents=True)
        (dataset / 'enron' / '.hidden').mkdir(parents=True)
        (dataset / 'enron' / '.hidden' / 'mail.md').write_bytes(email_bytes(1))
        (dataset / 'enron' / 'unknown').write_bytes(b'not a MIME email')
        (dataset / 'enron' / 'empty').write_bytes(b'')
        (dataset / 'enron' / 'README.md').write_text('Dataset documentation')
        box = mailbox.mbox(str(dataset / 'phishing' / 'phishing0.mbox'), create=True)
        for index in range(4):
            box.add(email_bytes(10 + index, True))
        box.close()
        with (dataset / 'phishing' / 'extra.csv').open('w', newline='', encoding='utf-8') as handle:
            writer = csv.writer(handle)
            writer.writerow(['subject', 'body', 'label'])
            writer.writerow(['Subject', 'A multiline\nbody', 'phishing'])
            writer.writerow(['Bad', 'alignment', 'phishing', 'extra field'])
            writer.writerow(['', '', 'ham'])
        with (dataset / 'phishing' / 'vectors.csv').open('w', newline='') as handle:
            handle.write('0,label\n1,0\n2,1\n')
        output = self.root / 'load'
        data = load_training_emails(output, dataset)
        self.assertEqual(len(data), 7)
        self.assertEqual(int(data.Class.isna().sum()), 1)
        summary = json.loads((output / 'load_summary.json').read_text())
        self.assertEqual(summary['status'], 'complete')
        self.assertEqual(summary['record_outcomes'], {'loaded': 7, 'no_email_content': 2, 'quarantined': 3})
        self.assertEqual(summary['file_outcomes'], {'complete': 6, 'metadata': 1})
        with closing(sqlite3.connect(output / 'ingestion.sqlite')) as db:
            quarantined = db.execute("SELECT rejected_payload FROM records WHERE status='quarantined'").fetchall()
        self.assertTrue(all(value[0] is not None for value in quarantined))

    def test_unreadable_archive_stops_instead_of_silently_losing_file(self):
        dataset = self.root / 'Dataset'
        (dataset / 'phishing').mkdir(parents=True)
        (dataset / 'enron').mkdir()
        (dataset / 'phishing' / 'not_extracted.zip').write_bytes(b'PK fixture')
        with self.assertRaises(RuntimeError):
            load_training_emails(self.root / 'load', dataset)
        summary = json.loads((self.root / 'load/load_summary.json').read_text())
        self.assertEqual(summary['status'], 'incomplete')
        self.assertEqual(summary['file_outcomes']['failed'], 1)

    def test_negations_are_kept_in_doc2vec_text(self):
        for phrase in ("Don't share", "Do not share", "You can't share", "You won’t share", "It isn't safe"):
            msg = EmailMessage(); msg.set_content(phrase)
            self.assertIn('not', remove_stopwords(cleanText(msg.as_bytes())).split())

    def test_parallel_cache_preserves_order_and_matches_uncached_features(self):
        frame = training_frame(12)
        frame.loc[len(frame)] = frame.iloc[0]
        with PreprocessingCache(self.root / 'cache', workers=1, batch_size=5) as cache:
            sequential = cache.prepare(frame, 'identity', self.root)
            sequential = cache.prepare(sequential, 'features', self.root)
        with PreprocessingCache(self.root / 'other-cache', workers=2, batch_size=5) as cache:
            parallel = cache.prepare(frame, 'identity', self.root)
            parallel = cache.prepare(parallel, 'features', self.root)
        for field in ('RecordId', '_features', '_dedup_key', '_semantic_text', '_clean_text', '_html_features'):
            self.assertEqual(sequential[field].to_list(), parallel[field].to_list())
        with PreprocessingCache(self.root / 'cache', workers=1) as cache:
            warm = cache.prepare(frame, 'features', self.root)
        self.assertEqual(len(warm), len(frame))
        stats = json.loads((self.root / 'preprocessing_features.json').read_text())
        self.assertEqual(stats['computed'], 0)
        self.assertEqual(stats['cache_hits'], len(frame))

    def test_processing_error_is_recorded_and_stops(self):
        frame = training_frame(3)
        with PreprocessingCache(self.root / 'cache', workers=1) as cache:
            with patch('pipeline_cache._process', return_value={'_error': 'fixture failure'}):
                with self.assertRaises(RuntimeError):
                    cache.prepare(frame, 'identity', self.root)
        errors = pd.read_csv(self.root / 'preprocessing_identity_errors.csv')
        self.assertEqual(set(errors.RecordId), set(frame.RecordId))

    def test_shared_attachment_stays_in_one_group_despite_different_bodies(self):
        messages = []
        for body in ('First independent body', 'Completely unrelated second message'):
            msg = EmailMessage(); msg['Subject'] = body; msg.set_content(body)
            msg.add_attachment(b'identical harmless fixture payload', maintype='application',
                               subtype='octet-stream', filename='shared.txt')
            messages.append(msg.as_bytes())
        frame = pd.DataFrame({'Text': messages})
        with PreprocessingCache(self.root / 'cache', workers=1) as cache:
            frame = cache.prepare(frame, 'identity', self.root)
        self.assertNotEqual(frame._dedup_key.iloc[0], frame._dedup_key.iloc[1])
        grouped = group_email_variants(frame, group_attachments=True)
        self.assertEqual(grouped[0], grouped[1])

    def test_vectorized_simhash_matches_reference_bit_voting(self):
        import re
        for length in (0, 19, 20, 50, 400, 5000):
            text = ' '.join(hashlib.sha256(str(index).encode()).hexdigest() for index in range(length))
            normalized = re.sub(r'\d+', '0', URL_REGEX.sub(' URL ', text))
            words = re.findall(r'\w+', normalized)
            if len(words) < 20:
                expected = None
            else:
                votes = [0] * 64
                for shingle in {' '.join(words[i:i + 3]) for i in range(len(words) - 2)}:
                    value = int.from_bytes(hashlib.blake2b(shingle.encode(), digest_size=8).digest(), 'big')
                    for bit in range(64):
                        votes[bit] += 1 if value & (1 << bit) else -1
                expected = sum(1 << bit for bit, vote in enumerate(votes) if vote >= 0)
            self.assertEqual(_simhash(text), expected)

    def test_every_loaded_row_has_selection_disposition_and_empty_conflict_report(self):
        frame = training_frame(6)
        duplicate = frame.iloc[0].copy(); duplicate.RecordId = record_id('duplicate')
        conflict = frame.iloc[1].copy(); conflict.Class = 0; conflict.LabelKind = 'legitimate'; conflict.RecordId = record_id('conflict')
        unknown = frame.iloc[2].copy(); unknown.Class = np.nan; unknown.RecordId = record_id('unknown')
        spam = frame.iloc[3].copy(); spam.Text = email_bytes(999); spam.LabelKind = 'spam'; spam.RecordId = record_id('spam')
        frame = pd.concat([frame, pd.DataFrame([duplicate, conflict, unknown, spam])], ignore_index=True)
        selected = select_label_scope(frame, 'phishing', self.root)
        audit = pd.read_csv(self.root / 'training_dispositions.csv')
        self.assertEqual(len(audit), len(frame))
        self.assertEqual(audit.RecordId.nunique(), len(frame))
        self.assertEqual(len(selected), int(audit.Disposition.eq('selected').sum()))
        self.assertEqual(set(audit.Disposition), {'selected', 'duplicate_content', 'conflicting_labels',
                                                'unknown_label_or_missing_content', 'outside_label_scope'})
        select_label_scope(training_frame(4), 'phishing', self.root)
        self.assertEqual(len(pd.read_csv(self.root / 'label_conflicts.csv')), 0)

    def test_specialists_use_only_own_features_and_do_not_invent_malware_labels(self):
        frame = training_frame()
        parts, settings = prepare_specialists(frame, self.root, self.root / 'cache', workers=1)
        manifest = train_specialists(parts, self.root, preprocessing=settings, train_text=False)
        self.assertEqual(manifest['status'], 'partial')
        self.assertEqual(manifest['branches']['text']['status'], 'ready_to_train')
        self.assertEqual(manifest['branches']['html']['status'], 'awaiting_labels')
        self.assertEqual(manifest['branches']['attachment']['status'], 'awaiting_labels')
        self.assertEqual(manifest['branches']['header']['status'], 'trained')
        self.assertEqual(manifest['branches']['url']['status'], 'trained')
        recomputed = evaluate_specialists(self.root)
        self.assertEqual(set(recomputed.Branch), {'header', 'url'})
        original = pd.read_csv(self.root / 'specialist_evaluation.csv', float_precision='round_trip')
        for column in ('Precision', 'Recall', 'F1-score', 'TN', 'FP', 'FN', 'TP'):
            self.assertTrue(np.allclose(recomputed[column], original[column]))
        self.assertEqual(len(BRANCHES), 5)
        for name in ('header', 'url'):
            with (self.root / f'{name}.pkl').open('rb') as f:
                import pickle
                model = pickle.load(f)
            self.assertEqual(model.n_features_in_, len(BRANCHES[name]['features']))
        from predict_specialists import EmailSpecialists
        prediction = EmailSpecialists(self.root).assess(email_bytes(1000))
        self.assertIn('html', prediction['unassessed_branches'])
        self.assertEqual(prediction['malware_alerts'], [])
        self.assertEqual(set(prediction['branches']), set(BRANCHES))
        empty_msg = EmailMessage(); empty_msg.set_content('Plain note without URLs')
        unavailable = EmailSpecialists(self.root).assess(empty_msg.as_bytes())
        for name in ('header', 'url', 'html', 'attachment'):
            self.assertEqual(unavailable['branches'][name]['status'], 'no_applicable_input')
        combined = pd.concat(parts.values())
        for left, right in (('train','validation'), ('train','test'), ('validation','test')):
            self.assertTrue(set(parts[left]._group).isdisjoint(parts[right]._group))
        self.assertEqual(set(combined.RecordId), set(frame.RecordId))
        # Explicit synthetic target differs from the email phishing label.
        labels = combined[['RecordId']].copy()
        labels['html_malicious'] = [int(int(r.split(':')[-1]) % 3 == 0) for r in combined.Location]
        labels['attachment_malicious'] = [int(int(r.split(':')[-1]) % 4 == 0) for r in combined.Location]
        labels['label_basis'] = 'Synthetic test labels; not malware ground truth'
        label_path = self.root / 'fixture_labels.csv'; labels.to_csv(label_path, index=False)
        trained = self.root / 'labelled'; trained.mkdir()
        full = train_specialists(parts, trained, branch_labels=label_path, preprocessing=settings, train_text=False)
        for name in ('html', 'attachment'):
            self.assertEqual(full['branches'][name]['status'], 'trained')
            observed = pd.read_csv(trained / f'{name}_test_scores.csv').set_index('RecordId').Label
            expected = labels.set_index('RecordId')[BRANCHES[name]['target']]
            self.assertEqual(observed.to_dict(), expected.loc[observed.index].to_dict())
        resumed = run_specialists(self.root / 'resumed', resume=self.root, prepare_only=True)
        self.assertEqual(resumed['status'], 'partial')

    def test_baseline_preparation_still_has_all_rows_and_original_dimensions(self):
        frame = training_frame(60)
        arrays, _, _, settings = prepare_data(frame, self.root, epochs=2,
            cache_dir=self.root / 'cache', workers=1)
        self.assertEqual(sum(len(y) for _, y in arrays.values()), len(frame))
        for X, y in arrays.values():
            self.assertEqual(X.shape, (len(y), 153))
            self.assertTrue(np.isfinite(X).all())

    def test_local_tiny_bert_training_and_inference_without_download(self):
        from transformers import BertConfig, BertForSequenceClassification, BertTokenizerFast
        import torch
        torch.set_num_threads(2)
        checkpoint = self.root / 'tiny_initial'; checkpoint.mkdir()
        vocab = ['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]', 'message', 'notice',
                 'identifier', 'please', 'read', 'open', 'https', 'example', 'invalid']
        (checkpoint / 'vocab.txt').write_text('\n'.join(vocab), encoding='utf-8')
        tokenizer = BertTokenizerFast(vocab_file=str(checkpoint / 'vocab.txt'), do_lower_case=True)
        tokenizer.save_pretrained(checkpoint)
        model = BertForSequenceClassification(BertConfig(vocab_size=len(vocab), hidden_size=16,
            num_hidden_layers=1, num_attention_heads=2, intermediate_size=32,
            max_position_embeddings=512, num_labels=2))
        model.save_pretrained(checkpoint, safe_serialization=True)
        parts, settings = prepare_specialists(training_frame(40), self.root, self.root / 'cache', workers=1)
        manifest = train_specialists(parts, self.root, preprocessing=settings,
            bert_model_name=str(checkpoint), bert_epochs=1)
        self.assertEqual(manifest['branches']['text']['status'], 'trained')
        from predict_specialists import EmailSpecialists
        result = EmailSpecialists(self.root).assess(email_bytes(1001, True))
        self.assertEqual(result['branches']['text']['status'], 'assessed')
        self.assertTrue(0 <= result['branches']['text']['score'] <= 1)


if __name__ == '__main__':
    unittest.main()
