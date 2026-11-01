"""Final model review for baseline_v4. Selection uses validation only; test is reported as confirmation."""
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score, brier_score_loss, log_loss

ROOT = Path(r'D:\HUB\IAP491t_backups')
RUN, FEAT = ROOT / 'runs' / 'baseline_v4', ROOT / 'runs' / 'features_v4'
OUT = Path(sys.argv[1])
S = np.load(OUT / 'scores.npz')
timing = json.loads((OUT / 'timing.json').read_text())
manifest = json.loads((RUN / 'dumped_models/classification/run_manifest.json').read_text(encoding='utf-8'))
names = list(manifest['models'])
fn = manifest['settings']['feature_names']
REAL = ('phishing', 'malspam')
B = 300
rng = np.random.default_rng(1)


def load(split):
    m = pd.read_csv(RUN / f'{split}_metadata.csv', usecols=['Source', 'LabelKind', 'Class', '_group'])
    raw = np.load(FEAT / f'features_{split}.npy', mmap_mode='r')
    y = np.load(RUN / f'data_y_{split}.npy')
    assert np.array_equal(m.Class.to_numpy(), y) and np.array_equal(np.load(FEAT / f'labels_{split}.npy'), y)
    m['html'] = raw[:, fn.index('is_html')] > 0.5
    m['attach'] = raw[:, fn.index('num_attachments')] > 0.5
    m['links'] = raw[:, fn.index('num_urls')] > 0.5
    return m, y


def fpr_threshold(neg, target, w=None):
    """Smallest cut with weighted FPR <= target when predicting score > cut."""
    order = np.argsort(-neg, kind='stable')
    s = neg[order]
    cw = np.cumsum(np.ones_like(s) if w is None else w[order])
    k = np.searchsorted(cw / cw[-1], target, side='right')
    return s[min(k, len(s) - 1)]


def rate(mask_pred, mask):
    return float(mask_pred[mask].mean()) if mask.any() else np.nan


def ece(y, p, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    return float(sum(abs(y[idx == b].mean() - p[idx == b].mean()) * (idx == b).mean()
                     for b in range(bins) if (idx == b).any()))


mv, yv = load('validation')
mt, yt = load('test')
neg_v, real_v = yv == 0, mv.LabelKind.isin(REAL).to_numpy()
legit_src = [s for s, c in mv[neg_v].Source.value_counts().items() if c >= 300]

rows, preds05 = [], {}
for n in names:
    sv, st, sr = S[f'{n}|validation'], S[f'{n}|test'], S[f'{n}|train_sample']
    r = {'Model': n, 'val_AP': average_precision_score(yv, sv), 'val_AUC': roc_auc_score(yv, sv),
         'val_AP_real': average_precision_score(yv[neg_v | real_v], sv[neg_v | real_v])}
    for target in (0.001, 0.005, 0.01):
        cut = fpr_threshold(sv[neg_v], target)
        pv = sv > cut
        tag = f'@{target * 100:g}%'
        r[f'R_all{tag}'] = rate(pv, yv == 1)
        r[f'R_real{tag}'] = rate(pv, real_v)
        if target == 0.005:
            preds05[n] = pv
            for kind in ('phishing', 'malspam', 'malicious_synthetic'):
                r[f'R_{kind}{tag}'] = rate(pv, (mv.LabelKind == kind).to_numpy())
            r[f'R_real_plaintext{tag}'] = rate(pv, real_v & ~mv.html.to_numpy())
            r[f'FPR_legit_html{tag}'] = rate(pv, neg_v & mv.html.to_numpy())
            r[f'FPR_legit_attach{tag}'] = rate(pv, neg_v & mv.attach.to_numpy())
            src = {s: rate(pv, neg_v & (mv.Source == s).to_numpy()) for s in legit_src}
            worst = max(src, key=src.get)
            r[f'FPR_worst_src{tag}'], r['worst_src'] = src[worst], worst
            # Test confirmation with the validation cut.
            pt = st > cut
            r['test_FPR@val-cut0.5%'] = rate(pt, yt == 0)
            r['test_R_real@val-cut0.5%'] = rate(pt, mt.LabelKind.isin(REAL).to_numpy())
            r['test_FPR_html@val-cut0.5%'] = rate(pt, (yt == 0) & mt.html.to_numpy())
    # Deployed operating point (precision >= 0.9 on validation).
    thr = manifest['models'][n]['threshold']
    r['deployed_thr'] = thr
    r['deployed_FPR_legit_html'] = rate(sv >= thr, neg_v & mv.html.to_numpy())
    r['val_brier'] = brier_score_loss(yv, sv)
    r['val_ECE'] = ece(yv, sv)
    ytr = S['y_train_sample']
    r['train_AP'] = average_precision_score(ytr, sr)
    r['AP_gap_train_val'] = r['train_AP'] - r['val_AP']
    r['logloss_train'] = log_loss(ytr, np.clip(sr, 1e-7, 1 - 1e-7))
    r['logloss_val'] = log_loss(yv, np.clip(sv, 1e-7, 1 - 1e-7))
    r['test_AP'] = average_precision_score(yt, st)
    r.update(timing[n])
    rows.append(r)
res = pd.DataFrame(rows)

# Grouped paired bootstrap on validation: AP and real recall @ FPR 0.5%.
codes = pd.factorize(mv._group)[0]
G = codes.max() + 1
boot_ap, boot_rr = np.zeros((B, len(names))), np.zeros((B, len(names)))
for b in range(B):
    w = np.bincount(rng.integers(0, G, G), minlength=G)[codes].astype(float)
    for j, n in enumerate(names):
        sv = S[f'{n}|validation']
        boot_ap[b, j] = average_precision_score(yv, sv, sample_weight=w)
        cut = fpr_threshold(sv[neg_v], 0.005, w[neg_v])
        boot_rr[b, j] = np.average(sv[real_v] > cut, weights=w[real_v])
for arr, key in ((boot_ap, 'val_AP'), (boot_rr, 'R_real@0.5%')):
    res[f'{key}_lo'] = np.percentile(arr, 2.5, axis=0)
    res[f'{key}_hi'] = np.percentile(arr, 97.5, axis=0)
    best = int(np.argmax(arr.mean(axis=0)))
    d = arr - arr[:, [best]]
    res[f'{key}_P(worse than {names[best]})'] = (d < 0).mean(axis=0)

# Error overlap at FPR 0.5% (validation): Jaccard of misclassified rows between models.
err = {n: preds05[n] != (yv == 1) for n in names}
jac = pd.DataFrame([[ (err[a] & err[b]).sum() / max(1, (err[a] | err[b]).sum()) for b in names] for a in names],
                   index=names, columns=names)

res.to_csv(OUT / 'final_review.csv', index=False)
jac.to_csv(OUT / 'error_jaccard.csv')
np.savez(OUT / 'bootstrap.npz', ap=boot_ap, rr=boot_rr, names=np.array(names))
pd.set_option('display.width', 250, 'display.max_columns', 60, 'display.float_format', '{:.4f}'.format)
print(res.set_index('Model').T.to_string())
print('\nError Jaccard @FPR0.5% (validation)\n', jac.round(3).to_string())
print('\nval legit html rows:', int((neg_v & mv.html).sum()), ' real pos:', int(real_v.sum()),
      ' real plaintext pos:', int((real_v & ~mv.html).sum()))
