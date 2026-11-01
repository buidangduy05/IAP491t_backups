"""Score every saved baseline_v4 classifier on validation, test and a stratified train sample.

Inference only: nothing is fitted. Train rows need Doc2Vec vectors, which were never saved, so they
are re-inferred with the run's own model_dbow + scaler (deterministic, content-seeded) and checked
against saved validation rows first.
"""
import gzip, json, pickle, sys, time
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(r'D:\HUB\IAP491t_backups')
RUN = ROOT / 'runs' / 'baseline_v4'
FEAT = ROOT / 'runs' / 'features_v4'
OUT = Path(sys.argv[1])
sys.path.insert(0, str(ROOT))
from preprocess import infer_document_vector  # noqa: E402

MD = RUN / 'dumped_models' / 'classification'
manifest = json.loads((MD / 'run_manifest.json').read_text(encoding='utf-8'))
EPOCHS = manifest['settings']['inference_epochs']
TRAIN_SAMPLE = 20000
rng = np.random.default_rng(0)

with open(MD / 'model_dbow.pkl', 'rb') as h:
    dbow = pickle.load(h)
with open(MD / 'feature_scaler.pkl', 'rb') as h:
    scaler = pickle.load(h)


def read_texts(path, keep):
    keep, texts = set(int(i) for i in keep), {}
    with gzip.open(path, 'rt', encoding='utf-8') as h:
        for i, line in enumerate(h):
            if i in keep:
                texts[i] = json.loads(line)
    return texts


def rebuild(split, idx):
    feats = np.load(FEAT / f'features_{split}.npy', mmap_mode='r')[idx]
    texts = read_texts(FEAT / f'text_{split}.jsonl.gz', idx)
    doc = np.asarray([infer_document_vector(dbow, texts[int(i)], EPOCHS) for i in idx], dtype=np.float32)
    return np.hstack([doc, scaler.transform(feats).astype(np.float32)])


# 1. Reproduction check on 300 validation rows.
Xv = np.load(RUN / 'data_X_validation.npy')
probe = np.sort(rng.choice(len(Xv), 300, replace=False))
re = rebuild('validation', probe)
a, b = re[:, :100], Xv[probe, :100]
cos = (a * b).sum(1) / np.linalg.norm(a, axis=1) / np.linalg.norm(b, axis=1)
feat = np.abs(re[:, 100:] - Xv[probe, 100:]).max()
print(f'validation rebuild doc cosine min={cos.min():.5f} p1={np.percentile(cos, 1):.5f} feat max|diff|={feat:.1e}', flush=True)
assert np.percentile(cos, 1) > 0.995 and feat < 1e-4, 'cannot reproduce saved vectors; train-gap invalid'

# 2. Stratified train sample (by LabelKind) rebuilt the same way.
tm = pd.read_csv(FEAT / 'train_metadata.csv', usecols=['LabelKind', 'Class'])
frac = TRAIN_SAMPLE / len(tm)
tidx = np.sort(np.concatenate([rng.choice(g.index.to_numpy(), max(1, round(len(g) * frac)), replace=False)
                               for _, g in tm.groupby('LabelKind')]))
t0 = time.perf_counter()
Xt = rebuild('train', tidx)
print(f'train sample {len(tidx)} rebuilt in {time.perf_counter() - t0:.0f}s', flush=True)
yt = tm.Class.to_numpy()[tidx]

splits = {'validation': Xv, 'test': np.load(RUN / 'data_X_test.npy'), 'train_sample': Xt}
scores, timing = {}, {}
for name, cfg in manifest['models'].items():
    path = MD / cfg['file']
    with open(path, 'rb') as h:
        model = pickle.load(h)
    pos = int(np.flatnonzero(model.classes_ == 1)[0])
    for split, X in splits.items():
        t0 = time.perf_counter()
        scores[f'{name}|{split}'] = model.predict_proba(X)[:, pos].astype(np.float64)
        if split == 'validation':
            timing[name] = {'pred_s_per_10k': (time.perf_counter() - t0) / len(X) * 1e4,
                            'size_mb': path.stat().st_size / 2**20}
    print(f'{name}: scored, {timing[name]}', flush=True)
    del model

np.savez_compressed(OUT / 'scores.npz', train_idx=tidx, y_train_sample=yt, **scores)
(OUT / 'timing.json').write_text(json.dumps(timing, indent=1))
print('done')
