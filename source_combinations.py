"""Source-combination experiment: train on a subset of corpora, test on every corpus.

Reads a feature export (sandbox.ps1 features / train.py --features-only) and never writes to
it. For each combination of >= 1 phishing family and >= 1 legitimate family it keeps the
export's group-disjoint split, caps every family so no corpus dominates, fits the baseline
pipeline (train.vectorize + train.tune_classifiers) on the combination only and picks
thresholds on the combination's validation rows. Every family's test rows are then scored:
families inside the combination give in-distribution results, the others give transfer to
corpora the model never saw. One fixed seed; uncertainty comes from a group bootstrap over
test rows (near-duplicate groups are resampled whole).

  python source_combinations.py --features runs/features_v7 --output runs/source_combinations_v7
  python source_combinations.py --features runs/features_v7 --output <dir> --sizes 2 --doc2vec none

Re-running with the same --output resumes: finished combinations are skipped.
"""
from itertools import combinations
from pathlib import Path
import argparse
import gzip
import json
import time

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from evaluate import positive_scores, validation_rank, classification_metrics, file_digest
from load_data import NAZARIO_NAMES
from preprocess import BASELINE_FEATURE_NAMES
from train import (SEED, MIN_VALIDATION_PRECISION, DOC2VEC_EPOCHS, FEATURE_EXPORT,
                   vectorize, build_candidates, tune_classifiers)

SPLITS = ('train', 'validation', 'test')
# Class of each family. A family groups files that share an origin, so a held-out corpus is
# never seen through a copy: TREC 2005 ham is Enron mail (4,311 near-duplicate groups with
# the raw maildir in features_v6). EPVME was generated from phishing originals (507 groups
# shared with Nazario); it stays its own family and that link is reported, not hidden.
FAMILIES = {'Nazario': 1, 'Phishing Pot': 1, 'Malspam': 1, 'EPVME': 1,
            'Enron': 0, 'SpamAssassin': 0, 'TREC 2006': 0, 'TREC 2007': 0,
            'CEAS 2008': 0, 'Ling-Spam': 0}
SOURCE_FAMILY = {'phishing_pot': 'Phishing Pot', 'malspam': 'Malspam', 'EPVME': 'EPVME',
                 'enron': 'Enron', 'TREC_05.csv': 'Enron', 'spamassassin': 'SpamAssassin',
                 'TREC_06.csv': 'TREC 2006', 'TREC_07.csv': 'TREC 2007',
                 'CEAS_08.csv': 'CEAS 2008', 'Ling.csv': 'Ling-Spam'}
# Removed from Dataset/ on 2026-11-04 (copies of the raw Enron/SpamAssassin corpora). Older
# exports still hold them; their rows are dropped, but the raw copies that deduplication
# removed in their favour are missing too, so prefer an export made after the removal.
REMOVED_SOURCES = {'Enron.csv', 'SpamAssasin.csv'}
DEFAULT_MODELS = ('XGBoost', 'Logistic Regression')
SUMMARY = 'results_summary.csv'
BY_FAMILY = 'results_by_family.csv'


def source_family(source):
    if source in REMOVED_SOURCES:
        return None
    if source in NAZARIO_NAMES or source.startswith('phishing-'):
        return 'Nazario'
    if source not in SOURCE_FAMILY:
        raise ValueError(f'Unmapped source {source!r}; add it to SOURCE_FAMILY, do not guess')
    return SOURCE_FAMILY[source]


def load_pools(folder, train_cap, eval_cap, need_text):
    """Per split and family, a capped sample of rows: metadata, features and (optionally) text."""
    folder = Path(folder)
    manifest = json.loads((folder / FEATURE_EXPORT).read_text(encoding='utf-8'))
    if manifest.get('kind') != 'baseline_features' or manifest.get('schema') != 1:
        raise ValueError(f'{folder} is not a baseline feature export')
    if manifest['target_mode'] != 'phishing':
        raise ValueError('Families are defined for target mode "phishing" only')
    if manifest['feature_names'] != list(BASELINE_FEATURE_NAMES):
        raise ValueError('Exported feature names differ from this code; re-extract the features')
    for name, expected in manifest['files'].items():
        if need_text or not name.startswith('text_'):
            if file_digest(folder / name) != expected:
                raise RuntimeError(f'Feature export changed after it was written: {name}')
    rng = np.random.default_rng(SEED)
    pools, dropped = {}, {}
    for split in SPLITS:
        meta = pd.read_csv(folder / f'{split}_metadata.csv', usecols=['Source', 'Class', '_group'])
        features = np.load(folder / f'features_{split}.npy', allow_pickle=False)
        labels = np.load(folder / f'labels_{split}.npy', allow_pickle=False)
        if not (len(meta) == len(features) == len(labels)) or not np.array_equal(meta.Class.to_numpy(), labels):
            raise ValueError(f'Feature export split {split} is not aligned')
        meta['Family'] = meta.Source.map(source_family)
        dropped[split] = int(meta.Family.isna().sum())
        cap = train_cap if split == 'train' else eval_cap
        keep = []
        for family, rows in meta.dropna(subset=['Family']).groupby('Family').groups.items():
            if (meta.Class[rows] != FAMILIES[family]).any():
                raise ValueError(f'{family} rows in {split} carry the wrong class')
            rows = np.asarray(rows)
            keep.append(rows if cap is None or len(rows) <= cap else np.sort(rng.choice(rows, cap, replace=False)))
        keep = np.sort(np.concatenate(keep))
        frame = meta.loc[keep, ['Source', 'Family', 'Class', '_group']].reset_index(drop=True)
        frame['Features'] = list(features[keep])
        if need_text:
            wanted, texts = set(keep.tolist()), []
            with gzip.open(folder / f'text_{split}.jsonl.gz', 'rt', encoding='utf-8') as handle:
                for index, line in enumerate(handle):
                    if index in wanted:
                        texts.append(json.loads(line))
            if len(texts) != len(keep):
                raise ValueError(f'text_{split}.jsonl.gz is not aligned with its metadata')
            frame['Text'] = texts
        pools[split] = {family: part.reset_index(drop=True) for family, part in frame.groupby('Family')}
    missing = [(split, family) for split in SPLITS for family in FAMILIES if family not in pools[split]]
    return pools, manifest, dropped, missing


def enumerate_combinations(families, sizes):
    positives = [f for f in families if FAMILIES[f] == 1]
    negatives = [f for f in families if FAMILIES[f] == 0]
    result = []
    for size in sizes:
        for combo in combinations(positives + negatives, size):
            if set(combo) & set(positives) and set(combo) & set(negatives):
                result.append(combo)
    return result


def featurize(frames, preprocessing, doc2vec, epochs):
    if doc2vec == 'per-combination':
        arrays, _, _, _ = vectorize(frames, preprocessing, epochs=epochs)  # Doc2Vec + scaler fit on train only
        return arrays
    scaler = StandardScaler().fit(np.stack(frames['train'].Features.to_list()))  # TRAIN ONLY
    return {name: (scaler.transform(np.stack(frame.Features.to_list())).astype(np.float32),
                   frame.Class.to_numpy(dtype=np.int64)) for name, frame in frames.items()}


def bootstrap_counts(y, predicted, groups, draws, rng):
    """Resample near-duplicate groups whole; return TP, FP, FN, TN per bootstrap draw."""
    codes, inverse = np.unique(groups, return_inverse=True)
    cells = [np.bincount(inverse, weights=((y == a) & (predicted == b)).astype(float), minlength=len(codes))
             for a, b in ((1, 1), (0, 1), (1, 0), (0, 0))]
    picks = rng.integers(0, len(codes), size=(draws, len(codes)))
    return [cell[picks].sum(axis=1) for cell in cells]


def interval(values):
    values = values[np.isfinite(values)]
    return (float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))) if len(values) else (None, None)


def ratio(numerator, denominator):
    with np.errstate(divide='ignore', invalid='ignore'):
        return numerator / denominator


def score_combination(combo, pools, preprocessing, models, doc2vec, epochs, draws, precision_target):
    families = list(pools['test'])
    frames = {'train': pd.concat([pools['train'][f] for f in combo], ignore_index=True),
              'validation': pd.concat([pools['validation'][f] for f in combo], ignore_index=True),
              'test': pd.concat([pools['test'][f] for f in families], ignore_index=True)}
    test = frames['test'][['Family', 'Class', '_group']]
    arrays = featurize(frames, preprocessing, doc2vec, epochs)
    X_train, y_train = arrays['train']; X_val, y_val = arrays['validation']; X_test, y_test = arrays['test']
    candidates = {name: options for name, options in build_candidates(y_train).items() if name in models}
    winners, _ = tune_classifiers(X_train, y_train, X_val, y_val, candidates, precision_target)
    selected = max(winners, key=lambda name: validation_rank(winners[name][1]))
    inside = test.Family.isin(combo).to_numpy()
    rng = np.random.default_rng(SEED)
    base = {'combination': '+'.join(combo), 'size': len(combo),
            'phishing_families': sum(FAMILIES[f] for f in combo),
            'legit_families': sum(1 - FAMILIES[f] for f in combo),
            'train_rows': len(y_train), 'train_phishing': int(y_train.sum()),
            'validation_rows': len(y_val)}
    summary, by_family = [], []
    for name, (model, decision) in winners.items():
        scores = positive_scores(model, X_test)
        threshold = decision['threshold']
        predicted = (scores >= threshold).astype(int)
        row = {**base, 'model': name, 'selected_on_validation': name == selected,
               'threshold': threshold, 'validation_precision': decision['precision'],
               'validation_recall': decision['recall'], 'validation_f2': decision['f2'],
               'validation_average_precision': decision['average_precision'],
               'precision_target_met': decision['precision_target_met']}
        for scope, mask in (('in', inside), ('held_out', ~inside)):
            y = y_test[mask]
            row[f'{scope}_families'] = '+'.join(sorted(set(test.Family[mask])))
            row[f'{scope}_rows'] = int(mask.sum())
            if not mask.any():
                continue
            metrics = classification_metrics(y, scores[mask], threshold)
            for key in ('Precision', 'Recall', 'F1-score', 'F2-score', 'FPR', 'ROC-AUC', 'Average-Precision'):
                row[f'{scope}_{key}'] = metrics[key]
            tp, fp, fn, tn = bootstrap_counts(y, predicted[mask], test._group.to_numpy()[mask], draws, rng)
            for key, values in (('Recall', ratio(tp, tp + fn)), ('FPR', ratio(fp, fp + tn)),
                                ('F1-score', ratio(2 * tp, 2 * tp + fp + fn))):
                row[f'{scope}_{key}_ci_low'], row[f'{scope}_{key}_ci_high'] = interval(values)
        summary.append(row)
        for family in families:
            mask = (test.Family == family).to_numpy()
            flags = predicted[mask]
            tp, fp, fn, tn = bootstrap_counts(y_test[mask], flags, test._group.to_numpy()[mask], draws, rng)
            # Phishing family: share detected (recall). Legitimate family: share flagged (FPR).
            rates = ratio(tp, tp + fn) if FAMILIES[family] else ratio(fp, fp + tn)
            low, high = interval(rates)
            by_family.append({'combination': base['combination'], 'size': base['size'], 'model': name,
                              'selected_on_validation': name == selected, 'family': family,
                              'class': FAMILIES[family], 'in_combination': family in combo,
                              'rows': int(mask.sum()),
                              'metric': 'recall' if FAMILIES[family] else 'fpr',
                              'value': float(flags.mean()), 'ci_low': low, 'ci_high': high,
                              'mean_score': float(scores[mask].mean())})
    return summary, by_family


def append_csv(path, rows):
    frame = pd.DataFrame(rows)
    if path.exists():
        # Keep columns stable across appends (held-out columns are absent for the full set).
        columns = pd.read_csv(path, nrows=0).columns.tolist()
        frame = frame.reindex(columns=columns + [c for c in frame.columns if c not in columns])
        if len(frame.columns) > len(columns):
            existing = pd.read_csv(path)
            pd.concat([existing, frame], ignore_index=True).to_csv(path, index=False)
            return
    frame.to_csv(path, mode='a', header=not path.exists(), index=False)


def run(features, output, sizes=None, families=None, models=DEFAULT_MODELS, doc2vec='per-combination',
        epochs=DOC2VEC_EPOCHS, train_cap=5000, eval_cap=2000, draws=1000, limit=None,
        precision_target=MIN_VALIDATION_PRECISION):
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    unknown = set(models) - set(build_candidates(np.array([0, 1])))
    if unknown:
        raise ValueError(f'Unknown models: {sorted(unknown)}')
    families = list(families or FAMILIES)
    if set(families) - set(FAMILIES):
        raise ValueError(f'Unknown families: {sorted(set(families) - set(FAMILIES))}')
    sizes = sorted(set(sizes or range(2, len(families) + 1)))
    config = {'features': str(features), 'families': families, 'sizes': sizes, 'models': list(models),
              'doc2vec': doc2vec, 'epochs': epochs, 'train_cap': train_cap, 'eval_cap': eval_cap,
              'bootstrap_draws': draws, 'precision_target': precision_target, 'seed': SEED}
    config_path = output / 'config.json'
    if config_path.exists():
        previous = json.loads(config_path.read_text(encoding='utf-8'))
        if {k: v for k, v in previous.items() if k != 'export'} != config:
            raise ValueError(f'{output} holds a run with other settings; choose a new --output')
    pools, manifest, dropped, missing = load_pools(features, train_cap, eval_cap, doc2vec != 'none')
    absent = sorted({family for _, family in missing if family in families})
    if absent:
        raise ValueError(f'Families without rows in some split: {absent}; drop them with --families')
    pools = {split: {f: part for f, part in parts.items() if f in families} for split, parts in pools.items()}
    config['export'] = {'seed': manifest['seed'], 'dropped_removed_source_rows': dropped,
                        'pool_rows': {split: {f: len(part) for f, part in parts.items()}
                                      for split, parts in pools.items()}}
    config_path.write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding='utf-8')
    if any(dropped.values()):
        print(f'Dropped rows of removed sources {sorted(REMOVED_SOURCES)}: {dropped}. '
              'Raw copies deduplicated in their favour are missing too; prefer a newer export.', flush=True)
    todo = enumerate_combinations(families, sizes)
    pd.DataFrame({'combination': ['+'.join(c) for c in todo], 'size': [len(c) for c in todo]}).to_csv(
        output / 'combinations.csv', index=False)
    done = set(pd.read_csv(output / SUMMARY).combination) if (output / SUMMARY).exists() else set()
    todo = [c for c in todo if '+'.join(c) not in done]
    if limit is not None:
        todo = todo[:limit]
    print(f'{len(done)} combinations already done, {len(todo)} to run', flush=True)
    for index, combo in enumerate(todo, 1):
        started = time.perf_counter()
        print(f'[{index}/{len(todo)}] {"+".join(combo)}', flush=True)
        summary, by_family = score_combination(combo, pools, manifest['preprocessing'], models, doc2vec,
                                               epochs, draws, precision_target)
        seconds = round(time.perf_counter() - started, 1)
        for row in summary:
            row['seconds'] = seconds
        append_csv(output / BY_FAMILY, by_family)
        append_csv(output / SUMMARY, summary)  # written last: its rows mark a combination as done
        print(f'  done in {seconds}s', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--features', type=Path, required=True, help='Feature export folder (features_manifest.json)')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--sizes', type=lambda s: [int(x) for x in s.split(',')],
                        help='Combination sizes, e.g. 2,3 (default: all)')
    parser.add_argument('--families', type=lambda s: [x.strip() for x in s.split(',')],
                        help=f'Subset of: {", ".join(FAMILIES)}')
    parser.add_argument('--models', type=lambda s: [x.strip() for x in s.split(',')], default=list(DEFAULT_MODELS),
                        help='Classifier names as in train.py, or "all" for all 9')
    parser.add_argument('--doc2vec', choices=('per-combination', 'none'), default='per-combination',
                        help='per-combination: fit Doc2Vec on each combination (as train.py); none: 134 features only')
    parser.add_argument('--epochs', type=int, default=DOC2VEC_EPOCHS)
    parser.add_argument('--train-cap', type=int, default=5000, help='Max train rows per family')
    parser.add_argument('--eval-cap', type=int, default=2000, help='Max validation/test rows per family')
    parser.add_argument('--bootstrap', type=int, default=1000, help='Group bootstrap draws for intervals')
    parser.add_argument('--limit', type=int, help='Run at most this many pending combinations')
    args = parser.parse_args()
    models = list(build_candidates(np.array([0, 1]))) if args.models == ['all'] else args.models
    run(args.features, args.output, sizes=args.sizes, families=args.families, models=models,
        doc2vec=args.doc2vec, epochs=args.epochs, train_cap=args.train_cap, eval_cap=args.eval_cap,
        draws=args.bootstrap, limit=args.limit)
