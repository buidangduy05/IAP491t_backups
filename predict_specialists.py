"""Assess raw email with available specialists and report missing coverage."""
import argparse
import json
import pickle
from pathlib import Path

import numpy as np

from evaluate import file_digest, positive_scores
from pipeline_cache import _process, preprocessing_signature
from preprocess import inspect_email_attachments, BertEmailClassifier
from specialists import availability, feature_dict


class EmailSpecialists:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.manifest = json.loads((self.directory / 'specialist_manifest.json').read_text())
        if self.manifest['status'] not in {'complete', 'partial'}:
            raise ValueError('Specialist run is incomplete')
        signature, _ = preprocessing_signature()
        if signature != self.manifest['preprocessing']['signature']:
            raise ValueError('Inference preprocessing differs from training')
        for name, expected in self.manifest['inference_artifacts'].items():
            if file_digest(self.directory / name) != expected:
                raise ValueError(f'Artifact changed: {name}')
        self.models = {}
        for name, config in self.manifest['branches'].items():
            if config['status'] == 'trained':
                if name == 'text':
                    self.models[name] = BertEmailClassifier.from_pretrained(self.directory / config['directory'])
                else:
                    with (self.directory / config['file']).open('rb') as handle:
                        self.models[name] = pickle.load(handle)

    def assess(self, raw):
        identity = _process((raw, 'identity', False))
        features = _process((raw, 'features', self.manifest['preprocessing']['lemmatization']))
        if '_error' in identity or '_error' in features:
            raise ValueError(identity.get('_error') or features.get('_error'))
        row = {**identity, **features}
        present, values = availability(row), feature_dict(row)
        result = {'branches': {}, 'attachment_evidence': inspect_email_attachments(raw)}
        for name, config in self.manifest['branches'].items():
            entry = {'target': config['target'], 'algorithm': config['algorithm']}
            if not present[name]:
                entry['status'] = 'no_applicable_input'
            elif config['status'] != 'trained':
                entry.update(status=config['status'], reason=config.get('reason', ''))
            else:
                inputs = [row['_semantic_text']] if name == 'text' else np.asarray(
                    [[values[n] for n in config['features']]], dtype=np.float32)
                score = float(positive_scores(self.models[name], inputs)[0])
                entry.update(status='assessed', score=score, threshold=config['threshold'],
                             validation_precision_target_met=config['precision_target_met'],
                             positive=score >= config['threshold'])
            result['branches'][name] = entry
        result['phishing_evidence'] = [n for n, e in result['branches'].items()
                                      if e['target'] == 'email_phishing' and e.get('positive')]
        result['malware_alerts'] = [n for n, e in result['branches'].items()
                                  if e['target'] != 'email_phishing' and e.get('positive')]
        result['unassessed_branches'] = [n for n, e in result['branches'].items()
                                        if e['status'] not in {'assessed', 'no_applicable_input'}]
        return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('email_file')
    parser.add_argument('--run', required=True)
    args = parser.parse_args()
    print(json.dumps(EmailSpecialists(args.run).assess(Path(args.email_file).read_bytes()), indent=2))
