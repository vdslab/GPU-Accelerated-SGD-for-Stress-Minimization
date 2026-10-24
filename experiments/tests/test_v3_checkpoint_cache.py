"""Tiny CSV roundtrips; no graph files, snapshots or stress recomputation."""
import csv
import tempfile
import unittest
from pathlib import Path

from experiments import summarize_v3 as summary


class CheckpointCacheTests(unittest.TestCase):
    def roundtrip(self, base, checkpoint):
        path = base / 'tables/convergence.csv'
        summary.dump_csv(path, [checkpoint])
        with path.open(newline='') as source:
            cached = next(csv.DictReader(source))
        return summary.parse_checkpoint_row(cached)

    def audit(self, base, checkpoint, *, kind='sampled', seed=0):
        native = dict(study='convergence', graph='fixture', method='sparse_sgd',
                      seed=0, status='success', stress_value=12.5,
                      stress_kind=kind, stress_samples=256 if kind == 'sampled' else None,
                      stress_seed=seed if kind == 'sampled' else None)
        summary.convergence_stress_audit(base, [native], [checkpoint])
        with (base / 'tables/convergence-stress-audit.csv').open(newline='') as source:
            return next(csv.DictReader(source))

    def checkpoint(self, **changes):
        value = dict(cache_key='fixture-key', graph='fixture', method='sparse_sgd',
                     family='sparse', seed=0, epoch=15, elapsed_ms=0., stress=12.5,
                     stress_kind='sampled', stress_samples=256, stress_seed=0,
                     snapshot='unused.f64le')
        value.update(changes)
        return value

    def test_sampled_zero_seed_survives_csv_and_matches_native_definition(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            (base / 'tables').mkdir()
            checkpoint = self.roundtrip(base, self.checkpoint())
            self.assertEqual(checkpoint, self.checkpoint())
            for key in ['seed', 'epoch', 'stress_samples', 'stress_seed']:
                self.assertIs(type(checkpoint[key]), int)
            for key in ['stress', 'elapsed_ms']:
                self.assertIs(type(checkpoint[key]), float)
            self.assertEqual(summary.parse_checkpoint_row(checkpoint), checkpoint)
            audit = self.audit(base, checkpoint)
            self.assertEqual(audit['evaluator_definition_compatible'], 'True')
            self.assertEqual(audit['flag'], 'within_0_1_percent')
            self.assertEqual(audit['offline_stress_seed'], '0')

    def test_exact_evaluation_ignores_cached_source_count_and_seed(self):
        for source_count, source_seed in [(4, 0), (None, None)]:
            with self.subTest(source_seed=source_seed), tempfile.TemporaryDirectory() as temporary:
                base = Path(temporary)
                (base / 'tables').mkdir()
                checkpoint = self.roundtrip(base, self.checkpoint(
                    stress_kind='exact', stress_samples=source_count, stress_seed=source_seed))
                audit = self.audit(base, checkpoint, kind='exact')
                self.assertEqual(audit['evaluator_definition_compatible'], 'True')
                self.assertEqual(audit['flag'], 'within_0_1_percent')
                self.assertEqual(audit['offline_stress_samples'], '')
                self.assertEqual(audit['offline_stress_seed'], '')

    def test_different_sampled_seed_remains_definition_mismatch(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            (base / 'tables').mkdir()
            checkpoint = self.roundtrip(base, self.checkpoint(stress_seed=1))
            audit = self.audit(base, checkpoint)
            self.assertEqual(audit['evaluator_definition_compatible'], 'False')
            self.assertEqual(audit['flag'], 'definition_mismatch')

    def test_missing_and_invalid_numeric_metadata_are_not_fabricated(self):
        self.assertEqual(summary.parse_checkpoint_row({'stress_seed': '', 'elapsed_ms': None}),
                         {'stress_seed': '', 'elapsed_ms': None})
        with self.assertRaises(ValueError):
            summary.parse_checkpoint_row({'stress_seed': 'not-a-seed'})


if __name__ == '__main__':
    unittest.main()
