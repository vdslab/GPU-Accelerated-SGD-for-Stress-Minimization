import unittest
from unittest.mock import patch
from experiments import summarize_v1 as summary


class SourceExtensionTests(unittest.TestCase):
    def setUp(self):
        summary.source_extension_evidence.cache_clear()

    def test_unchanged_shared_trees_allow_an_additive_atomic_method(self):
        with patch.object(summary.subprocess, 'check_output', return_value='tree-id\n'):
            evidence = summary.source_extension_evidence('baseline', 'addition')
        self.assertEqual(evidence['baseline_commit'], 'baseline')
        self.assertEqual(set(evidence['shared_source_trees']), set(summary.SHARED_SOURCE_PATHS))

    def test_any_changed_shared_implementation_prevents_baseline_reuse(self):
        def git_tree(command, **kwargs):
            return 'changed\n' if command[-1] == 'addition:experiment-common' else 'same\n'
        with patch.object(summary.subprocess, 'check_output', side_effect=git_tree):
            self.assertIsNone(summary.source_extension_evidence('baseline', 'addition'))
