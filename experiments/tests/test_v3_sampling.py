"""Small numerical tests; no graph files, native runs or stress recomputation."""
import math
import unittest
from unittest.mock import patch
import warnings

import numpy as np

from experiments.check_v3_sampling import source_terms


def scalar_terms(xy, distances, sources):
    """Independent scalar oracle: math.hypot and math.fsum, without NumPy BLAS."""
    result = []
    for source, row in zip(sources, distances):
        squares = []
        for target, distance in enumerate(row):
            if distance > 0 and math.isfinite(distance):
                actual = math.hypot(float(xy[target][0]) - float(xy[source][0]),
                                    float(xy[target][1]) - float(xy[source][1]))
                residual = (actual - float(distance)) / float(distance)
                squares.append(residual * residual)
        result.append(math.fsum(squares))
    return result


class SourceTermsTests(unittest.TestCase):
    def test_known_three_point_values(self):
        xy = np.array([[0., 0.], [3., 0.], [3., 4.]])
        distances = np.array([[0., 1., 2.], [1., 0., 1.], [2., 1., 0.]])
        self.assertEqual(source_terms(xy, distances, np.array([0, 1, 2])).tolist(),
                         [6.25, 13., 11.25])

    def test_zero_and_unreachable_distances_contribute_zero(self):
        xy = np.array([[0., 0.], [3., 4.], [100., 100.]])
        sources = np.array([0, 2])
        distances = np.array([[0., 2., np.inf], [np.inf, np.inf, 0.]])
        with warnings.catch_warnings():
            warnings.simplefilter('error', RuntimeWarning)
            result = source_terms(xy, distances, sources)
        self.assertEqual(result.tolist(), [2.25, 0.])

    def test_large_finite_vector_uses_no_dot_and_matches_fsum(self):
        # Length 1024 also triggers Accelerate ddot's false flags on this host.
        indices = np.arange(1024, dtype=np.float64)
        xy = np.column_stack((indices / 8, (indices % 7) / 16))
        sources = np.array([0, 513])
        distances = np.vstack(((indices % 31) + 1, (indices % 17) + 1))
        distances[0, 0] = 0
        distances[1, 513] = 0
        expected = scalar_terms(xy, distances, sources)
        with patch('numpy.dot', side_effect=AssertionError('BLAS dot must not be used')):
            with warnings.catch_warnings():
                warnings.simplefilter('error', RuntimeWarning)
                actual = source_terms(xy, distances, sources)
        for value, reference in zip(actual, expected):
            self.assertTrue(math.isclose(value, reference, rel_tol=2e-14))

    def test_saved_coordinate_prefix_matches_independent_fsum(self):
        # First 8 actual final coordinates: Luxembourg CPU/basic/seed0/P200.
        # Distances below are synthetic; this is not an actual stress fixture.
        xy = np.array([
            [111.53315791698144, -164.81612351553673],
            [113.45376067381919, -164.42814122722365],
            [114.51524846727828, -163.81744632373778],
            [115.29365022800893, -162.48825918884472],
            [116.30715322327829, -161.7032118909745],
            [117.25617255435527, -161.05846614400662],
            [117.29978687872573, -160.1064415475874],
            [117.49268390754959, -159.5090794844436],
        ])
        sources = np.array([0, 7])
        distances = np.array([[0., 1., 2., 3., 4., 5., 6., np.inf],
                              [np.inf, 6., 5., 4., 3., 2., 1., 0.]])
        actual = source_terms(xy, distances, sources)
        expected = scalar_terms(xy, distances, sources)
        for value, reference in zip(actual, expected):
            self.assertTrue(math.isclose(value, reference, rel_tol=2e-14))

    def test_nonfinite_coordinates_rejected(self):
        for value in [np.nan, np.inf, -np.inf]:
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, 'coordinates'):
                source_terms(np.array([[0., 0.], [value, 1.]]),
                             np.array([[0., 1.]]), np.array([0]))

    def test_invalid_graph_distances_rejected(self):
        for value in [np.nan, -np.inf, -1.]:
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, 'Graph distances'):
                source_terms(np.array([[0., 0.], [1., 1.]]),
                             np.array([[0., value]]), np.array([0]))

    def test_dimensions_and_source_indices_rejected(self):
        xy = np.array([[0., 0.], [1., 1.]])
        for coords, distances, sources in [
                (xy[:, 0], np.array([[0., 1.]]), np.array([0])),
                (xy, np.array([[0.]]), np.array([0])),
                (xy, np.array([[0., 1.]]), np.array([-1])),
                (xy, np.array([[0., 1.]]), np.array([2])),
                (xy, np.array([[0., 1.]]), np.array([0.])),
                (xy, np.array([[0., 1.]]), np.array([[0]]))]:
            with self.subTest(sources=sources), self.assertRaises(ValueError):
                source_terms(coords, distances, sources)

    def test_nonfinite_intermediate_or_squared_sum_rejected(self):
        for coordinate, distance in [(1e308, 1.), (1., 1e-309), (1e100, 1e-60)]:
            with self.subTest(coordinate=coordinate, distance=distance):
                with self.assertRaisesRegex(ValueError, 'Nonfinite'):
                    source_terms(np.array([[0., 0.], [coordinate, 0.]]),
                                 np.array([[0., distance]]), np.array([0]))


if __name__ == '__main__':
    unittest.main()
