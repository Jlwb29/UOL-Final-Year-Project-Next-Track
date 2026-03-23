from django.test import TestCase
import numpy as np

from api.services import cosine_similarity


class TestCosineSimilarity(TestCase):
    def test_identical_vectors_return_1(self):
        a = [1, 0, 1, 0, 0]
        b = [1, 0, 1, 0, 0]
        s = cosine_similarity(a, b)
        self.assertAlmostEqual(s, 1.0, places=6)

    def test_orthogonal_vectors_return_0(self):
        a = [1, 0, 0, 0]
        b = [0, 1, 0, 0]
        s = cosine_similarity(a, b)
        self.assertAlmostEqual(s, 0.0, places=6)

    def test_zero_vector_returns_0(self):
        a = [0, 0, 0]
        b = [1, 2, 3]
        s = cosine_similarity(a, b)
        self.assertAlmostEqual(s, 0.0, places=6)

    def test_minor_variation_reduces_similarity_proportionally(self):
        a = np.asarray([1.0, 0.0, 1.0, 0.0], dtype=float)
        b_small = np.asarray([1.0, 0.0, 0.9, 0.0], dtype=float)
        b_large = np.asarray([1.0, 0.0, 0.5, 0.0], dtype=float)
        s_small = cosine_similarity(a, b_small)
        s_large = cosine_similarity(a, b_large)
        self.assertLess(s_small, 1.0)
        self.assertGreater(s_small, 0.0)
        self.assertLess(s_large, s_small)

    def test_symmetry_property(self):
        a = [0.2, 0.1, 0.0, 0.9]
        b = [0.1, 0.3, 0.0, 0.7]
        self.assertAlmostEqual(cosine_similarity(a, b), cosine_similarity(b, a), places=6)
