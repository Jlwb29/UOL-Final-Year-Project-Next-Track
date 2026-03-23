from django.test import TestCase
import numpy as np

from api.services import (
    build_feature_vector,
    normalize_minmax,
    GENRE_VOCAB, MOOD_VOCAB, ERA_VOCAB,
    TEMPO_MIN, TEMPO_MAX, YEAR_MIN, YEAR_MAX,
)

class TestMetadataEncoding(TestCase):
    def setUp(self):
        self.genre_len = len(GENRE_VOCAB)
        self.mood_len = len(MOOD_VOCAB)
        self.era_len = len(ERA_VOCAB)

        self.genre_slice = slice(0, self.genre_len)
        self.mood_slice = slice(self.genre_len, self.genre_len + self.mood_len)
        self.era_slice = slice(self.genre_len + self.mood_len, self.genre_len + self.mood_len + self.era_len)
        self.numeric_slice = slice(self.genre_len + self.mood_len + self.era_len, None)  

    def test_identical_metadata_produces_identical_vectors(self):
        v1 = build_feature_vector(genre="pop", mood="relaxed", era="2010s", tempo_bpm=120, year=2014)
        v2 = build_feature_vector(genre="pop", mood="relaxed", era="2010s", tempo_bpm=120, year=2014)
        self.assertEqual(v1, v2)

    def test_different_genre_is_orthogonal_in_genre_subvector(self):
        v_pop = build_feature_vector(genre="pop", mood=None, era=None, tempo_bpm=None, year=None)
        v_rock = build_feature_vector(genre="rock", mood=None, era=None, tempo_bpm=None, year=None)

        g_pop = np.asarray(v_pop[self.genre_slice], dtype=float)
        g_rock = np.asarray(v_rock[self.genre_slice], dtype=float)

        self.assertEqual(float(np.dot(g_pop, g_rock)), 0.0) 

    def test_different_mood_is_orthogonal_in_mood_subvector(self):
        v_relaxed = build_feature_vector(genre=None, mood="relaxed", era=None)
        v_energetic = build_feature_vector(genre=None, mood="energetic", era=None)

        m_relaxed = np.asarray(v_relaxed[self.mood_slice], dtype=float)
        m_energy = np.asarray(v_energetic[self.mood_slice], dtype=float)

        self.assertEqual(float(np.dot(m_relaxed, m_energy)), 0.0)

    def test_different_era_is_orthogonal_in_era_subvector(self):
        v_2000s = build_feature_vector(era="2000s")
        v_2010s = build_feature_vector(era="2010s")

        e_2000s = np.asarray(v_2000s[self.era_slice], dtype=float)
        e_2010s = np.asarray(v_2010s[self.era_slice], dtype=float)

        self.assertEqual(float(np.dot(e_2000s, e_2010s)), 0.0)

    def test_numeric_features_are_scaled_between_0_and_1(self):
        v = build_feature_vector(genre="pop", mood="relaxed", era="2010s", tempo_bpm=120, year=2014)
        tempo_norm, year_norm = v[self.numeric_slice]

        self.assertGreaterEqual(tempo_norm, 0.0)
        self.assertLessEqual(tempo_norm, 1.0)

        self.assertGreaterEqual(year_norm, 0.0)
        self.assertLessEqual(year_norm, 1.0)

    def test_normalization_clips_out_of_range_values(self):
        self.assertEqual(normalize_minmax(TEMPO_MIN - 999, TEMPO_MIN, TEMPO_MAX), 0.0)
        self.assertEqual(normalize_minmax(YEAR_MIN - 999, YEAR_MIN, YEAR_MAX), 0.0)
        self.assertEqual(normalize_minmax(TEMPO_MAX + 999, TEMPO_MIN, TEMPO_MAX), 1.0)
        self.assertEqual(normalize_minmax(YEAR_MAX + 999, YEAR_MIN, YEAR_MAX), 1.0)

    def test_missing_numeric_values_default_to_zero(self):
        v = build_feature_vector(genre="pop", mood="relaxed", era="2010s", tempo_bpm=None, year=None)
        tempo_norm, year_norm = v[self.numeric_slice]
        self.assertEqual(tempo_norm, 0.0)
        self.assertEqual(year_norm, 0.0)
