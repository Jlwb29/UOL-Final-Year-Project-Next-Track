import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


GENRE_VOCAB = ["pop", "rock", "hiphop", "rap", "edm", "jazz", "classical", "rnb", "metal", "indie"]
MOOD_VOCAB = ["happy", "sad", "relaxed", "energetic", "chill", "angry", "romantic", "focus"]
ERA_VOCAB = ["1980s", "1990s", "2000s", "2010s", "2020s"]

TEMPO_MIN = 60.0
TEMPO_MAX = 200.0
YEAR_MIN = 1980.0
YEAR_MAX = 2025.0


def _one_hot(value, vocab):
    vec = [0.0] * len(vocab)

    if value is None:
        return vec

    v = str(value).strip().lower()

    if v in vocab:
        vec[vocab.index(v)] = 1.0

    return vec


def _clip01(x):
    if x < 0.0:
        return 0.0

    if x > 1.0:
        return 1.0

    return x


def normalize_minmax(value, vmin, vmax):
    if value is None:
        return 0.0

    try:
        v = float(value)
    except (TypeError, ValueError):
        return 0.0

    vmin = float(vmin)
    vmax = float(vmax)

    if vmax == vmin:
        return 0.0

    return _clip01((v - vmin) / (vmax - vmin))


def get_metadata_vector_labels():
    labels = []

    for item in GENRE_VOCAB:
        labels.append(f"genre::{item}")

    for item in MOOD_VOCAB:
        labels.append(f"mood::{item}")

    for item in ERA_VOCAB:
        labels.append(f"era::{item}")

    labels.append("tempo_norm")
    labels.append("year_norm")

    return labels


def build_feature_vector(genre=None, mood=None, era=None, tempo_bpm=None, year=None):
    tempo_norm = normalize_minmax(tempo_bpm, TEMPO_MIN, TEMPO_MAX)
    year_norm = normalize_minmax(year, YEAR_MIN, YEAR_MAX)

    return (
        _one_hot(genre, GENRE_VOCAB)
        + _one_hot(mood, MOOD_VOCAB)
        + _one_hot(era, ERA_VOCAB)
        + [tempo_norm, year_norm]
    )


def build_metadata_vector_breakdown(genre=None, mood=None, era=None, tempo_bpm=None, year=None):
    vector = build_feature_vector(
        genre=genre,
        mood=mood,
        era=era,
        tempo_bpm=tempo_bpm,
        year=year,
    )

    labels = get_metadata_vector_labels()

    return {
        "labels": labels,
        "values": vector,
        "pairs": [
            {
                "label": labels[i],
                "value": vector[i],
            }
            for i in range(len(labels))
        ],
    }


def build_track_text(t):
    parts = []

    raw_tags = getattr(t, "tags", "")
    if raw_tags:
        clean_tags = str(raw_tags).strip()
        if clean_tags:
            parts.append(clean_tags)

    if getattr(t, "genre", None):
        parts.append(str(t.genre).strip().lower())

    if getattr(t, "mood", None):
        parts.append(str(t.mood).strip().lower())

    if getattr(t, "era", None):
        parts.append(str(t.era).strip().lower())

    if getattr(t, "title", None):
        parts.append(str(t.title).strip().lower())

    if getattr(t, "artist_name", None):
        parts.append(str(t.artist_name).strip().lower())

    combined_text = " ".join([p for p in parts if p]).strip()

    if combined_text:
        return combined_text

    return "untagged"


def build_tfidf_matrix(tracks, return_details=False):
    texts = []

    for t in tracks:
        texts.append(build_track_text(t))

    try:
        vectorizer = TfidfVectorizer(stop_words="english")
        tfidf_matrix = vectorizer.fit_transform(texts)
        tfidf_array = tfidf_matrix.toarray()

        if return_details:
            try:
                feature_names = vectorizer.get_feature_names_out().tolist()
            except AttributeError:
                feature_names = []

            return {
                "matrix": tfidf_array,
                "labels": [f"tfidf::{name}" for name in feature_names],
                "texts": texts,
            }

        return tfidf_array

    except ValueError:
        zero_matrix = np.zeros((len(tracks), 1))

        if return_details:
            return {
                "matrix": zero_matrix,
                "labels": ["tfidf::empty"],
                "texts": texts,
            }

        return zero_matrix


def build_combined_vectors(tracks):
    metadata_vectors = []

    for t in tracks:
        metadata_vectors.append(
            build_feature_vector(
                genre=t.genre,
                mood=t.mood,
                era=t.era,
                tempo_bpm=t.tempo_bpm,
                year=t.year,
            )
        )

    tfidf_vectors = build_tfidf_matrix(tracks)

    vector_map = {}

    for i, t in enumerate(tracks):
        combined_vector = np.concatenate(
            [
                metadata_vectors[i],
                tfidf_vectors[i],
            ]
        )

        vector_map[str(t.track_id)] = combined_vector.tolist()

    return vector_map


def build_combined_vectors_with_labels(tracks):
    metadata_labels = get_metadata_vector_labels()
    metadata_vectors = []

    for t in tracks:
        metadata_vectors.append(
            build_feature_vector(
                genre=t.genre,
                mood=t.mood,
                era=t.era,
                tempo_bpm=t.tempo_bpm,
                year=t.year,
            )
        )

    tfidf_result = build_tfidf_matrix(tracks, return_details=True)
    tfidf_vectors = tfidf_result["matrix"]
    tfidf_labels = tfidf_result["labels"]
    texts = tfidf_result["texts"]

    combined_labels = metadata_labels + tfidf_labels
    vector_map = {}
    transparency_map = {}

    for i, t in enumerate(tracks):
        metadata_vector = metadata_vectors[i]
        tfidf_vector = tfidf_vectors[i].tolist()

        combined_vector = np.concatenate(
            [
                metadata_vector,
                tfidf_vectors[i],
            ]
        ).tolist()

        track_id = str(t.track_id)

        vector_map[track_id] = combined_vector
        transparency_map[track_id] = {
            "track_id": track_id,
            "metadata_vector": metadata_vector,
            "metadata_labels": metadata_labels,
            "metadata_pairs": [
                {
                    "label": metadata_labels[j],
                    "value": metadata_vector[j],
                }
                for j in range(len(metadata_labels))
            ],
            "tfidf_text": texts[i],
            "tfidf_vector": tfidf_vector,
            "tfidf_labels": tfidf_labels,
            "tfidf_pairs": [
                {
                    "label": tfidf_labels[j],
                    "value": tfidf_vector[j],
                }
                for j in range(len(tfidf_labels))
            ],
            "combined_vector": combined_vector,
            "combined_labels": combined_labels,
            "combined_pairs": [
                {
                    "label": combined_labels[j],
                    "value": combined_vector[j],
                }
                for j in range(len(combined_labels))
            ],
        }

    return {
        "vector_map": vector_map,
        "combined_labels": combined_labels,
        "metadata_labels": metadata_labels,
        "tfidf_labels": tfidf_labels,
        "transparency_map": transparency_map,
    }


def get_track_vector_transparency(track, all_tracks):
    result = build_combined_vectors_with_labels(all_tracks)
    track_id = str(track.track_id)

    return result["transparency_map"].get(
        track_id,
        {
            "track_id": track_id,
            "metadata_vector": [],
            "metadata_labels": [],
            "metadata_pairs": [],
            "tfidf_text": "",
            "tfidf_vector": [],
            "tfidf_labels": [],
            "tfidf_pairs": [],
            "combined_vector": [],
            "combined_labels": [],
            "combined_pairs": [],
        },
    )


def cosine_similarity(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    denom = np.linalg.norm(a) * np.linalg.norm(b)

    if denom == 0:
        return 0.0

    return float(np.dot(a, b) / denom)


def average_seed_metadata(seeds):
    tempo_values = []
    year_values = []

    for seed in seeds:
        if getattr(seed, "tempo_bpm", None) is not None:
            try:
                tempo_values.append(float(seed.tempo_bpm))
            except (TypeError, ValueError):
                pass

        if getattr(seed, "year", None) is not None:
            try:
                year_values.append(float(seed.year))
            except (TypeError, ValueError):
                pass

    avg_tempo = None
    avg_year = None

    if tempo_values:
        avg_tempo = round(sum(tempo_values) / len(tempo_values), 2)

    if year_values:
        avg_year = round(sum(year_values) / len(year_values), 2)

    return {
        "avg_seed_tempo_bpm": avg_tempo,
        "avg_seed_year": avg_year,
    }


def similarity_distribution(avg_seed_vec, candidate_tracks, bins=20, vector_map=None):
    scores = []

    if vector_map is None:
        vector_map = {}

    for t in candidate_tracks:
        candidate_vector = vector_map.get(str(t.track_id), [])

        if candidate_vector:
            scores.append(cosine_similarity(avg_seed_vec, candidate_vector))
        else:
            scores.append(0.0)

    if not scores:
        return {
            "scores": [],
            "bin_edges": [],
            "counts": [],
            "min": None,
            "max": None,
            "mean": None,
            "zero_norm_candidates": 0,
        }

    hist, edges = np.histogram(scores, bins=int(bins), range=(-1.0, 1.0))

    zero_norm = 0

    for t in candidate_tracks:
        v = np.asarray(vector_map.get(str(t.track_id), []), dtype=float)

        if v.size == 0 or np.linalg.norm(v) == 0:
            zero_norm += 1

    return {
        "scores": [round(float(s), 6) for s in scores],
        "bin_edges": [round(float(e), 6) for e in edges],
        "counts": hist.tolist(),
        "min": round(float(min(scores)), 6),
        "max": round(float(max(scores)), 6),
        "mean": round(float(np.mean(scores)), 6),
        "zero_norm_candidates": int(zero_norm),
    }

def validate_track_metadata(track):

    errors = []

    if not track.genre:
        errors.append("missing genre")

    if not track.mood:
        errors.append("missing mood")

    if track.tempo_bpm is None:
        errors.append("missing tempo")

    if track.year is None:
        errors.append("missing year")

    if track.tempo_bpm and (track.tempo_bpm < 40 or track.tempo_bpm > 250):
        errors.append("invalid tempo")

    if track.year and (track.year < 1900 or track.year > 2026):
        errors.append("invalid year")

    return {
        "valid": len(errors) == 0,
        "errors": errors
    }

def metadata_completeness(track):

    fields = [
        track.genre,
        track.mood,
        track.tempo_bpm,
        track.year,
        track.tags
    ]

    total = len(fields)

    present = 0

    for f in fields:
        if f not in [None, "", []]:
            present += 1

    return round(present / total, 2)

def similarity_distribution_from_scores(scores, bins=20):
    if not scores:
        return {
            "bin_edges": [],
            "counts": [],
        }

    min_score = min(scores)
    max_score = max(scores)

    if min_score == max_score:
        min_score -= 0.01
        max_score += 0.01

    step = (max_score - min_score) / bins
    bin_edges = [min_score + i * step for i in range(bins + 1)]
    counts = [0] * bins

    for score in scores:
        placed = False

        for i in range(bins):
            left = bin_edges[i]
            right = bin_edges[i + 1]

            if i == bins - 1:
                if left <= score <= right:
                    counts[i] += 1
                    placed = True
                    break
            else:
                if left <= score < right:
                    counts[i] += 1
                    placed = True
                    break

        if not placed and score == max_score:
            counts[-1] += 1

    return {
        "bin_edges": [round(x, 4) for x in bin_edges],
        "counts": counts,
    }

