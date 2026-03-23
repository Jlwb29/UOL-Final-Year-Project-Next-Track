import math


WEIGHTS = {
    "cosine_score": 0.60,
    "genre_match": 0.12,
    "mood_match": 0.05,
    "era_match": 0.03,
    "tempo_match": 0.08,
    "year_match": 0.08,
    "metadata_bonus": 0.02,
    "discovery_bonus": 0.02,
}


def cosine_similarity(vec1, vec2):
    if not vec1 or not vec2:
        return 0.0

    if len(vec1) != len(vec2):
        return 0.0

    dot_product = 0.0
    norm_a = 0.0
    norm_b = 0.0

    for a, b in zip(vec1, vec2):
        dot_product += a * b
        norm_a += a * a
        norm_b += b * b

    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0

    return dot_product / (math.sqrt(norm_a) * math.sqrt(norm_b))


def compute_baseline_score(seed_vector, candidate_vector):
    return round(cosine_similarity(seed_vector, candidate_vector), 4)


def get_baseline_breakdown(seed_vector, candidate_vector):
    
    if seed_vector is not None and candidate_vector is not None:
        
        if len(seed_vector) > 0 and len(candidate_vector) > 0:
            
            if len(seed_vector) == len(candidate_vector):
                
                cosine_score = cosine_similarity(seed_vector, candidate_vector)
                
                return {
                    "method": "baseline",
                    "vector_length_seed": len(seed_vector),
                    "vector_length_candidate": len(candidate_vector),
                    "cosine_score": round(cosine_score, 4),
                    "final_score": round(cosine_score, 4),
                    "status": "success"
                }
            
            else:
                return {
                    "method": "baseline",
                    "vector_length_seed": len(seed_vector),
                    "vector_length_candidate": len(candidate_vector),
                    "cosine_score": 0.0,
                    "final_score": 0.0,
                    "status": "error: vector length mismatch"
                }
        
        else:
            return {
                "method": "baseline",
                "vector_length_seed": len(seed_vector),
                "vector_length_candidate": len(candidate_vector),
                "cosine_score": 0.0,
                "final_score": 0.0,
                "status": "error: empty vector"
            }
    
    else:
        return {
            "method": "baseline",
            "vector_length_seed": len(seed_vector) if seed_vector else 0,
            "vector_length_candidate": len(candidate_vector) if candidate_vector else 0,
            "cosine_score": 0.0,
            "final_score": 0.0,
            "status": "error: missing vector"
        }


def get_metadata_completeness_bonus(track):
    filled = 0

    if track.genre:
        filled += 1

    if track.mood:
        filled += 1

    if track.era:
        filled += 1

    if getattr(track, "tags", None):
        if track.tags.strip():
            filled += 1

    if getattr(track, "tempo_bpm", None) is not None:
        filled += 1

    if getattr(track, "year", None) is not None:
        filled += 1

    return 0.02 * filled


def get_discovery_bonus(candidate_track):
    popularity = getattr(candidate_track, "popularity", None)

    if popularity is None:
        return 0.0

    popularity = float(popularity)

    if popularity >= 80:
        return 0.0
    elif popularity >= 60:
        return 0.01
    elif popularity >= 40:
        return 0.02
    else:
        return 0.03


def get_tempo_match(seed_track, candidate_track, max_diff=80.0):
    seed_tempo = getattr(seed_track, "tempo_bpm", None)
    candidate_tempo = getattr(candidate_track, "tempo_bpm", None)

    if seed_tempo is None or candidate_tempo is None:
        return 0.0

    try:
        diff = abs(float(seed_tempo) - float(candidate_tempo))
    except (TypeError, ValueError):
        return 0.0

    if diff >= max_diff:
        return 0.0

    return round(1.0 - (diff / max_diff), 4)


def get_year_match(seed_track, candidate_track, max_diff=30.0):
    seed_year = getattr(seed_track, "year", None)
    candidate_year = getattr(candidate_track, "year", None)

    if seed_year is None or candidate_year is None:
        return 0.0

    try:
        diff = abs(float(seed_year) - float(candidate_year))
    except (TypeError, ValueError):
        return 0.0

    if diff >= max_diff:
        return 0.0

    return round(1.0 - (diff / max_diff), 4)


def get_exact_match_flags(seed_track, candidate_track):
    genre_match = 0.0
    mood_match = 0.0
    era_match = 0.0

    if seed_track.genre and candidate_track.genre:
        if seed_track.genre.strip().lower() == candidate_track.genre.strip().lower():
            genre_match = 1.0

    if seed_track.mood and candidate_track.mood:
        if seed_track.mood.strip().lower() == candidate_track.mood.strip().lower():
            mood_match = 1.0

    if seed_track.era and candidate_track.era:
        if seed_track.era.strip().lower() == candidate_track.era.strip().lower():
            era_match = 1.0

    return genre_match, mood_match, era_match


def compute_weighted_score(seed_track, candidate_track, seed_vector, candidate_vector):
    breakdown = get_score_breakdown(seed_track, candidate_track, seed_vector, candidate_vector)
    return breakdown["final_score"]


def get_score_breakdown(seed_track, candidate_track, seed_vector, candidate_vector):
    cosine_score = cosine_similarity(seed_vector, candidate_vector)

    genre_match, mood_match, era_match = get_exact_match_flags(seed_track, candidate_track)

    tempo_match = get_tempo_match(seed_track, candidate_track)
    year_match = get_year_match(seed_track, candidate_track)

    completeness_bonus_raw = get_metadata_completeness_bonus(candidate_track)
    discovery_bonus_raw = get_discovery_bonus(candidate_track)

    cosine_part = WEIGHTS["cosine_score"] * cosine_score
    genre_part = WEIGHTS["genre_match"] * genre_match
    mood_part = WEIGHTS["mood_match"] * mood_match
    era_part = WEIGHTS["era_match"] * era_match
    tempo_part = WEIGHTS["tempo_match"] * tempo_match
    year_part = WEIGHTS["year_match"] * year_match
    metadata_part = WEIGHTS["metadata_bonus"] * completeness_bonus_raw
    discovery_part = WEIGHTS["discovery_bonus"] * discovery_bonus_raw

    final_score = (
        cosine_part +
        genre_part +
        mood_part +
        era_part +
        tempo_part +
        year_part +
        metadata_part +
        discovery_part
    )

    return {
        "method": "weighted",
        "weights": WEIGHTS,
        "raw_inputs": {
            "seed_genre": getattr(seed_track, "genre", None),
            "candidate_genre": getattr(candidate_track, "genre", None),
            "seed_mood": getattr(seed_track, "mood", None),
            "candidate_mood": getattr(candidate_track, "mood", None),
            "seed_era": getattr(seed_track, "era", None),
            "candidate_era": getattr(candidate_track, "era", None),
            "seed_tempo_bpm": getattr(seed_track, "tempo_bpm", None),
            "candidate_tempo_bpm": getattr(candidate_track, "tempo_bpm", None),
            "seed_year": getattr(seed_track, "year", None),
            "candidate_year": getattr(candidate_track, "year", None),
            "candidate_popularity": getattr(candidate_track, "popularity", None),
            "candidate_tags": getattr(candidate_track, "tags", None),
            "vector_length_seed": len(seed_vector) if seed_vector else 0,
            "vector_length_candidate": len(candidate_vector) if candidate_vector else 0,
        },
        "matches": {
            "cosine_score": round(cosine_score, 4),
            "genre_match": round(genre_match, 4),
            "mood_match": round(mood_match, 4),
            "era_match": round(era_match, 4),
            "tempo_match": round(tempo_match, 4),
            "year_match": round(year_match, 4),
            "metadata_completeness_bonus_raw": round(completeness_bonus_raw, 4),
            "discovery_bonus_raw": round(discovery_bonus_raw, 4),
        },
        "weighted_parts": {
            "cosine_part": round(cosine_part, 4),
            "genre_part": round(genre_part, 4),
            "mood_part": round(mood_part, 4),
            "era_part": round(era_part, 4),
            "tempo_part": round(tempo_part, 4),
            "year_part": round(year_part, 4),
            "metadata_part": round(metadata_part, 4),
            "discovery_part": round(discovery_part, 4),
        },
        "formula": "0.60*cosine + 0.12*genre + 0.05*mood + 0.03*era + 0.08*tempo + 0.08*year + 0.02*metadata_bonus + 0.02*discovery_bonus",
        "final_score": round(final_score, 4),
    }


def rerank_with_diversity(recommendations):
    seen_artists = {}
    reranked = []

    for item in recommendations:
        artist = (item.get("artist_name") or "").strip().lower()

        penalty = 0.0

        if artist in seen_artists:
            penalty = 0.05 * seen_artists[artist]

        current_similarity = item.get("similarity", 0.0)
        new_similarity = round(current_similarity - penalty, 4)

        item["similarity"] = new_similarity

        if "explanation" in item:
            item["explanation"]["diversity_penalty"] = round(penalty, 4)
            item["explanation"]["similarity_after_diversity"] = new_similarity

        seen_artists[artist] = seen_artists.get(artist, 0) + 1
        reranked.append(item)

    reranked.sort(key=lambda x: x["similarity"], reverse=True)
    return reranked