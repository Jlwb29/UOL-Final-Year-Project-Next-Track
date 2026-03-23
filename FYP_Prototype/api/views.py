from django.shortcuts import render

from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status

from .models import Track
from .serializers import TrackSerializer, TrackReadSerializer, RecommendQuerySerializer

from .services import (
    build_feature_vector,
    build_combined_vectors_with_labels,
    similarity_distribution,
    similarity_distribution_from_scores,
    average_seed_metadata,
    get_track_vector_transparency,
    validate_track_metadata,
)

from .recommender import (
    compute_weighted_score,
    compute_baseline_score,
    rerank_with_diversity,
    get_score_breakdown,
    get_baseline_breakdown,
)

def format_display_text(value):
    if not value:
        return value
    return str(value).strip().title()

def dashboard_view(request):
    return render(request, "dashboard.html")

def get_request_bool(value, default=False):
    if value is None:
        return default

    return str(value).strip().lower() == "true"


def build_preference_summary(data):
    return {
        "genre": data.get("genre"),
        "mood": data.get("mood"),
        "era": data.get("era"),
        "tempo_bpm": data.get("tempo_bpm"),
        "year": data.get("year"),
        "tags": data.get("tags"),
    }


def has_preference_input(data):
    return any([
        data.get("genre"),
        data.get("mood"),
        data.get("era"),
        data.get("tempo_bpm") is not None,
        data.get("year") is not None,
        data.get("tags"),
    ])


def compute_cold_start_score(candidate, preferences):
    score = 0.0
    reason_parts = []

    pref_genre = preferences.get("genre")
    pref_mood = preferences.get("mood")
    pref_era = preferences.get("era")
    pref_tempo = preferences.get("tempo_bpm")
    pref_year = preferences.get("year")
    pref_tags = preferences.get("tags", "")

    if pref_genre and candidate.genre:
        if str(candidate.genre).strip().lower() == pref_genre:
            score += 0.30
            reason_parts.append("genre match")

    if pref_mood and candidate.mood:
        if str(candidate.mood).strip().lower() == pref_mood:
            score += 0.20
            reason_parts.append("mood match")

    if pref_era and candidate.era:
        if str(candidate.era).strip().lower() == str(pref_era).strip().lower():
            score += 0.15
            reason_parts.append("era match")

    if pref_tempo is not None and candidate.tempo_bpm is not None:
        tempo_diff = abs(float(candidate.tempo_bpm) - float(pref_tempo))
        tempo_score = max(0.0, 0.20 - min(tempo_diff / 100.0, 0.20))
        score += tempo_score
        if tempo_score > 0.10:
            reason_parts.append("close tempo")

    if pref_year is not None and candidate.year is not None:
        year_diff = abs(int(candidate.year) - int(pref_year))
        year_score = max(0.0, 0.10 - min(year_diff / 100.0, 0.10))
        score += year_score
        if year_score > 0.05:
            reason_parts.append("close year")

    if pref_tags and candidate.tags:
        pref_tag_set = {t.strip() for t in pref_tags.split(",") if t.strip()}
        candidate_tag_set = {t.strip() for t in str(candidate.tags).lower().split(",") if t.strip()}
        overlap = pref_tag_set.intersection(candidate_tag_set)

        if overlap:
            score += min(0.15, 0.05 * len(overlap))
            reason_parts.append("tag overlap")

    return round(score, 4), reason_parts


def get_confidence_label(similarity):
    if similarity >= 0.8:
        return "high"
    if similarity >= 0.6:
        return "medium"
    return "low"

class HealthView(APIView):
    def get(self, request):
        return Response({"status": "ok"}, status=status.HTTP_200_OK)


class TrackListCreateView(APIView):

    def get(self, request):
        
        tracks = Track.objects.all().order_by("track_id")
        
        if tracks is not None:
            
            serialized_data = TrackReadSerializer(tracks, many=True).data
            
            return Response(
                serialized_data,
                status=status.HTTP_200_OK
            )
        
        else:
            return Response(
                {
                    "status": "error",
                    "message": "No tracks found"
                },
                status=status.HTTP_404_NOT_FOUND
            )


    def post(self, request):
        
        serializer = TrackSerializer(data=request.data)
        
        if serializer is not None:
            
            if serializer.is_valid():
                
                data = serializer.validated_data
                
                if data is not None:
                    
                    vec = build_feature_vector(
                        genre=data.get("genre"),
                        mood=data.get("mood"),
                        era=data.get("era"),
                        tempo_bpm=data.get("tempo_bpm"),
                        year=data.get("year"),
                        tags=data.get("tags", ""),
                    )
                    
                    if vec is not None:
                        
                        track, created = Track.objects.update_or_create(
                            track_id=data["track_id"],
                            defaults={
                                "title": data["title"],
                                "artist_name": data["artist_name"],
                                "genre": data.get("genre"),
                                "mood": data.get("mood"),
                                "era": data.get("era"),
                                "tempo_bpm": data.get("tempo_bpm"),
                                "year": data.get("year"),
                                "popularity": data.get("popularity"),
                                "tags": data.get("tags", ""),
                                "feature_vector": vec,
                            },
                        )
                        
                        if track is not None:
                            
                            if created == True:
                                response_status = status.HTTP_201_CREATED
                            else:
                                response_status = status.HTTP_200_OK
                            
                            if track.feature_vector is not None:
                                feature_dim = len(track.feature_vector)
                            else:
                                feature_dim = 0
                            
                            return Response(
                                {
                                    "status": "success",
                                    "created": created,
                                    "message": "Track stored and feature vector generated.",
                                    "track_id": track.track_id,
                                    "feature_vector_dim": feature_dim,
                                    "tempo_bpm": track.tempo_bpm,
                                    "year": track.year,
                                },
                                status=response_status,
                            )
                        
                        else:
                            return Response(
                                {
                                    "status": "error",
                                    "message": "Track creation failed"
                                },
                                status=status.HTTP_500_INTERNAL_SERVER_ERROR
                            )
                    
                    else:
                        return Response(
                            {
                                "status": "error",
                                "message": "Feature vector generation failed"
                            },
                            status=status.HTTP_500_INTERNAL_SERVER_ERROR
                        )
                
                else:
                    return Response(
                        {
                            "status": "error",
                            "message": "Validated data is empty"
                        },
                        status=status.HTTP_400_BAD_REQUEST
                    )
            
            else:
                return Response(
                    {
                        "status": "error",
                        "errors": serializer.errors
                    },
                    status=status.HTTP_400_BAD_REQUEST
                )
        
        else:
            return Response(
                {
                    "status": "error",
                    "message": "Serializer initialization failed"
                },
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )


class TrackDetailView(APIView):
    def get(self, request, track_id):
        transparent = request.query_params.get("transparent", "false").lower() == "true"

        try:
            track = Track.objects.get(track_id=str(track_id))
        except Track.DoesNotExist:
            return Response(
                {"error": "NOT_FOUND", "message": f"Track {track_id} not found"},
                status=status.HTTP_404_NOT_FOUND,
            )

        data = TrackReadSerializer(track).data

        if not transparent:
            return Response(data, status=status.HTTP_200_OK)

        all_tracks = list(Track.objects.all())
        vector_details = get_track_vector_transparency(track, all_tracks)

        return Response(
            {
                "track": data,
                "stored_feature_vector": track.feature_vector,
                "vector_details": vector_details,
            },
            status=status.HTTP_200_OK,
        )


class RecommendView(APIView):
    def get(self, request):
        mode = request.query_params.get("mode", "weighted")
        seed_ids_raw = request.query_params.get("seed_ids", "")
        if seed_ids_raw is not None:
    
            if len(seed_ids_raw.strip()) > 0:
                
                seed_ids = []
                
                parts = seed_ids_raw.split(",")
                
                for s in parts:
                    
                    if s is not None:
                        
                        cleaned = s.strip()
                        
                        if len(cleaned) > 0:
                            seed_ids.append(cleaned)
                        else:
                            pass
                    
                    else:
                        pass
            
            else:
                seed_ids = []

        else:
            seed_ids = []

        q = RecommendQuerySerializer(
            data={
                "seed_ids": seed_ids,
                "genre": request.query_params.get("genre"),
                "mood": request.query_params.get("mood"),
                "era": request.query_params.get("era"),
                "tempo_bpm": request.query_params.get("tempo_bpm"),
                "year": request.query_params.get("year"),
                "tags": request.query_params.get("tags"),
                "limit": request.query_params.get("limit", 10),
                "transparent": request.query_params.get("transparent", "false"),
            }
        )
        q.is_valid(raise_exception=True)

        data = q.validated_data
        limit = data["limit"]
        seed_ids = data["seed_ids"]
        transparent = data["transparent"]
        preferences = build_preference_summary(data)
        has_preferences = has_preference_input(data)

        seeds = []
        invalid_seeds = []
        missing = []

        if seed_ids:
            seeds = list(Track.objects.filter(track_id__in=seed_ids))

            found_ids = {str(t.track_id) for t in seeds}
            missing = [sid for sid in seed_ids if sid not in found_ids]

            if missing:
                return Response(
                    {"error": "SEED_NOT_FOUND", "missing": missing},
                    status=status.HTTP_404_NOT_FOUND,
                )

            valid_seeds = []

            for seed in seeds:
                result = validate_track_metadata(seed)

                if result["valid"]:
                    valid_seeds.append(seed)
                else:
                    invalid_seeds.append(
                        {
                            "track_id": seed.track_id,
                            "errors": result["errors"],
                        }
                    )

            if not valid_seeds and not has_preferences:
                return Response(
                    {
                        "error": "INVALID_SEEDS",
                        "invalid_seeds": invalid_seeds,
                    },
                    status=status.HTTP_400_BAD_REQUEST,
                )

            seeds = valid_seeds

        all_tracks = list(Track.objects.all())
        vector_result = build_combined_vectors_with_labels(all_tracks)
        vector_map = vector_result["vector_map"]
        transparency_map = vector_result["transparency_map"]

        candidates = list(Track.objects.exclude(track_id__in=seed_ids))

        active_mode = "cold_start"
        if seeds and has_preferences:
            active_mode = "hybrid"
        elif seeds:
            active_mode = "seed_based"

        if seeds is not None:
            
            if len(seeds) > 0:
                
                seed_metadata_summary = average_seed_metadata(seeds)
            
            else:
                seed_metadata_summary = {
                    "avg_seed_tempo_bpm": None,
                    "avg_seed_year": None,
                }

        else:
            seed_metadata_summary = {
                "avg_seed_tempo_bpm": None,
                "avg_seed_year": None,
            }

        seed_metadata = []
        seed_vectors = {}
        seed_vector_details = {}

        for seed in seeds:
            seed_id = str(seed.track_id)
            seed_metadata.append(TrackReadSerializer(seed).data)
            seed_vectors[seed_id] = vector_map.get(seed_id, [])
            seed_vector_details[seed_id] = transparency_map.get(seed_id, {})

        recommendations = []

        for candidate in candidates:
            candidate_id = str(candidate.track_id)
            candidate_vector = vector_map.get(candidate_id, [])

            final_similarity = 0.0
            explanations = []

            if seeds:
                seed_scores = []

                for seed_track in seeds:
                    seed_id = str(seed_track.track_id)
                    seed_vector = vector_map.get(seed_id, [])

                    if mode == "baseline":
                        score = compute_baseline_score(seed_vector, candidate_vector)
                        explanation = {
                            "source": "seed",
                            "seed_track_id": seed_track.track_id,
                            **get_baseline_breakdown(seed_vector, candidate_vector),
                            "raw_inputs": {
                                "seed_tempo_bpm": seed_track.tempo_bpm,
                                "candidate_tempo_bpm": candidate.tempo_bpm,
                                "seed_year": seed_track.year,
                                "candidate_year": candidate.year,
                            },
                        }
                    else:
                        score = compute_weighted_score(
                            seed_track,
                            candidate,
                            seed_vector,
                            candidate_vector,
                        )
                        explanation = {
                            "source": "seed",
                            "seed_track_id": seed_track.track_id,
                            **get_score_breakdown(
                                seed_track,
                                candidate,
                                seed_vector,
                                candidate_vector,
                            ),
                        }

                    if transparent:
                        explanation["seed_vector"] = seed_vector
                        explanation["candidate_vector"] = candidate_vector
                        explanation["seed_vector_details"] = transparency_map.get(seed_id, {})
                        explanation["candidate_vector_details"] = transparency_map.get(candidate_id, {})

                    seed_scores.append(score)
                    explanations.append(explanation)

                final_similarity = sum(seed_scores) / len(seed_scores)

            preference_score = 0.0
            preference_reasons = []

            if has_preferences:
                preference_score, preference_reasons = compute_cold_start_score(candidate, preferences)
                explanations.append(
                    {
                        "source": "preferences",
                        "preferences": preferences,
                        "score": preference_score,
                        "reasons": preference_reasons,
                    }
                )

            if seeds and has_preferences:
                final_similarity = round((0.7 * final_similarity) + (0.3 * preference_score), 4)
            elif not seeds and has_preferences:
                final_similarity = round(preference_score, 4)
            else:
                final_similarity = round(final_similarity, 4)

            total_genre_matches = 0
            total_mood_matches = 0
            total_era_matches = 0
            total_tempo_match = 0.0
            total_year_match = 0.0

            for e in explanations:
                matches = e.get("matches", {})
                total_genre_matches += matches.get("genre_match", 0)
                total_mood_matches += matches.get("mood_match", 0)
                total_era_matches += matches.get("era_match", 0)
                total_tempo_match += matches.get("tempo_match", 0.0)
                total_year_match += matches.get("year_match", 0.0)

            avg_tempo_match = 0.0
            avg_year_match = 0.0

            if seeds and mode == "weighted":
                seed_explanations = [e for e in explanations if e.get("source") == "seed"]
                if seed_explanations:
                    avg_tempo_match = round(total_tempo_match / len(seed_explanations), 4)
                    avg_year_match = round(total_year_match / len(seed_explanations), 4)

            recommendation_item = {
                "track_id": candidate.track_id,
                "title": format_display_text(candidate.title),
                "artist_name": format_display_text(candidate.artist_name),
                "genre": format_display_text(candidate.genre),
                "mood": format_display_text(candidate.mood),
                "era": candidate.era,
                "tempo_bpm": candidate.tempo_bpm,
                "year": candidate.year,
                "similarity": final_similarity,
                "similarity_type": active_mode,
                "confidence": get_confidence_label(final_similarity),
                "explanation": {
                    "seed_count": len(seeds),
                    "avg_seed_tempo_bpm": seed_metadata_summary["avg_seed_tempo_bpm"],
                    "avg_seed_year": seed_metadata_summary["avg_seed_year"],
                    "candidate_tempo_bpm": candidate.tempo_bpm,
                    "candidate_year": candidate.year,
                    "genre_matches": total_genre_matches,
                    "mood_matches": total_mood_matches,
                    "era_matches": total_era_matches,
                    "avg_tempo_match": avg_tempo_match,
                    "avg_year_match": avg_year_match,
                    "preference_input": preferences,
                    "per_seed": explanations,
                },
            }

            if transparent:
                recommendation_item["candidate_metadata"] = TrackReadSerializer(candidate).data
                recommendation_item["candidate_vector"] = candidate_vector
                recommendation_item["candidate_vector_details"] = transparency_map.get(candidate_id, {})
                recommendation_item["seed_vectors"] = seed_vectors
                recommendation_item["seed_vector_details"] = seed_vector_details

            recommendations.append(recommendation_item)

        recommendations.sort(key=lambda x: x["similarity"], reverse=True)

        if seeds and mode == "weighted":
            recommendations = rerank_with_diversity(recommendations)
            recommendations.sort(key=lambda x: x["similarity"], reverse=True)

        recommendations = recommendations[:limit]

        response_data = {
            "mode": mode,
            "recommendation_mode": active_mode,
            "requested_seeds": seed_ids,
            "used_seeds": [str(seed.track_id) for seed in seeds],
            "invalid_seeds": invalid_seeds,
            "seed_summary": seed_metadata_summary,
            "preference_input": preferences,
            "recommendations": recommendations,
        }

        if transparent:
            response_data["seed_metadata"] = seed_metadata
            response_data["seed_vectors"] = seed_vectors
            response_data["seed_vector_details"] = seed_vector_details
            response_data["vector_labels"] = vector_result["combined_labels"]
            response_data["metadata_labels"] = vector_result["metadata_labels"]
            response_data["tfidf_labels"] = vector_result["tfidf_labels"]

        return Response(response_data, status=status.HTTP_200_OK)


class SimilarityDistributionView(APIView):
    def get(self, request):
        seed_ids_raw = request.query_params.get("seed_ids", "")
        if seed_ids_raw is not None:
            
            if len(seed_ids_raw.strip()) > 0:
                
                parts = seed_ids_raw.split(",")
                seed_ids = []
                
                for s in parts:
                    
                    if s is not None:
                        
                        cleaned = s.strip()
                        
                        if len(cleaned) > 0:
                            seed_ids.append(cleaned)
                        else:
                            pass
                    
                    else:
                        pass
            
            else:
                seed_ids = []

        else:
            seed_ids = []
        bins = int(request.query_params.get("bins", 20))
        limit = int(request.query_params.get("limit", 10))
        mode = request.query_params.get("mode", "weighted")

        q = RecommendQuerySerializer(
            data={
                "seed_ids": seed_ids,
                "limit": limit,
                "transparent": request.query_params.get("transparent", "false"),
            }
        )
        q.is_valid(raise_exception=True)

        seed_ids = q.validated_data["seed_ids"]

        if not seed_ids:
            return Response(
                {"error": "NO_SEEDS", "message": "Provide seed_ids, e.g. ?seed_ids=101,102"},
                status=status.HTTP_400_BAD_REQUEST,
            )

        seeds = list(Track.objects.filter(track_id__in=seed_ids))

        found_ids = set()

        if seeds is not None:
            
            if len(seeds) > 0:
                
                for t in seeds:
                    
                    if t is not None:
                        
                        if hasattr(t, "track_id"):
                            
                            if t.track_id is not None:
                                found_ids.add(str(t.track_id))
                            else:
                                pass
                        
                        else:
                            pass
                    
                    else:
                        pass
            
            else:
                found_ids = set()

        else:
            found_ids = set()
        missing = [sid for sid in seed_ids if sid not in found_ids]

        if missing:
            return Response(
                {"error": "SEED_NOT_FOUND", "missing": missing},
                status=status.HTTP_404_NOT_FOUND,
            )

        valid_seeds = []
        invalid_seeds = []

        for seed in seeds:
            result = validate_track_metadata(seed)

            if result["valid"]:
                valid_seeds.append(seed)
            else:
                invalid_seeds.append(
                    {
                        "track_id": seed.track_id,
                        "errors": result["errors"],
                    }
                )

        if not valid_seeds:
            return Response(
                {
                    "error": "INVALID_SEEDS",
                    "invalid_seeds": invalid_seeds,
                },
                status=status.HTTP_400_BAD_REQUEST,
            )

        seeds = valid_seeds

        all_tracks = list(Track.objects.all())
        vector_result = build_combined_vectors_with_labels(all_tracks)
        vector_map = vector_result["vector_map"]

        candidates = list(Track.objects.exclude(track_id__in=seed_ids))
        scored_candidates = []

        for candidate in candidates:
            candidate_id = str(candidate.track_id)
            candidate_vector = vector_map.get(candidate_id, [])

            seed_scores = []

            for seed_track in seeds:
                seed_id = str(seed_track.track_id)
                seed_vector = vector_map.get(seed_id, [])

                if mode == "baseline":
                    score = compute_baseline_score(seed_vector, candidate_vector)
                else:
                    score = compute_weighted_score(
                        seed_track,
                        candidate,
                        seed_vector,
                        candidate_vector,
                    )

                seed_scores.append(score)

            final_similarity = round(sum(seed_scores) / len(seed_scores), 4)

            scored_candidates.append(
                {
                    "track_id": candidate.track_id,
                    "title": candidate.title,
                    "similarity": final_similarity,
                }
            )

        all_scores = [item["similarity"] for item in scored_candidates]

        ranked = sorted(scored_candidates, key=lambda x: x["similarity"], reverse=True)

        if mode == "weighted":
            ranked = rerank_with_diversity(ranked)
            ranked = sorted(ranked, key=lambda x: x["similarity"], reverse=True)

        final_scores = [item["similarity"] for item in ranked[:limit]]

        response_data = {
            "requested_seeds": seed_ids,
            "used_seeds": [str(seed.track_id) for seed in seeds],
            "invalid_seeds": invalid_seeds,
            "seed_summary": average_seed_metadata(seeds),
            "groups": len(seeds),
            "total": len(candidates),
            "num_candidates": len(candidates),
            "bins": bins,
            "all_candidates_distribution": similarity_distribution_from_scores(all_scores, bins=bins),
            "final_recommendations_distribution": similarity_distribution_from_scores(final_scores, bins=bins), 
            "top_hidden_candidates": ranked[:20],
        }

        return Response(response_data, status=status.HTTP_200_OK)