from django.urls import path
from .views import (
    dashboard_view,
    HealthView,
    TrackListCreateView,
    TrackDetailView,
    RecommendView,
    SimilarityDistributionView,
)

urlpatterns = [
    path("", dashboard_view, name="dashboard"),
    path("health", HealthView.as_view(), name="health"),
    path("tracks", TrackListCreateView.as_view(), name="tracks"),
    path("tracks/<str:track_id>", TrackDetailView.as_view(), name="track-detail"),
    path("recommendations", RecommendView.as_view(), name="recommendations"),
    path("recommendations/distribution", SimilarityDistributionView.as_view(), name="recommendations-distribution"),
]