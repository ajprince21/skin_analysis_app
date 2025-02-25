from django.urls import path, include
from rest_framework.routers import DefaultRouter
from .views import SkinAnalysisViewSet

router = DefaultRouter()
router.register(r'skin_analysis', SkinAnalysisViewSet)

urlpatterns = [
    path('', include(router.urls)),
]