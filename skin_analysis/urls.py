from django.urls import path
from .views import ImageUploadView

urlpatterns = [
    path('skin-analysis/', ImageUploadView.as_view(), name='image-upload'),
]