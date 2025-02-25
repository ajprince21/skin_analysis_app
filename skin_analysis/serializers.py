from rest_framework import serializers
from .models import SkinAnalysis

class SkinAnalysisSerializer(serializers.ModelSerializer):
    class Meta:
        model = SkinAnalysis
        fields = '__all__'