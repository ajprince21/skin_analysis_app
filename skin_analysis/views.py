from rest_framework.views import APIView
from rest_framework.response import Response
from django.core.files.storage import default_storage
from rest_framework import status
from .models import SkinAnalysis
from .utils import analyze_image 
import cv2
import numpy as np
import logging
import tempfile

logger = logging.getLogger(__name__)
class ImageUploadView(APIView):
    def post(self, request):
        if 'image' not in request.FILES:
            return Response({"error": "No image file provided."}, status=status.HTTP_400_BAD_REQUEST)

        file = request.FILES['image']
        
        # Validate file type
        if not file.name.endswith(('.png', '.jpg', '.jpeg')):
            return Response({"error": "File type not supported, please upload an image."}, status=status.HTTP_400_BAD_REQUEST)

        # Use a NamedTemporaryFile for processing
        with tempfile.NamedTemporaryFile(delete=True) as temp_file:
            for chunk in file.chunks():  # Handle large files
                temp_file.write(chunk)
            temp_file.flush()  # Ensure the file is written
            temp_file.seek(0)  # Go back to the beginning of the file

            logger.info(f'File saved at: {temp_file.name}')
            analysis_results = analyze_image(temp_file.name)

        
        if analysis_results is None: 
            return Response({"error": "Image processing failed."}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        # Save the analysis results to the database
        skin_analysis_instance = SkinAnalysis(
            image=file,
            skin_tone=analysis_results.get('skin_tone', ''),
            skin_type=analysis_results.get('skin_type', ''),
            percentage=analysis_results.get('percentage', ''),
            acne_types=analysis_results.get('acne_types', ''),
            redness_score=analysis_results.get('redness_score', ''),
            pores_score=analysis_results.get('pores_score', ''),
            pigmentation_score=analysis_results.get('pigmentation_score', '')
        )
        skin_analysis_instance.save()

        # Return the analysis results as a response
        return Response(analysis_results, status=status.HTTP_201_CREATED)