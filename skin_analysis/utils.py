import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
# import file
from model_result import dry_percent as dry_file
from model_result import acne_detect as acne_file
from model_result import oil_detect as oil_file
from model_result import skin_tone_analysis as skin_tone_result
from model_result import skin_details as scores_details

def detect_face(image):
    """ Detects face and extracts the skin region from an image """
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Initialize Mediapipe Face Detection
    mp_face_detection = mp.solutions.face_detection
    face_detection = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)
    results = face_detection.process(image_rgb)
    if results.detections:
        for detection in results.detections:
            bboxC = detection.location_data.relative_bounding_box
            h, w, _ = image.shape
            x, y, w, h = int(bboxC.xmin * w), int(bboxC.ymin * h), int(bboxC.width * w), int(bboxC.height * h)
            return image[y:y+h, x:x+w]  # Crop face region
    return None

def analyze_skin(image):
    """ Preprocesses and predicts skin condition using AI model """
    image = cv2.resize(image, (128, 128))  # Resize for model input
    image = image / 255.0  # Normalize
    image = np.expand_dims(image, axis=0)  # Expand dimensions for model

    # Load pre-trained AI skin analysis model (Example: Custom CNN Model)
    model = load_model("models/skin_analysis_model_2.h5")
    predictions = model.predict(image)  # AI prediction
    labels = ["Acne", "Oily", "Dry"]
    return labels[np.argmax(predictions)]  # Return predicted label

def analyze_image(image_path):
    # Load selfie image
    image_path = image_path
    image = cv2.imread(image_path)

    if image is None:
      print(f"❌ OpenCV cannot read {image_path}. Check if it's a valid image.")
      exit()

    print("✅ OpenCV successfully loaded the image.")

    # Step 1: Detect Face
    face_region = detect_face(image)
    if face_region is None:
        print("❌ No face detected or cropped image is empty!")
        return {"error": "No face detected. Try a clearer image."}
    skin_details = {}
    skin_tone = skin_tone_result.skin_tone_analysis(image_path)
    skin_details["skin_tone"] = skin_tone

    skin_scores = scores_details.details_scores(image_path)
    print("skin_scores------", skin_scores)
    skin_details["redness_score"] = f'{skin_scores[0]}%'
    skin_details["pigmentation_score"] = f'{skin_scores[1]}%'
    skin_details["pores_score"] = f'{skin_scores[2]}%'
    print("skin_details" ,skin_details)

    if face_region is not None:
        # Step 2: AI Analysis
        skin_condition = analyze_skin(face_region)
        skin_details['skin_type'] = skin_condition
        print(f"🔍 Detected Skin Condition: {skin_condition}")
        if skin_condition =='Dry':
            dry_percentage = dry_file.predict_dry_skin(image_path)
            skin_details['percentage'] = f'{dry_percentage}%'
            skin_details['acne_types'] = False
            skin_details['ingredients'] = "Hyaluronic Acid, Glycerin, Ceramides, Squalane, Aloe Vera"
            skin_details['avoid'] = "Alcohol-based toners, Harsh exfoliants"
            print('dry_percentage------->',dry_percentage)
        elif skin_condition =='Oily':
            oily_percentages = oil_file.calculate_oiliness(image_path)
            skin_details['percentage'] = f'{oily_percentages}%'
            skin_details['acne_types'] = False
            skin_details['ingredients'] = "Salicylic Acid, Zinc, Niacinamide, Green Tea"
            skin_details['avoid'] = "Thick creams, Comedogenic oils"
            print("oily_percentages------->",oily_percentages)
        else:
            acne_type, confidence = acne_file.predict_acne_type(image_path)
            skin_details['acne_types'] = acne_type
            skin_details['percentage'] = False
            if(acne_type in ["Whiteheads", "Blackheads"]):
                skin_details['ingredients'] = "Salicylic Acid (BHA), Niacinamide, Clay Masks"
            elif(acne_type == "Papule"):
                skin_details['ingredients'] = "Benzoyl Peroxide, Tea Tree Oil, Retinol"
            elif(acne_type == "Pustule"):
                skin_details['ingredients'] = "Benzoyl Peroxide, Sulfur, Centella Asiatica"
            else:
                skin_details['ingredients'] = "Salicylic acid, benzoyl peroxide, and azelaic acid"
            print("predicted_class_label------>",acne_type)
            print("confidence-------->",confidence)
    else:
        print("❌ No face detected in the image.")
    return skin_details