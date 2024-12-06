import os
import json
from PIL import Image

import numpy as np
import tensorflow as tf
import streamlit as st

working_dir = os.path.dirname(os.path.abspath(__file__))
model_path = f"{working_dir}/trained_model/plant_disease_prediction_model.h5"
# Load the pre-trained model
model = tf.keras.models.load_model(model_path)

# loading the class names
class_indices = json.load(open(f"{working_dir}/class_indices.json"))

# Dictionary mapping disease classes to their cures and medicines
disease_cures = {
    "Apple___Apple_scab": {
        "cure": "Apply fungicide and remove infected leaves to control Apple scab.",
        "medicine": "Fungicides like Captan, Myclobutanil, or Thiophanate-methyl."
    },
    "Apple___Black_rot": {
        "cure": "Prune infected twigs and apply fungicide to manage Black rot in apples.",
        "medicine": "Fungicides like Chlorothalonil or Thiophanate-methyl."
    },
    "Apple___Cedar_apple_rust": {
        "cure": "Remove cedar trees near apple orchards and apply fungicide to control Cedar apple rust.",
        "medicine": "Fungicides like Myclobutanil, Propiconazole, or Thiophanate-methyl."
    },
    "Apple___healthy": {
        "cure": "No specific cure needed for healthy apple trees.",
        "medicine": ""
    },
    "Blueberry___healthy": {
        "cure": "No specific cure needed for healthy blueberry plants.",
        "medicine": ""
    },
    "Cherry_(including_sour)___Powdery_mildew": {
        "cure": "Apply fungicide and improve air circulation to manage Powdery mildew in cherries.",
        "medicine": "Fungicides like Sulfur, Potassium bicarbonate, or Azoxystrobin."
    },
    "Cherry_(including_sour)___healthy": {
        "cure": "No specific cure needed for healthy cherry trees.",
        "medicine": ""
    },
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": {
        "cure": "Rotate crops and apply fungicide to control Cercospora leaf spot in corn.",
        "medicine": "Fungicides like Azoxystrobin, Pyraclostrobin, or Chlorothalonil."
    },
    "Corn_(maize)___Common_rust_": {
        "cure": "Plant resistant varieties and apply fungicide to manage Common rust in corn.",
        "medicine": "Fungicides like Propiconazole or Tebuconazole."
    },
    "Corn_(maize)___Northern_Leaf_Blight": {
        "cure": "Rotate crops and apply fungicide to control Northern Leaf Blight in corn.",
        "medicine": "Fungicides like Chlorothalonil or Azoxystrobin."
    },
    "Corn_(maize)___healthy": {
        "cure": "No specific cure needed for healthy corn plants.",
        "medicine": ""
    },
    "Grape___Black_rot": {
        "cure": "Prune infected canes and apply fungicide to manage Black rot in grapes.",
        "medicine": "Fungicides like Myclobutanil, Boscalid, or Captan."
    },
    "Grape___Esca_(Black_Measles)": {
        "cure": "Prune infected wood and apply fungicide to control Esca in grapes.",
        "medicine": "Fungicides like Propiconazole, Fludioxonil, or Phosphorous acid."
    },
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": {
        "cure": "Apply fungicide and prune infected leaves to manage Leaf blight in grapes.",
        "medicine": "Fungicides like Copper sulfate, Mancozeb, or Myclobutanil."
    },
    "Grape___healthy": {
        "cure": "No specific cure needed for healthy grape vines.",
        "medicine": ""
    },
    "Orange___Haunglongbing_(Citrus_greening)": {
        "cure": "Remove infected trees and control insect vectors to manage Citrus greening in oranges.",
        "medicine": "Systemic antibiotics like Tetracycline or Penicillin."
    },
    "Peach___Bacterial_spot": {
        "cure": "Apply copper sprays and prune infected branches to manage Bacterial spot in peaches.",
        "medicine": "Copper-based fungicides like Copper hydroxide or Copper oxychloride."
    },
    "Peach___healthy": {
        "cure": "No specific cure needed for healthy peach trees.",
        "medicine": ""
    },
    "Pepper,_bell___Bacterial_spot": {
        "cure": "Apply copper-based fungicides and rotate crops to control Bacterial spot in bell peppers.",
        "medicine": "Copper-based fungicides like Copper hydroxide or Copper oxychloride."
    },
    "Pepper,_bell___healthy": {
        "cure": "No specific cure needed for healthy bell pepper plants.",
        "medicine": ""
    },
    "Potato___Early_blight": {
        "cure": "Remove infected leaves and apply fungicide to control Early blight in potatoes.",
        "medicine": "Fungicides like Chlorothalonil or Mancozeb."
    },
    "Potato___Late_blight": {
        "cure": "Remove infected leaves and apply fungicide to manage Late blight in potatoes.",
        "medicine": "Fungicides like Phosphorous acid or Azoxystrobin."
    },
    "Potato___healthy": {
        "cure": "No specific cure needed for healthy potato plants.",
        "medicine": ""
    },
    "Raspberry___healthy": {
        "cure": "No specific cure needed for healthy raspberry plants.",
        "medicine": ""
    },
    "Soybean___healthy": {
        "cure": "No specific cure needed for healthy soybean plants.",
        "medicine": ""
    },
    "Squash___Powdery_mildew": {
        "cure": "Apply fungicide and improve air circulation to manage Powdery mildew in squash.",
        "medicine": "Fungicides like Sulfur, Potassium bicarbonate, or Myclobutanil."
    },
    "Strawberry___Leaf_scorch": {
        "cure": "Apply fungicide and improve soil drainage to manage Leaf scorch in strawberries.",
        "medicine": "Fungicides like Captan, Propiconazole, or Fenhexamid."
    },
    "Strawberry___healthy": {
        "cure": "No specific cure needed for healthy strawberry plants.",
        "medicine": ""
    },
    "Tomato___Bacterial_spot": {
        "cure": "Apply copper-based fungicides and rotate crops to control Bacterial spot in tomatoes.",
        "medicine": "Copper-based fungicides like Copper hydroxide or Copper oxychloride."
    },
    "Tomato___Early_blight": {
        "cure": "Remove infected leaves and apply fungicide to manage Early blight in tomatoes.",
        "medicine": "Fungicides like Chlorothalonil or Mancozeb."
    },
    "Tomato___Late_blight": {
        "cure": "Remove infected leaves and apply fungicide to control Late blight in tomatoes.",
        "medicine": "Fungicides like Phosphorous acid or Azoxystrobin."
    },
    "Tomato___Leaf_Mold": {
        "cure": "Improve air circulation and remove infected leaves to manage Leaf Mold in tomatoes.",
        "medicine": "Fungicides like Chlorothalonil or Mancozeb."
    },
    "Tomato___Septoria_leaf_spot": {
        "cure": "Apply fungicide and remove infected leaves to control Septoria leaf spot in tomatoes.",
        "medicine": "Fungicides like Azoxystrobin or Propiconazole."
    },
    "Tomato___Spider_mites Two-spotted_spider_mite": {
        "cure": "Apply miticide and improve air circulation to manage Spider mites in tomatoes.",
        "medicine": "Miticides like Abamectin or Spiromesifen."
    },
    "Tomato___Target_Spot": {
        "cure": "Remove infected leaves and apply fungicide to manage Target Spot in tomatoes.",
        "medicine": "Fungicides like Chlorothalonil or Mancozeb."
    },
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": {
        "cure": "Control whiteflies and remove infected plants to manage Tomato Yellow Leaf Curl Virus.",
        "medicine": "Systemic insecticides like Imidacloprid or Thiamethoxam."
    },
    "Tomato___Tomato_mosaic_virus": {
        "cure": "Control aphids and remove infected plants to manage Tomato Mosaic Virus.",
        "medicine": "Systemic insecticides like Acetamiprid or Thiamethoxam."
    },
    "Tomato___healthy": {
        "cure": "No specific cure needed for healthy tomato plants.",
        "medicine": ""
    }
}

# Function to get cure and medicines for predicted disease
def get_disease_info(predicted_class_name):
    return disease_cures.get(predicted_class_name, {'cure': 'No cure information available.', 'medicine': 'No medicine information available.'})

# Function to Load and Preprocess the Image using Pillow
def load_and_preprocess_image(image_path, target_size=(224, 224)):
    # Load the image
    img = Image.open(image_path)
    # Resize the image
    img = img.resize(target_size)
    # Convert the image to a numpy array
    img_array = np.array(img)
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    # Scale the image values to [0, 1]
    img_array = img_array.astype('float32') / 255.
    return img_array

# Function to Predict the Class of an Image
def predict_image_class(model, image_path, class_indices):
    preprocessed_img = load_and_preprocess_image(image_path)
    predictions = model.predict(preprocessed_img)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    predicted_class_name = class_indices[str(predicted_class_index)]
    return predicted_class_name

# Streamlit App
st.markdown(
    """
    <style>
    .title {
        background-color: black;
        color: white;
        padding: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown('<h1 class="title">Plant Disease Classifier</h1>', unsafe_allow_html=True)

st.markdown(
     f"""
     <style>
     .stApp {{
         background-image: url("https://www.innovationnewsnetwork.com/wp-content/uploads/2020/11/self-watering-soil-1024x576.jpg");
         background-attachment: fixed;
         background-size: cover
     }}
     </style>
     """,
     unsafe_allow_html=True
 )

uploaded_image = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    image = Image.open(uploaded_image)
    col1, col2 = st.columns(2)

    with col1:
        resized_img = image.resize((150, 150))
        st.image(resized_img)

    with col2:
        if st.button('Classify'):
            # Preprocess the uploaded image and predict the class
            prediction = predict_image_class(model, uploaded_image, class_indices)
            # Display the predicted disease
            st.markdown(
                f'<div style="background-color: black; color: white; padding: 10px;">'
                f'Prediction: {str(prediction)}'
                f'</div>',
                unsafe_allow_html=True
            )
            # Display the cure and medicine if available
            disease_info = get_disease_info(prediction)
            st.markdown(
                f'<div style="background-color: black; color: white; padding: 10px; margin-top: 10px;">'
                f'Cure: {disease_info["cure"]}'
                f'</div>',
                unsafe_allow_html=True
            )
            st.markdown(
                f'<div style="background-color: black; color: white; padding: 10px; margin-top: 10px;">'
                f'Medicine: {disease_info["medicine"]}'
                f'</div>',
                unsafe_allow_html=True
            )
