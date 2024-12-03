
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import io
from fastapi import FastAPI, UploadFile, File

# Initialize FastAPI app
app = FastAPI()

# Load the pre-trained model
model = load_model("C:/Users/YOSHITA/OneDrive/Desktop/smbhv/demodayexportease/imageClassifier/classification_model.h5")


class_names = ['bed', 'chair', 'sofa', 'swivelchair', 'table']

def predict_image(img: Image.Image):
    """
    Preprocess the image and make predictions using the loaded model.
    """
    # Resize and preprocess the image
    img = img.resize((224, 224))  # Adjust to your model's input size
    img_array = image.img_to_array(img)  # Convert to array
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = preprocess_input(img_array)  # Preprocess input for VGG16 or similar models
    
    # Predict using the model
    predictions = model.predict(img_array)
    return predictions

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    """
    Accepts an uploaded image file and returns the predicted class name.
    """
    try:
        # Read the uploaded image
        image_bytes = await file.read()
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")  # Convert to RGB if needed
        
        # Get model predictions
        predictions = predict_image(img)
        
        # Extract the class name based on the highest probability
        max_index = predictions.argmax()  # Index of the max probability
        result_label = class_names[max_index]  # Get the corresponding class name
        
        return {"class_name": result_label}
    except Exception as e:
        return {"error": str(e)}
