import logging
import os
from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from io import BytesIO
import numpy as np
import tensorflow as tf
from keras_preprocessing.image import ImageDataGenerator

from DB_Interface import login_user, register_user

app = FastAPI()

# Get the current directory (where main.py is)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Point to the images folder relative to the backend folder
DATASET_PATH = os.path.join(BASE_DIR, "Dataset", "images")
MODEL_PATH = os.path.join(BASE_DIR, "models", "waste_classifier.h5")

# Image parameters
IMG_SIZE = 224
BATCH_SIZE = 32

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Load the trained model
model = tf.keras.models.load_model(MODEL_PATH)

# Data Augmentation and Preprocessing
datagen = ImageDataGenerator(
    rescale=1.0 / 255, 
    rotation_range=30, 
    width_shift_range=0.2, 
    height_shift_range=0.2, 
    shear_range=0.2,
    zoom_range=0.2, 
    horizontal_flip=True, 
    validation_split=0.2  # 20% validation split
)

train_data = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="training"
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)

def predict_waste_category(image: Image.Image):

    # Load the trained model
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)

    class_types = train_data.class_indices

    # Preprocess the image
    img = image.convert('RGB').resize((224, 224))
    img_array = np.array(img) / 255.0  # Rescale
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    
    # Make prediction
    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    class_labels = {v: k for k, v in class_types.items()}
    predicted_class = class_labels[predicted_class_index]
    
    return predicted_class

@app.get("/test")
async def test():
    logging.info("Test...")  # Debugging
    return {"Test": "Working with Github Actions"}

@app.post("/login")
async def login(request: Request):
    logging.info("Incoming POST request to /login")
    try:
        user_data = await request.json()
        logging.info("Received login data: %s", user_data)

        # Call login_user function from DB_Interface
        response = login_user(user_data)

        logging.info("Login successful.")
        return response
    except Exception as e:
        logging.error("Error during login: %s", str(e), exc_info=True)
        raise HTTPException(status_code=401, detail="Invalid credentials")

@app.post("/register")
async def register(request: Request):
    logging.info("Incoming POST request to /register")
    try:
        user_data = await request.json()
        logging.info("Received registration data: %s", user_data)

        # Call register_user function from DB_Interface
        response = register_user(user_data)

        logging.info("User registered successfully.")
        return response
    except Exception as e:
        logging.error("Error during registration: %s", str(e), exc_info=True)
        raise HTTPException(status_code=500, detail="Registration failed")

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image = Image.open(BytesIO(await file.read()))
    prediction = predict_waste_category(image)
    return {"predicted_category": prediction}
