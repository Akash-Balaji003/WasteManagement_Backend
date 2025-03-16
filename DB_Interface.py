import mysql.connector
from passlib.context import CryptContext
from fastapi import HTTPException
from keras_preprocessing.image import ImageDataGenerator
from PIL import Image
import numpy as np
import tensorflow as tf
import os

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


# Get the current directory (where main.py is)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Point to the images folder relative to the backend folder
DATASET_PATH = os.path.join(BASE_DIR, "Dataset", "images")

# Image parameters
IMG_SIZE = 224
BATCH_SIZE = 32

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

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def hash_password(password: str):
    return pwd_context.hash(password)

def get_db_connection():
    return mysql.connector.connect(
        host="devops-project-db.cza0amg6av33.ap-northeast-1.rds.amazonaws.com",
        port=3306,
        user="Akash",
        password="Akash003!",
        database="waste_management"
    )

def login_user(user_data: dict):
    connection = get_db_connection()
    cursor = connection.cursor(dictionary=True)

    try:
        # Fetch user details
        query_user = """SELECT user_id, username, phone_number, password FROM Users WHERE phone_number = %s"""
        cursor.execute(query_user, (user_data['phone_number'],))
        db_user = cursor.fetchone()

        if not db_user or not verify_password(user_data['password'], db_user['password']):
            raise HTTPException(status_code=401, detail="Invalid credentials")

        return {
            "user_id": db_user['user_id'],
            "username": db_user['username'],
            "phone_number": db_user['phone_number'],
        }
    
    except mysql.connector.Error as err:
        raise HTTPException(status_code=500, detail=f"Error: {err}")
    
    finally:
        cursor.close()
        connection.close()

def register_user(user_data: dict):
    connection = get_db_connection()
    cursor = connection.cursor(dictionary=True)

    try:
        # Check if user already exists with the same phone number
        check_query = "SELECT user_id FROM Users WHERE phone_number = %s"
        cursor.execute(check_query, (user_data['phone_number'],))
        existing_user = cursor.fetchone()

        if existing_user:
            raise HTTPException(status_code=400, detail="Phone number already registered")

        # Hash the password before storing it
        hashed_password = hash_password(user_data['password'])

        # Insert new user data
        insert_query = """INSERT INTO Users (username, phone_number, password) VALUES (%s, %s, %s)"""
        cursor.execute(insert_query, (user_data['username'], user_data['phone_number'], hashed_password))
        connection.commit()

        return {"message": "User registered successfully"}

    except mysql.connector.Error as err:
        raise HTTPException(status_code=500, detail=f"Error: {err}")

    finally:
        cursor.close()
        connection.close()

def predict_waste_category(image_path):

    # Load the trained model
    model = tf.keras.models.load_model("./models/waste_classifier.h5")

    class_types = train_data.class_indices
    
    # Load and preprocess the image
    img = Image.open(image_path).convert('RGB').resize((224, 224))
    img_array = np.array(img) / 255.0  # Rescale
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    
    # Make prediction
    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    class_labels = {v: k for k, v in class_types.items()}
    predicted_class = class_labels[predicted_class_index]
    
    return predicted_class

def predict_waste_categoryV2(image: Image.Image):

    # Load the trained model
    model = tf.keras.models.load_model("./models/waste_classifier.h5")

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

# Example usage:
# result = predict_waste_category("./Test_images/Test_image3.png")
# print("Predicted category:", result)
