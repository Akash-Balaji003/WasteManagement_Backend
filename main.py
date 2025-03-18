import logging
from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from io import BytesIO
from DB_Interface import login_user, register_user, predict_waste_categoryV2

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

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
    prediction = predict_waste_categoryV2(image)
    return {"predicted_category": prediction}
