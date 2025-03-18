import mysql.connector
from passlib.context import CryptContext
from fastapi import HTTPException

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

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
