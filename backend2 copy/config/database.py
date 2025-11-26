import os
from motor.motor_asyncio import AsyncIOMotorClient
from dotenv import load_dotenv

load_dotenv()

# MongoDB Configuration
MONGODB_URL = os.getenv("MONGODB_URL", "mongodb+srv://ambot_db_user:yW3oWYg4V9RJV2ws@ambot.0o8lghp.mongodb.net/?appName=ambot")
DATABASE_NAME = os.getenv("DATABASE_NAME", "ambot_db")

class Database:
    client: AsyncIOMotorClient = None
    
db = Database()

async def connect_to_mongo():
    """Create database connection."""
    db.client = AsyncIOMotorClient(MONGODB_URL)
    print("Connected to MongoDB!")

async def close_mongo_connection():
    """Close database connection."""
    if db.client:
        db.client.close()
        print("MongoDB connection closed.")

def get_database():
    """Get database instance."""
    return db.client[DATABASE_NAME]
