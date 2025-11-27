from datetime import timedelta
from fastapi import APIRouter, HTTPException, status, Depends
from config.database import get_database
from config.settings import ACCESS_TOKEN_EXPIRE_MINUTES
from models.user import UserCreate, UserLogin, UserResponse, Token
from utils.security import (
    verify_password,
    get_password_hash,
    create_access_token,
    get_current_user,
    get_current_admin_user
)

router = APIRouter(prefix="/auth", tags=["Authentication"])


@router.post("/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def register(user_data: UserCreate):
    """Register a new admin user."""
    db = get_database()
    
    # Check if user already exists
    existing_user = await db.users.find_one({
        "$or": [
            {"email": user_data.email},
            {"username": user_data.username}
        ]
    })
    
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="User with this email or username already exists"
        )
    
    # Create new user
    hashed_password = get_password_hash(user_data.password)
    
    new_user = {
        "email": user_data.email,
        "username": user_data.username,
        "hashed_password": hashed_password,
        "is_active": True,
        "is_admin": True,
        "created_at": None,
        "updated_at": None
    }
    
    from datetime import datetime
    new_user["created_at"] = datetime.utcnow()
    new_user["updated_at"] = datetime.utcnow()
    
    result = await db.users.insert_one(new_user)
    
    # Fetch and return the created user
    created_user = await db.users.find_one({"_id": result.inserted_id})
    
    return UserResponse(
        id=str(created_user["_id"]),
        email=created_user["email"],
        username=created_user["username"],
        is_active=created_user["is_active"],
        is_admin=created_user["is_admin"],
        created_at=created_user["created_at"]
    )


@router.post("/login", response_model=Token)
async def login(credentials: UserLogin):
    """Login and receive an access token."""
    db = get_database()
    
    # Find user by email
    user = await db.users.find_one({"email": credentials.email})
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Verify password
    if not verify_password(credentials.password, user["hashed_password"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Check if user is active
    if not user.get("is_active", False):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User account is disabled"
        )
    
    # Create access token
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user["email"], "user_id": str(user["_id"])},
        expires_delta=access_token_expires
    )
    
    return Token(access_token=access_token, token_type="bearer")


@router.get("/me", response_model=UserResponse)
async def get_me(current_user: dict = Depends(get_current_user)):
    """Get the current authenticated user's profile."""
    return UserResponse(
        id=str(current_user["_id"]),
        email=current_user["email"],
        username=current_user["username"],
        is_active=current_user["is_active"],
        is_admin=current_user["is_admin"],
        created_at=current_user["created_at"]
    )


@router.post("/verify-token")
async def verify_token(current_user: dict = Depends(get_current_user)):
    """Verify if the current token is valid."""
    return {
        "valid": True,
        "user_id": str(current_user["_id"]),
        "email": current_user["email"],
        "is_admin": current_user.get("is_admin", False)
    }


@router.post("/logout")
async def logout():
    """
    Logout endpoint. 
    Note: JWT tokens are stateless, so we just return success.
    Client should remove the token from storage.
    """
    return {"message": "Successfully logged out"}
