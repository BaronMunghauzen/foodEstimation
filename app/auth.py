"""
Модуль аутентификации по токену.
"""

from fastapi import Security, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import os
from dotenv import load_dotenv

load_dotenv()

security = HTTPBearer()
API_TOKEN = os.getenv("API_TOKEN", "your-secret-token-change-this-in-production")

def verify_token(credentials: HTTPAuthorizationCredentials = Security(security)):
    """
    Проверяет токен аутентификации.
    
    Args:
        credentials: HTTP Bearer токен из заголовка Authorization
        
    Returns:
        str: токен (если валиден)
        
    Raises:
        HTTPException: если токен невалиден
    """
    if credentials.credentials != API_TOKEN:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return credentials.credentials

