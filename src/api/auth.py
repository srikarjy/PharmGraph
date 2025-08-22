"""Authentication and authorization system with JWT tokens and RBAC."""

import jwt
import bcrypt
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Set
import secrets
import logging
from enum import Enum

from fastapi import HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel

from ..config import get_logger, get_config

logger = get_logger(__name__)


class UserRole(str, Enum):
    """User roles enumeration."""
    ADMIN = "admin"
    CLINICIAN = "clinician"
    RESEARCHER = "researcher"
    API_USER = "api_user"
    READONLY = "readonly"


class Permission(str, Enum):
    """Permission enumeration."""
    # Model operations
    MODEL_PREDICT = "model:predict"
    MODEL_BATCH_PREDICT = "model:batch_predict"
    MODEL_INFO = "model:info"
    MODEL_MANAGE = "model:manage"
    
    # Clinical operations
    DRUG_GENE_ANALYSIS = "clinical:drug_gene"
    RISK_SCORING = "clinical:risk_scoring"
    DOSING_RECOMMENDATIONS = "clinical:dosing"
    CLINICAL_SUPPORT = "clinical:support"
    
    # Admin operations
    USER_MANAGE = "admin:users"
    SYSTEM_MONITOR = "admin:monitor"
    SYSTEM_CONFIG = "admin:config"
    
    # Data operations
    DATA_READ = "data:read"
    DATA_WRITE = "data:write"
    DATA_DELETE = "data:delete"


class User(BaseModel):
    """User model."""
    user_id: str
    username: str
    email: Optional[str] = None
    roles: List[UserRole]
    permissions: Set[Permission]
    is_active: bool = True
    created_at: datetime
    last_login: Optional[datetime] = None
    api_key: Optional[str] = None


class TokenData(BaseModel):
    """JWT token data."""
    user_id: str
    username: str
    roles: List[str]
    permissions: List[str]
    exp: datetime
    iat: datetime


class AuthManager:
    """Authentication and authorization manager."""
    
    def __init__(self):
        """Initialize auth manager."""
        self.config = get_config()
        
        # JWT configuration
        self.secret_key = self.config.get("auth.jwt_secret", self._generate_secret_key())
        self.algorithm = "HS256"
        self.access_token_expire_minutes = self.config.get("auth.access_token_expire_minutes", 30)
        self.refresh_token_expire_days = self.config.get("auth.refresh_token_expire_days", 7)
        
        # Security settings
        self.password_min_length = self.config.get("auth.password_min_length", 8)
        self.max_login_attempts = self.config.get("auth.max_login_attempts", 5)
        self.lockout_duration_minutes = self.config.get("auth.lockout_duration_minutes", 15)
        
        # Role-permission mapping
        self.role_permissions = self._setup_role_permissions()
        
        # In-memory user store (in production, use database)
        self.users = self._create_default_users()
        self.api_keys = {}
        self.login_attempts = {}
        
        # HTTP Bearer security
        self.security = HTTPBearer()
        
        logger.info("Auth manager initialized")
    
    def _generate_secret_key(self) -> str:
        """Generate a secure secret key."""
        return secrets.token_urlsafe(32)
    
    def _setup_role_permissions(self) -> Dict[UserRole, Set[Permission]]:
        """Setup role-permission mapping."""
        return {
            UserRole.ADMIN: {
                # All permissions
                Permission.MODEL_PREDICT,
                Permission.MODEL_BATCH_PREDICT,
                Permission.MODEL_INFO,
                Permission.MODEL_MANAGE,
                Permission.DRUG_GENE_ANALYSIS,
                Permission.RISK_SCORING,
                Permission.DOSING_RECOMMENDATIONS,
                Permission.CLINICAL_SUPPORT,
                Permission.USER_MANAGE,
                Permission.SYSTEM_MONITOR,
                Permission.SYSTEM_CONFIG,
                Permission.DATA_READ,
                Permission.DATA_WRITE,
                Permission.DATA_DELETE,
            },
            UserRole.CLINICIAN: {
                Permission.MODEL_PREDICT,
                Permission.MODEL_BATCH_PREDICT,
                Permission.MODEL_INFO,
                Permission.DRUG_GENE_ANALYSIS,
                Permission.RISK_SCORING,
                Permission.DOSING_RECOMMENDATIONS,
                Permission.CLINICAL_SUPPORT,
                Permission.DATA_READ,
            },
            UserRole.RESEARCHER: {
                Permission.MODEL_PREDICT,
                Permission.MODEL_BATCH_PREDICT,
                Permission.MODEL_INFO,
                Permission.DRUG_GENE_ANALYSIS,
                Permission.RISK_SCORING,
                Permission.DATA_READ,
                Permission.DATA_WRITE,
            },
            UserRole.API_USER: {
                Permission.MODEL_PREDICT,
                Permission.MODEL_BATCH_PREDICT,
                Permission.MODEL_INFO,
                Permission.DATA_READ,
            },
            UserRole.READONLY: {
                Permission.MODEL_INFO,
                Permission.DATA_READ,
            }
        }
    
    def _create_default_users(self) -> Dict[str, User]:
        """Create default users for development."""
        users = {}
        
        # Admin user
        admin_user = User(
            user_id="admin_001",
            username="admin",
            email="admin@pharmacogenomics.com",
            roles=[UserRole.ADMIN],
            permissions=self.role_permissions[UserRole.ADMIN],
            created_at=datetime.utcnow()
        )
        users["admin"] = admin_user
        
        # Clinician user
        clinician_user = User(
            user_id="clinician_001",
            username="clinician",
            email="clinician@hospital.com",
            roles=[UserRole.CLINICIAN],
            permissions=self.role_permissions[UserRole.CLINICIAN],
            created_at=datetime.utcnow()
        )
        users["clinician"] = clinician_user
        
        # API user
        api_user = User(
            user_id="api_001",
            username="api_user",
            roles=[UserRole.API_USER],
            permissions=self.role_permissions[UserRole.API_USER],
            created_at=datetime.utcnow(),
            api_key=self._generate_api_key()
        )
        users["api_user"] = api_user
        self.api_keys[api_user.api_key] = api_user
        
        return users
    
    def _generate_api_key(self) -> str:
        """Generate API key."""
        return f"pgx_{secrets.token_urlsafe(32)}"
    
    def hash_password(self, password: str) -> str:
        """Hash password using bcrypt."""
        salt = bcrypt.gensalt()
        return bcrypt.hashpw(password.encode('utf-8'), salt).decode('utf-8')
    
    def verify_password(self, password: str, hashed_password: str) -> bool:
        """Verify password against hash."""
        return bcrypt.checkpw(password.encode('utf-8'), hashed_password.encode('utf-8'))
    
    def create_access_token(self, user: User) -> str:
        """Create JWT access token."""
        now = datetime.utcnow()
        expire = now + timedelta(minutes=self.access_token_expire_minutes)
        
        payload = {
            "user_id": user.user_id,
            "username": user.username,
            "roles": [role.value for role in user.roles],
            "permissions": [perm.value for perm in user.permissions],
            "exp": expire,
            "iat": now,
            "type": "access"
        }
        
        return jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
    
    def create_refresh_token(self, user: User) -> str:
        """Create JWT refresh token."""
        now = datetime.utcnow()
        expire = now + timedelta(days=self.refresh_token_expire_days)
        
        payload = {
            "user_id": user.user_id,
            "username": user.username,
            "exp": expire,
            "iat": now,
            "type": "refresh"
        }
        
        return jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
    
    def verify_token(self, token: str) -> TokenData:
        """Verify and decode JWT token."""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            
            # Check token type
            if payload.get("type") != "access":
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid token type"
                )
            
            # Create token data
            token_data = TokenData(
                user_id=payload["user_id"],
                username=payload["username"],
                roles=payload["roles"],
                permissions=payload["permissions"],
                exp=datetime.fromtimestamp(payload["exp"]),
                iat=datetime.fromtimestamp(payload["iat"])
            )
            
            return token_data
            
        except jwt.ExpiredSignatureError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token has expired"
            )
        except jwt.JWTError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token"
            )
    
    def authenticate_user(self, username: str, password: str) -> Optional[User]:
        """Authenticate user with username and password."""
        # Check login attempts
        if self._is_account_locked(username):
            raise HTTPException(
                status_code=status.HTTP_423_LOCKED,
                detail="Account temporarily locked due to too many failed attempts"
            )
        
        # Get user
        user = self.users.get(username)
        if not user or not user.is_active:
            self._record_failed_attempt(username)
            return None
        
        # For demo purposes, accept any password for existing users
        # In production, verify against stored hash
        if username in self.users:
            # Reset login attempts on successful login
            self._reset_login_attempts(username)
            user.last_login = datetime.utcnow()
            return user
        
        self._record_failed_attempt(username)
        return None
    
    def authenticate_api_key(self, api_key: str) -> Optional[User]:
        """Authenticate user with API key."""
        return self.api_keys.get(api_key)
    
    def _is_account_locked(self, username: str) -> bool:
        """Check if account is locked due to failed attempts."""
        if username not in self.login_attempts:
            return False
        
        attempts = self.login_attempts[username]
        if attempts["count"] >= self.max_login_attempts:
            # Check if lockout period has expired
            lockout_end = attempts["last_attempt"] + timedelta(minutes=self.lockout_duration_minutes)
            if datetime.utcnow() < lockout_end:
                return True
            else:
                # Reset attempts after lockout period
                self._reset_login_attempts(username)
        
        return False
    
    def _record_failed_attempt(self, username: str):
        """Record failed login attempt."""
        now = datetime.utcnow()
        
        if username not in self.login_attempts:
            self.login_attempts[username] = {"count": 0, "last_attempt": now}
        
        self.login_attempts[username]["count"] += 1
        self.login_attempts[username]["last_attempt"] = now
    
    def _reset_login_attempts(self, username: str):
        """Reset login attempts for user."""
        if username in self.login_attempts:
            del self.login_attempts[username]
    
    def has_permission(self, user: User, permission: Permission) -> bool:
        """Check if user has specific permission."""
        return permission in user.permissions
    
    def has_role(self, user: User, role: UserRole) -> bool:
        """Check if user has specific role."""
        return role in user.roles
    
    def create_user(self, username: str, email: str, roles: List[UserRole], 
                   password: Optional[str] = None) -> User:
        """Create new user."""
        if username in self.users:
            raise ValueError(f"User {username} already exists")
        
        # Combine permissions from all roles
        permissions = set()
        for role in roles:
            permissions.update(self.role_permissions.get(role, set()))
        
        user = User(
            user_id=f"user_{len(self.users) + 1:03d}",
            username=username,
            email=email,
            roles=roles,
            permissions=permissions,
            created_at=datetime.utcnow()
        )
        
        # Generate API key for API users
        if UserRole.API_USER in roles:
            user.api_key = self._generate_api_key()
            self.api_keys[user.api_key] = user
        
        self.users[username] = user
        logger.info(f"Created user: {username} with roles: {[r.value for r in roles]}")
        
        return user
    
    def get_user(self, username: str) -> Optional[User]:
        """Get user by username."""
        return self.users.get(username)
    
    def get_user_by_id(self, user_id: str) -> Optional[User]:
        """Get user by ID."""
        for user in self.users.values():
            if user.user_id == user_id:
                return user
        return None
    
    def update_user_roles(self, username: str, roles: List[UserRole]) -> bool:
        """Update user roles."""
        user = self.users.get(username)
        if not user:
            return False
        
        # Update roles
        user.roles = roles
        
        # Update permissions
        permissions = set()
        for role in roles:
            permissions.update(self.role_permissions.get(role, set()))
        user.permissions = permissions
        
        logger.info(f"Updated roles for user {username}: {[r.value for r in roles]}")
        return True
    
    def deactivate_user(self, username: str) -> bool:
        """Deactivate user."""
        user = self.users.get(username)
        if not user:
            return False
        
        user.is_active = False
        
        # Remove API key if exists
        if user.api_key and user.api_key in self.api_keys:
            del self.api_keys[user.api_key]
        
        logger.info(f"Deactivated user: {username}")
        return True
    
    def generate_new_api_key(self, username: str) -> Optional[str]:
        """Generate new API key for user."""
        user = self.users.get(username)
        if not user:
            return None
        
        # Remove old API key
        if user.api_key and user.api_key in self.api_keys:
            del self.api_keys[user.api_key]
        
        # Generate new API key
        new_api_key = self._generate_api_key()
        user.api_key = new_api_key
        self.api_keys[new_api_key] = user
        
        logger.info(f"Generated new API key for user: {username}")
        return new_api_key


# Global auth manager instance
auth_manager = AuthManager()


# Dependency functions for FastAPI
async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(auth_manager.security)) -> User:
    """Get current authenticated user from JWT token."""
    token = credentials.credentials
    
    # Try JWT token first
    try:
        token_data = auth_manager.verify_token(token)
        user = auth_manager.get_user(token_data.username)
        
        if not user or not user.is_active:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User not found or inactive"
            )
        
        return user
        
    except HTTPException:
        # Try API key authentication
        user = auth_manager.authenticate_api_key(token)
        
        if not user or not user.is_active:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication credentials",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        return user


def require_permission(permission: Permission):
    """Dependency factory for requiring specific permission."""
    def permission_checker(current_user: User = Depends(get_current_user)) -> User:
        if not auth_manager.has_permission(current_user, permission):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Permission required: {permission.value}"
            )
        return current_user
    
    return permission_checker


def require_role(role: UserRole):
    """Dependency factory for requiring specific role."""
    def role_checker(current_user: User = Depends(get_current_user)) -> User:
        if not auth_manager.has_role(current_user, role):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Role required: {role.value}"
            )
        return current_user
    
    return role_checker


# Convenience dependencies
require_admin = require_role(UserRole.ADMIN)
require_clinician = require_role(UserRole.CLINICIAN)
require_model_predict = require_permission(Permission.MODEL_PREDICT)
require_clinical_support = require_permission(Permission.CLINICAL_SUPPORT)