from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

from fastapi import Depends, HTTPException, Request, status
from jose import JWTError, jwt
from passlib.context import CryptContext

from .db import SessionLocal, User
from .settings import settings


_pwd = CryptContext(schemes=["bcrypt"], deprecated="auto")


def hash_password(password: str) -> str:
    return _pwd.hash(password)


def verify_password(password: str, password_hash: str) -> bool:
    try:
        return _pwd.verify(password, password_hash)
    except Exception:
        return False


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def create_access_token(*, user: User) -> str:
    exp = _utcnow() + timedelta(minutes=int(settings.jwt_exp_minutes))
    payload: dict[str, Any] = {"sub": user.id, "email": user.email, "exp": int(exp.timestamp())}
    return jwt.encode(payload, settings.jwt_secret, algorithm="HS256")


def _get_bearer_token(req: Request) -> str:
    h = req.headers.get("authorization") or req.headers.get("Authorization") or ""
    if not h:
        return ""
    parts = h.split()
    if len(parts) == 2 and parts[0].lower() == "bearer":
        return parts[1]
    return ""


def get_current_user(req: Request) -> User:
    if not settings.auth_enabled:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="AUTH_ENABLED=false")

    token = _get_bearer_token(req)
    if not token:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="missing token")

    try:
        payload = jwt.decode(token, settings.jwt_secret, algorithms=["HS256"])
    except JWTError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="invalid token")

    user_id = str(payload.get("sub") or "").strip()
    if not user_id:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="invalid token")

    db = SessionLocal()
    try:
        user = db.get(User, user_id)
        if not user:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="user not found")
        return user
    finally:
        db.close()


def require_admin(user: User = Depends(get_current_user)) -> User:
    if not user.is_admin:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="admin only")
    return user

