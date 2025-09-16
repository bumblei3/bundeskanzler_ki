"""
Database models and connection for Bundeskanzler KI
"""

import os
from datetime import datetime
from typing import Any, AsyncGenerator, Dict, Optional

from sqlalchemy import (
    JSON,
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

# Database URL from environment - default to SQLite for testing
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "sqlite+aiosqlite:///./bundeskanzler.db",
)


class Base(DeclarativeBase):
    """Base class for all database models"""

    pass


class Conversation(Base):
    """Model for storing chat conversations"""

    __tablename__ = "conversations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    user_id: Mapped[Optional[str]] = mapped_column(String(255), index=True)
    user_message: Mapped[str] = mapped_column(Text, nullable=False)
    ai_response: Mapped[str] = mapped_column(Text, nullable=False)
    confidence_score: Mapped[Optional[float]] = mapped_column(Float)
    response_time: Mapped[Optional[float]] = mapped_column(Float)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow)
    conversation_metadata: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON)

    # Relationship to feedback
    feedback = relationship("Feedback", back_populates="conversation", uselist=False)


class UserSession(Base):
    """Model for user session management"""

    __tablename__ = "user_sessions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(String(255), nullable=False, unique=True)
    user_id: Mapped[Optional[str]] = mapped_column(String(255))
    ip_address: Mapped[Optional[str]] = mapped_column(String(45))  # IPv6 support
    user_agent: Mapped[Optional[str]] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow)
    last_activity: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=datetime.utcnow
    )


class APIKey(Base):
    """Model for API key management"""

    __tablename__ = "api_keys"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    key_hash: Mapped[str] = mapped_column(String(255), nullable=False, unique=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    permissions: Mapped[Dict[str, Any]] = mapped_column(JSON, default=list)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow)
    last_used: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    usage_count: Mapped[int] = mapped_column(Integer, default=0)


class SystemLog(Base):
    """Model for system logging"""

    __tablename__ = "system_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    level: Mapped[str] = mapped_column(String(20), nullable=False, index=True)
    component: Mapped[Optional[str]] = mapped_column(String(100), index=True)
    message: Mapped[str] = mapped_column(Text, nullable=False)
    log_metadata: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow)


class Feedback(Base):
    """Model for user feedback on conversations"""

    __tablename__ = "feedback"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    conversation_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("conversations.id"), nullable=False
    )
    rating: Mapped[int] = mapped_column(Integer, nullable=False)  # 1-5 stars
    comment: Mapped[Optional[str]] = mapped_column(Text)
    user_id: Mapped[Optional[str]] = mapped_column(String(255))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow)

    # Relationship to conversation
    conversation = relationship("Conversation", back_populates="feedback")


# Create database engine with optimized connection pooling
if DATABASE_URL.startswith("sqlite"):
    # SQLite-spezifische Engine-Konfiguration
    engine = create_async_engine(
        DATABASE_URL,
        echo=False,
        connect_args={
            "check_same_thread": False,
        },
    )
else:
    # PostgreSQL/MySQL Engine-Konfiguration
    engine = create_async_engine(
        DATABASE_URL,
        echo=False,
        pool_size=20,
        max_overflow=30,
        pool_timeout=30,
        pool_recycle=1800,
        pool_pre_ping=True,
    )

# Create async session factory
async_session = async_sessionmaker(engine, expire_on_commit=False)


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """Dependency for getting database session"""
    async with async_session() as session:
        try:
            yield session
        finally:
            await session.close()


async def init_database():
    """Initialize database tables"""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def close_database():
    """Close database connections"""
    await engine.dispose()


# Database operations helper functions
async def create_conversation(
    session: AsyncSession,
    session_id: str,
    user_message: str,
    ai_response: str,
    user_id: Optional[str] = None,
    confidence_score: Optional[float] = None,
    response_time: Optional[float] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Conversation:
    """Create a new conversation record"""
    conversation = Conversation(
        session_id=session_id,
        user_id=user_id,
        user_message=user_message,
        ai_response=ai_response,
        confidence_score=confidence_score,
        response_time=response_time,
        conversation_metadata=metadata or {},
    )
    session.add(conversation)
    await session.commit()
    await session.refresh(conversation)
    return conversation


async def get_conversations_by_session(
    session: AsyncSession, session_id: str, limit: int = 50
) -> list[Conversation]:
    """Get conversations for a session"""
    result = await session.execute(
        session.query(Conversation)
        .filter(Conversation.session_id == session_id)
        .order_by(Conversation.created_at.desc())
        .limit(limit)
    )
    return result.scalars().all()


async def log_system_event(
    session: AsyncSession,
    level: str,
    message: str,
    component: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> SystemLog:
    """Log a system event"""
    log_entry = SystemLog(
        level=level, component=component, message=message, log_metadata=metadata or {}
    )
    session.add(log_entry)
    await session.commit()
    await session.refresh(log_entry)
    return log_entry


async def get_conversation_history(
    session: AsyncSession,
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
) -> tuple[list[Conversation], int]:
    """Get conversation history with optional filtering"""
    query = session.query(Conversation)

    if user_id:
        query = query.filter(Conversation.user_id == user_id)
    if session_id:
        query = query.filter(Conversation.session_id == session_id)

    # Get total count
    count_query = session.query(func.count(Conversation.id))
    if user_id:
        count_query = count_query.filter(Conversation.user_id == user_id)
    if session_id:
        count_query = count_query.filter(Conversation.session_id == session_id)

    total_count_result = await session.execute(count_query)
    total_count = total_count_result.scalar()

    # Get paginated results
    result = await session.execute(
        query.order_by(Conversation.created_at.desc()).limit(limit).offset(offset)
    )
    conversations = result.scalars().all()

    return conversations, total_count


async def validate_api_key(session: AsyncSession, key_hash: str) -> Optional[APIKey]:
    """Validate an API key"""
    result = await session.execute(
        session.query(APIKey).filter(APIKey.key_hash == key_hash, APIKey.is_active == True)
    )
    api_key = result.scalar_one_or_none()
    if api_key:
        # Update usage statistics
        api_key.usage_count += 1
        api_key.last_used = datetime.utcnow()
        await session.commit()
    return api_key