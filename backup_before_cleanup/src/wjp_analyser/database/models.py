"""
Database Models and Integration
===============================

SQLAlchemy models for user management, project tracking, and data persistence.
"""

from sqlalchemy import create_engine, Column, Integer, String, DateTime, Boolean, Text, Float, ForeignKey, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship, Session
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.sql import func
from datetime import datetime
from typing import Optional, List, Dict, Any
import uuid
import logging

logger = logging.getLogger(__name__)

Base = declarative_base()


class User(Base):
    """User model for authentication and user management."""
    __tablename__ = 'users'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    email = Column(String(255), unique=True, nullable=False, index=True)
    password_hash = Column(String(255), nullable=False)
    first_name = Column(String(100), nullable=True)
    last_name = Column(String(100), nullable=True)
    role = Column(String(50), nullable=False, default='user')
    is_active = Column(Boolean, default=True, nullable=False)
    is_verified = Column(Boolean, default=False, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    last_login = Column(DateTime(timezone=True), nullable=True)
    failed_login_attempts = Column(Integer, default=0)
    locked_until = Column(DateTime(timezone=True), nullable=True)
    
    # Relationships
    projects = relationship("Project", back_populates="user", cascade="all, delete-orphan")
    sessions = relationship("UserSession", back_populates="user", cascade="all, delete-orphan")
    audit_logs = relationship("AuditLog", back_populates="user")
    
    def __repr__(self):
        return f"<User(id={self.id}, email={self.email}, role={self.role})>"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert user to dictionary."""
        return {
            "id": str(self.id),
            "email": self.email,
            "first_name": self.first_name,
            "last_name": self.last_name,
            "role": self.role,
            "is_active": self.is_active,
            "is_verified": self.is_verified,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "last_login": self.last_login.isoformat() if self.last_login else None
        }


class UserSession(Base):
    """User session model for session management."""
    __tablename__ = 'user_sessions'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey('users.id'), nullable=False)
    session_token = Column(String(255), unique=True, nullable=False, index=True)
    refresh_token = Column(String(255), unique=True, nullable=True)
    ip_address = Column(String(45), nullable=True)  # IPv6 support
    user_agent = Column(Text, nullable=True)
    is_active = Column(Boolean, default=True, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    expires_at = Column(DateTime(timezone=True), nullable=False)
    last_activity = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    user = relationship("User", back_populates="sessions")
    
    def __repr__(self):
        return f"<UserSession(id={self.id}, user_id={self.user_id}, active={self.is_active})>"


class Project(Base):
    """Project model for organizing analyses and conversions."""
    __tablename__ = 'projects'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey('users.id'), nullable=False)
    name = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    status = Column(String(50), default='active', nullable=False)  # active, archived, deleted
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationships
    user = relationship("User", back_populates="projects")
    analyses = relationship("Analysis", back_populates="project", cascade="all, delete-orphan")
    conversions = relationship("Conversion", back_populates="project", cascade="all, delete-orphan")
    nestings = relationship("Nesting", back_populates="project", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<Project(id={self.id}, name={self.name}, user_id={self.user_id})>"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert project to dictionary."""
        return {
            "id": str(self.id),
            "name": self.name,
            "description": self.description,
            "status": self.status,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "analyses_count": len(self.analyses),
            "conversions_count": len(self.conversions),
            "nestings_count": len(self.nestings)
        }


class Analysis(Base):
    """DXF analysis model for storing analysis results."""
    __tablename__ = 'analyses'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    project_id = Column(UUID(as_uuid=True), ForeignKey('projects.id'), nullable=False)
    name = Column(String(255), nullable=False)
    dxf_file_path = Column(String(500), nullable=False)
    dxf_file_size = Column(Integer, nullable=True)
    analysis_type = Column(String(50), default='geometric', nullable=False)  # geometric, cost, quality
    status = Column(String(50), default='pending', nullable=False)  # pending, processing, completed, failed
    parameters = Column(JSON, nullable=True)  # Analysis parameters
    results = Column(JSON, nullable=True)  # Analysis results
    error_message = Column(Text, nullable=True)
    processing_time = Column(Float, nullable=True)  # Processing time in seconds
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    completed_at = Column(DateTime(timezone=True), nullable=True)
    
    # Relationships
    project = relationship("Project", back_populates="analyses")
    
    def __repr__(self):
        return f"<Analysis(id={self.id}, name={self.name}, status={self.status})>"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert analysis to dictionary."""
        return {
            "id": str(self.id),
            "name": self.name,
            "dxf_file_path": self.dxf_file_path,
            "dxf_file_size": self.dxf_file_size,
            "analysis_type": self.analysis_type,
            "status": self.status,
            "parameters": self.parameters,
            "results": self.results,
            "error_message": self.error_message,
            "processing_time": self.processing_time,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None
        }


class Conversion(Base):
    """Image to DXF conversion model."""
    __tablename__ = 'conversions'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    project_id = Column(UUID(as_uuid=True), ForeignKey('projects.id'), nullable=False)
    name = Column(String(255), nullable=False)
    image_file_path = Column(String(500), nullable=False)
    image_file_size = Column(Integer, nullable=True)
    dxf_file_path = Column(String(500), nullable=True)
    conversion_algorithm = Column(String(50), default='unified', nullable=False)  # potrace, opencv, unified
    status = Column(String(50), default='pending', nullable=False)  # pending, processing, completed, failed
    parameters = Column(JSON, nullable=True)  # Conversion parameters
    results = Column(JSON, nullable=True)  # Conversion results
    error_message = Column(Text, nullable=True)
    processing_time = Column(Float, nullable=True)  # Processing time in seconds
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    completed_at = Column(DateTime(timezone=True), nullable=True)
    
    # Relationships
    project = relationship("Project", back_populates="conversions")
    
    def __repr__(self):
        return f"<Conversion(id={self.id}, name={self.name}, status={self.status})>"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert conversion to dictionary."""
        return {
            "id": str(self.id),
            "name": self.name,
            "image_file_path": self.image_file_path,
            "image_file_size": self.image_file_size,
            "dxf_file_path": self.dxf_file_path,
            "conversion_algorithm": self.conversion_algorithm,
            "status": self.status,
            "parameters": self.parameters,
            "results": self.results,
            "error_message": self.error_message,
            "processing_time": self.processing_time,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None
        }


class Nesting(Base):
    """Nesting optimization model."""
    __tablename__ = 'nestings'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    project_id = Column(UUID(as_uuid=True), ForeignKey('projects.id'), nullable=False)
    name = Column(String(255), nullable=False)
    input_dxf_path = Column(String(500), nullable=False)
    sheet_width = Column(Float, nullable=False)
    sheet_height = Column(Float, nullable=False)
    status = Column(String(50), default='pending', nullable=False)  # pending, processing, completed, failed
    parameters = Column(JSON, nullable=True)  # Nesting parameters
    results = Column(JSON, nullable=True)  # Nesting results
    error_message = Column(Text, nullable=True)
    processing_time = Column(Float, nullable=True)  # Processing time in seconds
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    completed_at = Column(DateTime(timezone=True), nullable=True)
    
    # Relationships
    project = relationship("Project", back_populates="nestings")
    
    def __repr__(self):
        return f"<Nesting(id={self.id}, name={self.name}, status={self.status})>"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert nesting to dictionary."""
        return {
            "id": str(self.id),
            "name": self.name,
            "input_dxf_path": self.input_dxf_path,
            "sheet_width": self.sheet_width,
            "sheet_height": self.sheet_height,
            "status": self.status,
            "parameters": self.parameters,
            "results": self.results,
            "error_message": self.error_message,
            "processing_time": self.processing_time,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None
        }


class AuditLog(Base):
    """Audit log model for security and compliance."""
    __tablename__ = 'audit_logs'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey('users.id'), nullable=True)
    event_type = Column(String(50), nullable=False)  # login, logout, create, update, delete, etc.
    resource_type = Column(String(50), nullable=True)  # user, project, analysis, etc.
    resource_id = Column(String(255), nullable=True)
    action = Column(String(100), nullable=True)
    ip_address = Column(String(45), nullable=True)
    user_agent = Column(Text, nullable=True)
    success = Column(Boolean, nullable=False)
    details = Column(JSON, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    user = relationship("User", back_populates="audit_logs")
    
    def __repr__(self):
        return f"<AuditLog(id={self.id}, event_type={self.event_type}, success={self.success})>"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert audit log to dictionary."""
        return {
            "id": str(self.id),
            "user_id": str(self.user_id) if self.user_id else None,
            "event_type": self.event_type,
            "resource_type": self.resource_type,
            "resource_id": self.resource_id,
            "action": self.action,
            "ip_address": self.ip_address,
            "success": self.success,
            "details": self.details,
            "created_at": self.created_at.isoformat() if self.created_at else None
        }


class APIKey(Base):
    """API key model for external integrations."""
    __tablename__ = 'api_keys'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey('users.id'), nullable=False)
    name = Column(String(255), nullable=False)
    service = Column(String(100), nullable=False)  # openai, ollama, etc.
    encrypted_key = Column(Text, nullable=False)
    is_active = Column(Boolean, default=True, nullable=False)
    last_used = Column(DateTime(timezone=True), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    expires_at = Column(DateTime(timezone=True), nullable=True)
    
    def __repr__(self):
        return f"<APIKey(id={self.id}, name={self.name}, service={self.service})>"


class DatabaseManager:
    """Database manager for connection and session management."""
    
    def __init__(self, database_url: str):
        self.database_url = database_url
        self.engine = create_engine(database_url, echo=False)
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
        
        # Create tables
        Base.metadata.create_all(bind=self.engine)
        logger.info("Database tables created successfully")
    
    def get_session(self) -> Session:
        """Get database session."""
        return self.SessionLocal()
    
    def close_session(self, session: Session):
        """Close database session."""
        session.close()
    
    def create_user(self, session: Session, email: str, password_hash: str, 
                   first_name: Optional[str] = None, last_name: Optional[str] = None,
                   role: str = 'user') -> User:
        """Create new user."""
        user = User(
            email=email,
            password_hash=password_hash,
            first_name=first_name,
            last_name=last_name,
            role=role
        )
        session.add(user)
        session.commit()
        session.refresh(user)
        logger.info(f"User created: {email}")
        return user
    
    def get_user_by_email(self, session: Session, email: str) -> Optional[User]:
        """Get user by email."""
        return session.query(User).filter(User.email == email).first()
    
    def get_user_by_id(self, session: Session, user_id: str) -> Optional[User]:
        """Get user by ID."""
        return session.query(User).filter(User.id == user_id).first()
    
    def create_project(self, session: Session, user_id: str, name: str, 
                      description: Optional[str] = None) -> Project:
        """Create new project."""
        project = Project(
            user_id=user_id,
            name=name,
            description=description
        )
        session.add(project)
        session.commit()
        session.refresh(project)
        logger.info(f"Project created: {name}")
        return project
    
    def get_user_projects(self, session: Session, user_id: str) -> List[Project]:
        """Get all projects for a user."""
        return session.query(Project).filter(
            Project.user_id == user_id,
            Project.status == 'active'
        ).order_by(Project.created_at.desc()).all()
    
    def create_analysis(self, session: Session, project_id: str, name: str,
                       dxf_file_path: str, analysis_type: str = 'geometric',
                       parameters: Optional[Dict[str, Any]] = None) -> Analysis:
        """Create new analysis."""
        analysis = Analysis(
            project_id=project_id,
            name=name,
            dxf_file_path=dxf_file_path,
            analysis_type=analysis_type,
            parameters=parameters
        )
        session.add(analysis)
        session.commit()
        session.refresh(analysis)
        logger.info(f"Analysis created: {name}")
        return analysis
    
    def update_analysis_status(self, session: Session, analysis_id: str, 
                             status: str, results: Optional[Dict[str, Any]] = None,
                             error_message: Optional[str] = None,
                             processing_time: Optional[float] = None):
        """Update analysis status and results."""
        analysis = session.query(Analysis).filter(Analysis.id == analysis_id).first()
        if analysis:
            analysis.status = status
            if results:
                analysis.results = results
            if error_message:
                analysis.error_message = error_message
            if processing_time:
                analysis.processing_time = processing_time
            if status == 'completed':
                analysis.completed_at = datetime.utcnow()
            session.commit()
            logger.info(f"Analysis {analysis_id} status updated to {status}")
    
    def create_conversion(self, session: Session, project_id: str, name: str,
                         image_file_path: str, conversion_algorithm: str = 'unified',
                         parameters: Optional[Dict[str, Any]] = None) -> Conversion:
        """Create new conversion."""
        conversion = Conversion(
            project_id=project_id,
            name=name,
            image_file_path=image_file_path,
            conversion_algorithm=conversion_algorithm,
            parameters=parameters
        )
        session.add(conversion)
        session.commit()
        session.refresh(conversion)
        logger.info(f"Conversion created: {name}")
        return conversion
    
    def create_nesting(self, session: Session, project_id: str, name: str,
                      input_dxf_path: str, sheet_width: float, sheet_height: float,
                      parameters: Optional[Dict[str, Any]] = None) -> Nesting:
        """Create new nesting."""
        nesting = Nesting(
            project_id=project_id,
            name=name,
            input_dxf_path=input_dxf_path,
            sheet_width=sheet_width,
            sheet_height=sheet_height,
            parameters=parameters
        )
        session.add(nesting)
        session.commit()
        session.refresh(nesting)
        logger.info(f"Nesting created: {name}")
        return nesting
    
    def log_audit_event(self, session: Session, event_type: str, user_id: Optional[str] = None,
                       resource_type: Optional[str] = None, resource_id: Optional[str] = None,
                       action: Optional[str] = None, ip_address: Optional[str] = None,
                       user_agent: Optional[str] = None, success: bool = True,
                       details: Optional[Dict[str, Any]] = None):
        """Log audit event."""
        audit_log = AuditLog(
            user_id=user_id,
            event_type=event_type,
            resource_type=resource_type,
            resource_id=resource_id,
            action=action,
            ip_address=ip_address,
            user_agent=user_agent,
            success=success,
            details=details
        )
        session.add(audit_log)
        session.commit()
        logger.info(f"Audit event logged: {event_type}")


# Global database manager instance
database_manager: Optional[DatabaseManager] = None


def init_database(database_url: str) -> DatabaseManager:
    """Initialize database manager."""
    global database_manager
    database_manager = DatabaseManager(database_url)
    return database_manager


def get_database_manager() -> DatabaseManager:
    """Get database manager instance."""
    if database_manager is None:
        raise RuntimeError("Database not initialized. Call init_database() first.")
    return database_manager


def get_db_session() -> Session:
    """Get database session."""
    return get_database_manager().get_session()