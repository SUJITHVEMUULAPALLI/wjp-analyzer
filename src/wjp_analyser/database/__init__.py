"""
Database Initialization and Management
=====================================

Database connection, session management, and initialization utilities.
"""

import os
import logging
from typing import Optional
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import SQLAlchemyError

from .models import Base, DatabaseManager, get_database_manager, init_database as _init_db_models
from ..config.unified_config_manager import get_config

logger = logging.getLogger(__name__)


def create_database_engine(database_url: str):
    """Create database engine with proper configuration."""
    try:
        engine_kwargs = {
            "echo": False,  # Set to True for SQL debugging
            "pool_size": 10,
            "max_overflow": 20,
            "pool_pre_ping": True,  # Verify connections before use
            "pool_recycle": 3600,   # Recycle connections every hour
        }
        # Only sqlite uses check_same_thread
        if database_url.startswith("sqlite"):
            engine_kwargs["connect_args"] = {"check_same_thread": False}

        engine = create_engine(database_url, **engine_kwargs)
        
        # Test connection
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        
        logger.info(f"Database engine created successfully: {database_url}")
        return engine
        
    except SQLAlchemyError as e:
        logger.error(f"Failed to create database engine: {e}")
        raise


def init_database_from_config() -> DatabaseManager:
    """Initialize database from configuration."""
    config = get_config()
    # Build database URL directly from config
    if config.database.type == "sqlite":
        db_path = f"data/{config.database.name}.db"
        database_url = f"sqlite:///{db_path}"
    elif config.database.type == "postgresql":
        password = f":{config.database.password}" if config.database.password else ""
        database_url = f"postgresql://{config.database.username}{password}@{config.database.host}:{config.database.port}/{config.database.name}"
    elif config.database.type == "mysql":
        password = f":{config.database.password}" if config.database.password else ""
        database_url = f"mysql://{config.database.username}{password}@{config.database.host}:{config.database.port}/{config.database.name}"
    else:
        raise ValueError(f"Unsupported database type: {config.database.type}")
    
    logger.info(f"Initializing database: {config.database.type}")
    
    # Create database directory for SQLite
    if config.database.type == "sqlite":
        db_dir = os.path.dirname(database_url.replace("sqlite:///", ""))
        if db_dir and not os.path.exists(db_dir):
            os.makedirs(db_dir, exist_ok=True)
            logger.info(f"Created database directory: {db_dir}")
    
    # Initialize database manager
    db_manager = DatabaseManager(database_url)
    
    # Create default admin user if no users exist (best-effort, non-fatal)
    try:
        session = db_manager.get_session()
        try:
            from .models import User  # use direct model, not attribute on manager
            from ..auth.enhanced_auth import PasswordManager, UserRole
            user_count = session.query(User).count()
            if user_count == 0:
                logger.info("No users found, creating default admin user")
                admin_email = "admin@wjp-analyser.com"
                admin_password = "Admin123!@#"
                password_hash = PasswordManager().hash_password(admin_password)
                admin_user = db_manager.create_user(
                    session=session,
                    email=admin_email,
                    password_hash=password_hash,
                    first_name="System",
                    last_name="Administrator",
                    role=UserRole.SUPER_ADMIN.value
                )
                logger.info(f"Default admin user created: {admin_email}")
                db_manager.log_audit_event(
                    session=session,
                    event_type="user_created",
                    user_id=str(admin_user.id),
                    resource_type="user",
                    resource_id=str(admin_user.id),
                    action="create",
                    success=True,
                    details={"email": admin_email, "role": UserRole.SUPER_ADMIN.value}
                )
        except Exception as e:
            logger.warning(f"Skipping admin bootstrap: {e}")
        finally:
            session.close()
    except Exception as e:
        logger.warning(f"Database session unavailable during bootstrap: {e}")
    
    return db_manager


def init_database(database_url: str) -> DatabaseManager:
    """Compatibility export: initialize database with a URL.

    Tests import init_database from wjp_analyser.database; delegate to models.init_database.
    """
    return _init_db_models(database_url)


def get_db_session():
    """Compatibility export: return a new SQLAlchemy session from the global manager."""
    mgr = get_database_manager()
    return mgr.get_session()


__all__ = [
    "create_database_engine",
    "init_database_from_config",
    "init_database",
    "get_db_session",
    "get_database_manager",
    "DatabaseManager",
    "Base",
]


def get_database_url() -> str:
    """Get database URL from configuration."""
    config = get_config()
    
    if config.database.type == "sqlite":
        db_path = f"data/{config.database.name}.db"
        return f"sqlite:///{db_path}"
    
    elif config.database.type == "postgresql":
        password = f":{config.database.password}" if config.database.password else ""
        return f"postgresql://{config.database.username}{password}@{config.database.host}:{config.database.port}/{config.database.name}"
    
    elif config.database.type == "mysql":
        password = f":{config.database.password}" if config.database.password else ""
        return f"mysql://{config.database.username}{password}@{config.database.host}:{config.database.port}/{config.database.name}"
    
    else:
        raise ValueError(f"Unsupported database type: {config.database.type}")


def check_database_connection() -> bool:
    """Check if database connection is working."""
    try:
        db_manager = get_database_manager()
        session = db_manager.get_session()
        
        # Simple query to test connection
        result = session.execute(text("SELECT 1")).fetchone()
        session.close()
        
        return result is not None
        
    except Exception as e:
        logger.error(f"Database connection check failed: {e}")
        return False


def create_database_backup(backup_path: str) -> bool:
    """Create database backup."""
    try:
        config = get_config()
        
        if config.database.type == "sqlite":
            import shutil
            db_path = get_database_url().replace("sqlite:///", "")
            shutil.copy2(db_path, backup_path)
            logger.info(f"SQLite database backed up to: {backup_path}")
            return True
        
        elif config.database.type == "postgresql":
            import subprocess
            cmd = [
                "pg_dump",
                "-h", config.database.host,
                "-p", str(config.database.port),
                "-U", config.database.username,
                "-d", config.database.name,
                "-f", backup_path
            ]
            subprocess.run(cmd, check=True)
            logger.info(f"PostgreSQL database backed up to: {backup_path}")
            return True
        
        else:
            logger.warning(f"Backup not supported for database type: {config.database.type}")
            return False
            
    except Exception as e:
        logger.error(f"Database backup failed: {e}")
        return False


def restore_database_backup(backup_path: str) -> bool:
    """Restore database from backup."""
    try:
        config = get_config()
        
        if config.database.type == "sqlite":
            import shutil
            db_path = get_database_url().replace("sqlite:///", "")
            shutil.copy2(backup_path, db_path)
            logger.info(f"SQLite database restored from: {backup_path}")
            return True
        
        elif config.database.type == "postgresql":
            import subprocess
            cmd = [
                "psql",
                "-h", config.database.host,
                "-p", str(config.database.port),
                "-U", config.database.username,
                "-d", config.database.name,
                "-f", backup_path
            ]
            subprocess.run(cmd, check=True)
            logger.info(f"PostgreSQL database restored from: {backup_path}")
            return True
        
        else:
            logger.warning(f"Restore not supported for database type: {config.database.type}")
            return False
            
    except Exception as e:
        logger.error(f"Database restore failed: {e}")
        return False


def run_database_migrations() -> bool:
    """Run database migrations using Alembic."""
    try:
        import subprocess
        
        # Run alembic upgrade
        result = subprocess.run(
            ["alembic", "upgrade", "head"],
            capture_output=True,
            text=True,
            check=True
        )
        
        logger.info("Database migrations completed successfully")
        logger.debug(f"Migration output: {result.stdout}")
        return True
        
    except subprocess.CalledProcessError as e:
        logger.error(f"Database migration failed: {e}")
        logger.error(f"Migration error: {e.stderr}")
        return False
    except FileNotFoundError:
        logger.warning("Alembic not found, skipping migrations")
        return True


def get_database_stats() -> dict:
    """Get database statistics."""
    try:
        db_manager = get_database_manager()
        session = db_manager.get_session()
        
        stats = {}
        
        # Count records in each table
        tables = ["User", "Project", "Analysis", "Conversion", "Nesting", "AuditLog"]
        for table_name in tables:
            table_class = getattr(db_manager, table_name, None)
            if table_class:
                count = session.query(table_class).count()
                stats[f"{table_name.lower()}_count"] = count
        
        session.close()
        
        return stats
        
    except Exception as e:
        logger.error(f"Failed to get database stats: {e}")
        return {}


