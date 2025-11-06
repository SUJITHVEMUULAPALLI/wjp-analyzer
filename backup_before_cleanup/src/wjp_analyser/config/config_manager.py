"""
Unified Configuration Manager
============================

Centralized configuration management for WJP ANALYSER.
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass, field
from pydantic import BaseModel, Field, validator
import logging

logger = logging.getLogger(__name__)


class DatabaseConfig(BaseModel):
    """Database configuration."""
    type: str = Field(default="sqlite", description="Database type: sqlite, postgresql")
    host: str = Field(default="localhost", description="Database host")
    port: int = Field(default=5432, description="Database port")
    name: str = Field(default="wjp_analyser", description="Database name")
    username: str = Field(default="", description="Database username")
    password: str = Field(default="", description="Database password")
    pool_size: int = Field(default=10, description="Connection pool size")
    max_overflow: int = Field(default=20, description="Max overflow connections")
    echo: bool = Field(default=False, description="Echo SQL queries")


class RedisConfig(BaseModel):
    """Redis configuration."""
    host: str = Field(default="localhost", description="Redis host")
    port: int = Field(default=6379, description="Redis port")
    password: str = Field(default="", description="Redis password")
    db: int = Field(default=0, description="Redis database number")
    max_connections: int = Field(default=10, description="Max Redis connections")


class SecurityConfig(BaseModel):
    """Security configuration."""
    jwt_secret_key: str = Field(default="", description="JWT secret key")
    master_password: str = Field(default="", description="Master password for encryption")
    session_timeout: int = Field(default=3600, description="Session timeout in seconds")
    password_min_length: int = Field(default=8, description="Minimum password length")
    password_require_special: bool = Field(default=True, description="Require special characters")
    rate_limit_enabled: bool = Field(default=True, description="Enable rate limiting")
    rate_limit_per_minute: int = Field(default=60, description="Requests per minute")
    rate_limit_per_hour: int = Field(default=1000, description="Requests per hour")
    rate_limit_burst: int = Field(default=10, description="Burst limit")
    max_upload_size: int = Field(default=33554432, description="Max upload size in bytes (32MB)")
    allowed_extensions: list = Field(default=["dxf", "jpg", "jpeg", "png", "bmp", "tiff"], 
                                   description="Allowed file extensions")


class AIConfig(BaseModel):
    """AI configuration."""
    openai_api_key: str = Field(default="", description="OpenAI API key")
    openai_model: str = Field(default="gpt-4", description="OpenAI model")
    openai_max_tokens: int = Field(default=2000, description="Max tokens")
    openai_temperature: float = Field(default=0.7, description="Temperature")
    ollama_base_url: str = Field(default="http://localhost:11434", description="Ollama base URL")
    ollama_model: str = Field(default="llama2", description="Ollama model")
    ollama_timeout: int = Field(default=30, description="Ollama timeout")


class ServerConfig(BaseModel):
    """Server configuration."""
    host: str = Field(default="127.0.0.1", description="Server host")
    port: int = Field(default=5000, description="Server port")
    debug: bool = Field(default=False, description="Debug mode")
    threaded: bool = Field(default=True, description="Threaded mode")
    processes: int = Field(default=1, description="Number of processes")


class LoggingConfig(BaseModel):
    """Logging configuration."""
    level: str = Field(default="INFO", description="Log level")
    format: str = Field(default="%(asctime)s - %(name)s - %(levelname)s - %(message)s", 
                       description="Log format")
    file_rotation_enabled: bool = Field(default=True, description="Enable file rotation")
    max_bytes: int = Field(default=10485760, description="Max log file size (10MB)")
    backup_count: int = Field(default=5, description="Number of backup files")
    console_output: bool = Field(default=True, description="Console output")
    file_output: bool = Field(default=True, description="File output")


class PerformanceConfig(BaseModel):
    """Performance configuration."""
    cache_enabled: bool = Field(default=True, description="Enable caching")
    cache_size: int = Field(default=100, description="Cache size")
    cache_ttl: int = Field(default=3600, description="Cache TTL in seconds")
    max_workers: int = Field(default=4, description="Max workers")
    timeout: int = Field(default=300, description="Request timeout")


class FeaturesConfig(BaseModel):
    """Feature flags configuration."""
    ai_analysis: bool = Field(default=True, description="Enable AI analysis")
    nesting: bool = Field(default=True, description="Enable nesting")
    cost_estimation: bool = Field(default=True, description="Enable cost estimation")
    guided_mode: bool = Field(default=True, description="Enable guided mode")
    batch_processing: bool = Field(default=True, description="Enable batch processing")
    authentication: bool = Field(default=True, description="Enable authentication")
    rate_limiting: bool = Field(default=True, description="Enable rate limiting")


class DefaultsConfig(BaseModel):
    """Default parameters configuration."""
    material: str = Field(default="steel", description="Default material")
    thickness: float = Field(default=6.0, description="Default thickness")
    kerf: float = Field(default=1.1, description="Default kerf")
    cutting_speed: float = Field(default=1200.0, description="Default cutting speed")
    cost_per_meter: float = Field(default=50.0, description="Default cost per meter")
    sheet_width: float = Field(default=3000.0, description="Default sheet width")
    sheet_height: float = Field(default=1500.0, description="Default sheet height")
    spacing: float = Field(default=10.0, description="Default spacing")


class ImageProcessingConfig(BaseModel):
    """Image processing configuration."""
    edge_threshold: float = Field(default=0.33, description="Edge threshold")
    min_contour_area: int = Field(default=100, description="Minimum contour area")
    simplify_tolerance: float = Field(default=0.02, description="Simplify tolerance")
    blur_kernel_size: int = Field(default=5, description="Blur kernel size")
    canny_low: int = Field(default=50, description="Canny low threshold")
    canny_high: int = Field(default=150, description="Canny high threshold")


class UnifiedConfig(BaseModel):
    """Unified configuration model."""
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    redis: RedisConfig = Field(default_factory=RedisConfig)
    security: SecurityConfig = Field(default_factory=SecurityConfig)
    ai: AIConfig = Field(default_factory=AIConfig)
    server: ServerConfig = Field(default_factory=ServerConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    performance: PerformanceConfig = Field(default_factory=PerformanceConfig)
    features: FeaturesConfig = Field(default_factory=FeaturesConfig)
    defaults: DefaultsConfig = Field(default_factory=DefaultsConfig)
    
    class Config:
        env_prefix = "WJP_"


class ConfigManager:
    """Centralized configuration manager."""
    
    def __init__(self, config_file: Optional[str] = None):
        """
        Initialize configuration manager.
        
        Args:
            config_file: Path to configuration file
        """
        self.config_file = config_file or "config/unified_config.yaml"
        self.config: Optional[UnifiedConfig] = None
        self._load_config()
    
    def _load_config(self) -> None:
        """Load configuration from file and environment variables."""
        try:
            # Load from file if it exists
            if Path(self.config_file).exists():
                with open(self.config_file, 'r') as f:
                    config_data = yaml.safe_load(f) or {}
            else:
                config_data = {}
            
            # Override with environment variables
            config_data = self._load_from_env(config_data)
            
            # Create config object
            self.config = UnifiedConfig(**config_data)
            
            logger.info(f"Configuration loaded from {self.config_file}")
            
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            # Use default configuration
            self.config = UnifiedConfig()
            logger.info("Using default configuration")
    
    def _load_from_env(self, config_data: Dict[str, Any]) -> Dict[str, Any]:
        """Load configuration from environment variables."""
        env_mappings = {
            'WJP_DATABASE_TYPE': 'database.type',
            'WJP_DATABASE_HOST': 'database.host',
            'WJP_DATABASE_PORT': 'database.port',
            'WJP_DATABASE_NAME': 'database.name',
            'WJP_DATABASE_USERNAME': 'database.username',
            'WJP_DATABASE_PASSWORD': 'database.password',
            'WJP_REDIS_HOST': 'redis.host',
            'WJP_REDIS_PORT': 'redis.port',
            'WJP_REDIS_PASSWORD': 'redis.password',
            'WJP_JWT_SECRET_KEY': 'security.jwt_secret_key',
            'WJP_MASTER_PASSWORD': 'security.master_password',
            'WJP_OPENAI_API_KEY': 'ai.openai_api_key',
            'WJP_OPENAI_MODEL': 'ai.openai_model',
            'WJP_SERVER_HOST': 'server.host',
            'WJP_SERVER_PORT': 'server.port',
            'WJP_SERVER_DEBUG': 'server.debug',
            'WJP_LOG_LEVEL': 'logging.level',
        }
        
        for env_var, config_path in env_mappings.items():
            env_value = os.getenv(env_var)
            if env_value:
                self._set_nested_value(config_data, config_path, env_value)
        
        return config_data
    
    def _set_nested_value(self, data: Dict[str, Any], path: str, value: Any) -> None:
        """Set a nested value in a dictionary."""
        keys = path.split('.')
        current = data
        
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        
        # Convert value type based on key
        if 'port' in keys[-1]:
            current[keys[-1]] = int(value)
        elif 'debug' in keys[-1] or 'enabled' in keys[-1]:
            current[keys[-1]] = value.lower() in ('true', '1', 'yes', 'on')
        else:
            current[keys[-1]] = value
    
    def get_config(self) -> UnifiedConfig:
        """Get the current configuration."""
        if self.config is None:
            self._load_config()
        return self.config
    
    def save_config(self, config: Optional[UnifiedConfig] = None) -> None:
        """
        Save configuration to file.
        
        Args:
            config: Configuration to save. If None, saves current config.
        """
        if config is None:
            config = self.config
        
        if config is None:
            logger.error("No configuration to save")
            return
        
        try:
            # Ensure config directory exists
            os.makedirs(os.path.dirname(self.config_file), exist_ok=True)
            
            # Convert to dictionary and save
            config_dict = config.dict()
            
            with open(self.config_file, 'w') as f:
                yaml.dump(config_dict, f, default_flow_style=False, indent=2)
            
            logger.info(f"Configuration saved to {self.config_file}")
            
        except Exception as e:
            logger.error(f"Failed to save configuration: {e}")
    
    def reload_config(self) -> None:
        """Reload configuration from file."""
        self._load_config()
        logger.info("Configuration reloaded")
    
    def get_database_url(self) -> str:
        """Get database URL."""
        db_config = self.get_config().database
        
        if db_config.type == "sqlite":
            return f"sqlite:///{db_config.name}.db"
        elif db_config.type == "postgresql":
            auth = f"{db_config.username}:{db_config.password}@" if db_config.username else ""
            return f"postgresql://{auth}{db_config.host}:{db_config.port}/{db_config.name}"
        else:
            raise ValueError(f"Unsupported database type: {db_config.type}")
    
    def get_redis_url(self) -> str:
        """Get Redis URL."""
        redis_config = self.get_config().redis
        
        auth = f":{redis_config.password}@" if redis_config.password else ""
        return f"redis://{auth}{redis_config.host}:{redis_config.port}/{redis_config.db}"
    
    def validate_config(self) -> bool:
        """
        Validate configuration.
        
        Returns:
            True if configuration is valid, False otherwise
        """
        try:
            config = self.get_config()
            
            # Validate required fields
            if not config.security.jwt_secret_key:
                logger.error("JWT secret key is required")
                return False
            
            if config.features.authentication and not config.security.master_password:
                logger.error("Master password is required when authentication is enabled")
                return False
            
            if config.features.ai_analysis and not config.ai.openai_api_key:
                logger.warning("OpenAI API key not set - AI features will be disabled")
            
            logger.info("Configuration validation passed")
            return True
            
        except Exception as e:
            logger.error(f"Configuration validation failed: {e}")
            return False
    
    def create_default_config(self) -> None:
        """Create a default configuration file."""
        default_config = UnifiedConfig()
        
        # Set some secure defaults
        default_config.security.jwt_secret_key = "change-this-in-production"
        default_config.security.master_password = "change-this-in-production"
        
        self.save_config(default_config)
        logger.info("Default configuration created")


# Global configuration manager instance
config_manager = ConfigManager()


def get_config_manager() -> ConfigManager:
    """Get the global configuration manager instance."""
    return config_manager


def get_config() -> UnifiedConfig:
    """Get the global configuration."""
    return config_manager.get_config()


def reload_config() -> None:
    """Reload the global configuration."""
    config_manager.reload_config()
