"""
Unified Configuration Management System
======================================

Centralized configuration management with environment-specific overrides,
validation, and hot-reloading capabilities.
"""

import os
import yaml
import json
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, field
from pathlib import Path
import logging
from datetime import datetime
import threading
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

logger = logging.getLogger(__name__)


@dataclass
class ServerConfig:
    """Server configuration."""
    host: str = "127.0.0.1"
    port: int = 5000
    debug: bool = False
    threaded: bool = True
    processes: int = 1
    ssl_cert: Optional[str] = None
    ssl_key: Optional[str] = None


@dataclass
class PathsConfig:
    """File paths configuration."""
    upload_folder: str = "output/temp"
    output_folder: str = "output"
    logs_folder: str = "logs"
    temp_folder: str = "temp"
    cache_folder: str = "cache"
    templates_folder: str = "templates"
    static_folder: str = "static"


@dataclass
class LoggingConfig:
    """Logging configuration."""
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    console_output: bool = True
    file_output: bool = True
    file_rotation: Dict[str, Any] = field(default_factory=lambda: {
        "enabled": True,
        "max_bytes": 10485760,  # 10MB
        "backup_count": 5
    })
    structured_logging: bool = True
    log_sql_queries: bool = False


@dataclass
class PerformanceConfig:
    """Performance configuration."""
    cache_enabled: bool = True
    cache_size: int = 100
    cache_ttl: int = 3600
    max_workers: int = 4
    timeout: int = 300
    connection_pool_size: int = 10
    max_connections: int = 100
    enable_compression: bool = True
    compression_level: int = 6


@dataclass
class SecurityConfig:
    """Security configuration."""
    secret_key: Optional[str] = None
    jwt_secret_key: Optional[str] = None
    master_password: Optional[str] = None
    max_upload_size: int = 33554432  # 32MB
    allowed_extensions: List[str] = field(default_factory=lambda: [
        "dxf", "jpg", "jpeg", "png", "bmp", "tiff", "pdf"
    ])
    enable_cors: bool = False
    cors_origins: List[str] = field(default_factory=lambda: [
        "http://localhost:5000",
        "http://127.0.0.1:5000",
        "http://localhost:8501",
        "http://127.0.0.1:8501"
    ])
    security_headers: Dict[str, str] = field(default_factory=lambda: {
        "x_content_type_options": "nosniff",
        "x_frame_options": "DENY",
        "x_xss_protection": "1; mode=block",
        "strict_transport_security": "max-age=31536000; includeSubDomains"
    })
    session_settings: Dict[str, Any] = field(default_factory=lambda: {
        "permanent": False,
        "timeout": 3600,
        "secure": False,
        "httponly": True,
        "samesite": "Lax"
    })
    rate_limiting: Dict[str, Any] = field(default_factory=lambda: {
        "enabled": True,
        "requests_per_minute": 60,
        "burst_limit": 10,
        "window_size": 60
    })
    file_validation: Dict[str, Any] = field(default_factory=lambda: {
        "max_filename_length": 255,
        "allowed_chars": "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-",
        "scan_for_malware": False
    })


@dataclass
class AIConfig:
    """AI configuration."""
    openai: Dict[str, Any] = field(default_factory=lambda: {
        "api_key": None,
        "model": "gpt-4",
        "max_tokens": 2000,
        "temperature": 0.7,
        "timeout": 30
    })
    ollama: Dict[str, Any] = field(default_factory=lambda: {
        "base_url": "http://localhost:11434",
        "model": "llama2",
        "timeout": 60
    })
    enabled: bool = True
    fallback_enabled: bool = True


@dataclass
class DatabaseConfig:
    """Database configuration."""
    type: str = "sqlite"  # sqlite, postgresql, mysql
    host: str = "localhost"
    port: int = 5432
    name: str = "wjp_analyser"
    username: str = "wjp_user"
    password: Optional[str] = None
    pool_size: int = 10
    max_overflow: int = 20
    echo: bool = False
    ssl_mode: str = "prefer"


@dataclass
class RedisConfig:
    """Redis configuration."""
    host: str = "localhost"
    port: int = 6379
    password: Optional[str] = None
    db: int = 0
    max_connections: int = 20
    socket_timeout: int = 5
    socket_connect_timeout: int = 5
    retry_on_timeout: bool = True


@dataclass
class CeleryConfig:
    """Celery configuration."""
    broker_url: str = "redis://localhost:6379/0"
    result_backend: str = "redis://localhost:6379/0"
    task_serializer: str = "json"
    result_serializer: str = "json"
    accept_content: List[str] = field(default_factory=lambda: ["json"])
    timezone: str = "UTC"
    enable_utc: bool = True
    task_track_started: bool = True
    task_time_limit: int = 300
    task_soft_time_limit: int = 240
    worker_prefetch_multiplier: int = 1
    worker_max_tasks_per_child: int = 1000


@dataclass
class FeaturesConfig:
    """Feature flags configuration."""
    ai_analysis: bool = True
    nesting: bool = True
    cost_estimation: bool = True
    guided_mode: bool = True
    batch_processing: bool = True
    real_time_preview: bool = True
    collaborative_features: bool = False
    api_access: bool = True
    webhooks: bool = False


@dataclass
class DefaultsConfig:
    """Default parameters configuration."""
    material: str = "steel"
    thickness: float = 6.0
    kerf: float = 1.1
    cutting_speed: float = 1200.0
    cost_per_meter: float = 50.0
    sheet_width: float = 3000.0
    sheet_height: float = 1500.0
    spacing: float = 10.0
    quality_level: str = "high"


@dataclass
class ImageProcessingConfig:
    """Image processing configuration."""
    edge_threshold: float = 0.33
    min_contour_area: int = 100
    simplify_tolerance: float = 0.02
    blur_kernel_size: int = 5
    canny_low: int = 50
    canny_high: int = 150
    max_image_size: int = 4096
    supported_formats: List[str] = field(default_factory=lambda: [
        "jpg", "jpeg", "png", "bmp", "tiff", "webp"
    ])


@dataclass
class MonitoringConfig:
    """Monitoring configuration."""
    prometheus_enabled: bool = True
    prometheus_port: int = 8000
    grafana_enabled: bool = True
    grafana_port: int = 3000
    elk_enabled: bool = True
    jaeger_enabled: bool = True
    jaeger_endpoint: str = "http://localhost:14268/api/traces"
    metrics_interval: int = 30
    log_level: str = "INFO"


@dataclass
class DevelopmentConfig:
    """Development configuration."""
    hot_reload: bool = False
    profile_requests: bool = False
    debug_toolbar: bool = False
    mock_ai_responses: bool = False
    enable_caching: bool = True
    verbose_logging: bool = False
    test_mode: bool = False


@dataclass
class UnifiedConfig:
    """Unified configuration container."""
    environment: str = "development"
    version: str = "1.0.0"
    server: ServerConfig = field(default_factory=ServerConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    ai: AIConfig = field(default_factory=AIConfig)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    redis: RedisConfig = field(default_factory=RedisConfig)
    celery: CeleryConfig = field(default_factory=CeleryConfig)
    features: FeaturesConfig = field(default_factory=FeaturesConfig)
    defaults: DefaultsConfig = field(default_factory=DefaultsConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    development: DevelopmentConfig = field(default_factory=DevelopmentConfig)


class ConfigFileHandler(FileSystemEventHandler):
    """File system event handler for config file changes."""
    
    def __init__(self, config_manager):
        self.config_manager = config_manager
    
    def on_modified(self, event):
        if event.is_directory:
            return
        
        if event.src_path.endswith('.yaml') or event.src_path.endswith('.yml'):
            logger.info(f"Config file changed: {event.src_path}")
            self.config_manager.reload_config()


class UnifiedConfigManager:
    """Unified configuration manager with hot-reloading."""
    
    def __init__(self, config_file: str = "config/unified_config.yaml"):
        self.config_file = config_file
        self.config = UnifiedConfig()
        self.observer = None
        self.lock = threading.Lock()
        self.callbacks: List[callable] = []
        
        # Load initial configuration
        self.load_config()
        
        # Setup file watching for hot-reloading
        self.setup_file_watching()
    
    def load_config(self):
        """Load configuration from file and environment variables."""
        try:
            # Load from YAML file if it exists
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r') as f:
                    config_data = yaml.safe_load(f)
                
                # Merge with current config
                self._merge_config(config_data)
            
            # Override with environment variables
            self._load_from_environment()
            
            # Validate configuration
            self._validate_config()
            
            logger.info("Configuration loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            raise
    
    def _merge_config(self, config_data: Dict[str, Any]):
        """Merge configuration data into current config."""
        if not config_data:
            return
        
        # Update environment
        if 'environment' in config_data:
            self.config.environment = config_data['environment']
        
        # Update server config
        if 'server' in config_data:
            self._update_dataclass(self.config.server, config_data['server'])
        
        # Update paths config
        if 'paths' in config_data:
            self._update_dataclass(self.config.paths, config_data['paths'])
        
        # Update logging config
        if 'logging' in config_data:
            self._update_dataclass(self.config.logging, config_data['logging'])
        
        # Update performance config
        if 'performance' in config_data:
            self._update_dataclass(self.config.performance, config_data['performance'])
        
        # Update security config
        if 'security' in config_data:
            self._update_dataclass(self.config.security, config_data['security'])
        
        # Update AI config
        if 'ai' in config_data:
            self._update_dataclass(self.config.ai, config_data['ai'])
        
        # Update database config
        if 'database' in config_data:
            self._update_dataclass(self.config.database, config_data['database'])
        
        # Update Redis config
        if 'redis' in config_data:
            self._update_dataclass(self.config.redis, config_data['redis'])
        
        # Update Celery config
        if 'celery' in config_data:
            self._update_dataclass(self.config.celery, config_data['celery'])
        
        # Update features config
        if 'features' in config_data:
            self._update_dataclass(self.config.features, config_data['features'])
        
        # Update defaults config
        if 'defaults' in config_data:
            self._update_dataclass(self.config.defaults, config_data['defaults'])
        
        
        # Update monitoring config
        if 'monitoring' in config_data:
            self._update_dataclass(self.config.monitoring, config_data['monitoring'])
        
        # Update development config
        if 'development' in config_data:
            self._update_dataclass(self.config.development, config_data['development'])
    
    def _update_dataclass(self, obj, data: Dict[str, Any]):
        """Update dataclass fields from dictionary."""
        for key, value in data.items():
            if hasattr(obj, key):
                setattr(obj, key, value)
    
    def _load_from_environment(self):
        """Load configuration from environment variables."""
        # Server settings
        if os.getenv('WJP_HOST'):
            self.config.server.host = os.getenv('WJP_HOST')
        if os.getenv('WJP_PORT'):
            self.config.server.port = int(os.getenv('WJP_PORT'))
        if os.getenv('WJP_DEBUG'):
            self.config.server.debug = os.getenv('WJP_DEBUG').lower() == 'true'
        
        # Security settings
        if os.getenv('WJP_SECRET_KEY'):
            self.config.security.secret_key = os.getenv('WJP_SECRET_KEY')
        if os.getenv('WJP_JWT_SECRET_KEY'):
            self.config.security.jwt_secret_key = os.getenv('WJP_JWT_SECRET_KEY')
        if os.getenv('WJP_MASTER_PASSWORD'):
            self.config.security.master_password = os.getenv('WJP_MASTER_PASSWORD')
        
        # Database settings
        if os.getenv('WJP_DATABASE_TYPE'):
            self.config.database.type = os.getenv('WJP_DATABASE_TYPE')
        if os.getenv('WJP_DATABASE_HOST'):
            self.config.database.host = os.getenv('WJP_DATABASE_HOST')
        if os.getenv('WJP_DATABASE_PORT'):
            self.config.database.port = int(os.getenv('WJP_DATABASE_PORT'))
        if os.getenv('WJP_DATABASE_NAME'):
            self.config.database.name = os.getenv('WJP_DATABASE_NAME')
        if os.getenv('WJP_DATABASE_USERNAME'):
            self.config.database.username = os.getenv('WJP_DATABASE_USERNAME')
        if os.getenv('WJP_DATABASE_PASSWORD'):
            self.config.database.password = os.getenv('WJP_DATABASE_PASSWORD')
        
        # Redis settings
        if os.getenv('WJP_REDIS_HOST'):
            self.config.redis.host = os.getenv('WJP_REDIS_HOST')
        if os.getenv('WJP_REDIS_PORT'):
            self.config.redis.port = int(os.getenv('WJP_REDIS_PORT'))
        if os.getenv('WJP_REDIS_PASSWORD'):
            self.config.redis.password = os.getenv('WJP_REDIS_PASSWORD')
        
        # AI settings
        if os.getenv('OPENAI_API_KEY'):
            self.config.ai.openai['api_key'] = os.getenv('OPENAI_API_KEY')
        if os.getenv('OLLAMA_BASE_URL'):
            self.config.ai.ollama['base_url'] = os.getenv('OLLAMA_BASE_URL')
        
        # Environment
        if os.getenv('WJP_ENVIRONMENT'):
            self.config.environment = os.getenv('WJP_ENVIRONMENT')
    
    def _validate_config(self):
        """Validate configuration values."""
        # Validate server port
        if not (1 <= self.config.server.port <= 65535):
            raise ValueError(f"Invalid server port: {self.config.server.port}")
        
        # Validate database type
        if self.config.database.type not in ['sqlite', 'postgresql', 'mysql']:
            raise ValueError(f"Invalid database type: {self.config.database.type}")
        
        # Validate log level
        valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if self.config.logging.level.upper() not in valid_levels:
            raise ValueError(f"Invalid log level: {self.config.logging.level}")
        
        # Validate environment
        valid_environments = ['development', 'staging', 'production']
        if self.config.environment not in valid_environments:
            raise ValueError(f"Invalid environment: {self.config.environment}")
    
    def setup_file_watching(self):
        """Setup file watching for hot-reloading."""
        try:
            config_dir = os.path.dirname(self.config_file)
            if os.path.exists(config_dir):
                self.observer = Observer()
                event_handler = ConfigFileHandler(self)
                self.observer.schedule(event_handler, config_dir, recursive=False)
                self.observer.start()
                logger.info("Configuration file watching enabled")
        except Exception as e:
            logger.warning(f"Failed to setup file watching: {e}")
    
    def reload_config(self):
        """Reload configuration from file."""
        with self.lock:
            try:
                old_config = self.config
                self.load_config()
                
                # Notify callbacks
                for callback in self.callbacks:
                    try:
                        callback(old_config, self.config)
                    except Exception as e:
                        logger.error(f"Config reload callback failed: {e}")
                
                logger.info("Configuration reloaded successfully")
                
            except Exception as e:
                logger.error(f"Failed to reload configuration: {e}")
                raise
    
    def add_reload_callback(self, callback: callable):
        """Add callback for configuration reload events."""
        self.callbacks.append(callback)
    
    def get_config(self) -> UnifiedConfig:
        """Get current configuration."""
        return self.config
    
    def get_database_url(self) -> str:
        """Get database URL."""
        if self.config.database.type == 'sqlite':
            return f"sqlite:///{self.config.database.name}.db"
        elif self.config.database.type == 'postgresql':
            password = f":{self.config.database.password}" if self.config.database.password else ""
            return f"postgresql://{self.config.database.username}{password}@{self.config.database.host}:{self.config.database.port}/{self.config.database.name}"
        elif self.config.database.type == 'mysql':
            password = f":{self.config.database.password}" if self.config.database.password else ""
            return f"mysql://{self.config.database.username}{password}@{self.config.database.host}:{self.config.database.port}/{self.config.database.name}"
        else:
            raise ValueError(f"Unsupported database type: {self.config.database.type}")
    
    def get_redis_url(self) -> str:
        """Get Redis URL."""
        password = f":{self.config.redis.password}@" if self.config.redis.password else ""
        return f"redis://{password}{self.config.redis.host}:{self.config.redis.port}/{self.config.redis.db}"
    
    def save_config(self, file_path: Optional[str] = None):
        """Save current configuration to file."""
        file_path = file_path or self.config_file
        
        try:
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            # Convert config to dictionary
            config_dict = self._config_to_dict(self.config)
            
            with open(file_path, 'w') as f:
                yaml.dump(config_dict, f, default_flow_style=False, indent=2)
            
            logger.info(f"Configuration saved to: {file_path}")
            
        except Exception as e:
            logger.error(f"Failed to save configuration: {e}")
            raise
    
    def _config_to_dict(self, config: UnifiedConfig) -> Dict[str, Any]:
        """Convert config object to dictionary."""
        result = {}
        
        for field_name, field_value in config.__dict__.items():
            if hasattr(field_value, '__dict__'):
                result[field_name] = self._config_to_dict(field_value)
            else:
                result[field_name] = field_value
        
        return result
    
    def stop(self):
        """Stop the configuration manager."""
        if self.observer:
            self.observer.stop()
            self.observer.join()
            logger.info("Configuration manager stopped")


# Global configuration manager instance
config_manager = UnifiedConfigManager()


def get_config() -> UnifiedConfig:
    """Get the global configuration."""
    return config_manager.get_config()


def reload_config():
    """Reload the global configuration."""
    config_manager.reload_config()


def get_database_url() -> str:
    """Get database URL from global configuration."""
    return config_manager.get_database_url()


def get_redis_url() -> str:
    """Get Redis URL from global configuration."""
    return config_manager.get_redis_url()
