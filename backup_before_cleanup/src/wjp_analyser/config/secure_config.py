"""
Secure Configuration Management for WJP Analyser
===============================================

This module provides secure configuration management with proper secret handling,
environment variable support, and validation.
"""

import os
import secrets
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class SecurityConfig:
    """Security-related configuration."""
    secret_key: str
    max_upload_size: int = 32 * 1024 * 1024  # 32MB
    allowed_extensions: set = None
    enable_cors: bool = False
    cors_origins: list = None
    
    def __post_init__(self):
        if self.allowed_extensions is None:
            self.allowed_extensions = {"dxf", "jpg", "jpeg", "png", "bmp", "tiff"}
        if self.cors_origins is None:
            self.cors_origins = ["http://localhost:5000", "http://127.0.0.1:5000"]


@dataclass
class AIConfig:
    """AI service configuration."""
    openai_api_key: Optional[str] = None
    openai_model: str = "gpt-4o-mini"
    openai_image_model: str = "dall-e-3"
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "waterjet:latest"
    ai_timeout: int = 120


@dataclass
class AppConfig:
    """Application configuration."""
    debug: bool = False
    host: str = "127.0.0.1"
    port: int = 5000
    upload_folder: str = "output/temp"
    output_folder: str = "output"
    log_level: str = "INFO"


class SecureConfigManager:
    """Secure configuration manager with proper secret handling."""
    
    def __init__(self, config_dir: Optional[str] = None):
        self.config_dir = Path(config_dir) if config_dir else Path("config")
        self._config_cache: Dict[str, Any] = {}
        
    def _load_yaml_config(self, filename: str) -> Dict[str, Any]:
        """Load configuration from YAML file with error handling."""
        config_path = self.config_dir / filename
        
        if not config_path.exists():
            logger.warning(f"Configuration file not found: {config_path}")
            return {}
            
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f) or {}
        except yaml.YAMLError as e:
            logger.error(f"Error parsing YAML config {config_path}: {e}")
            return {}
        except Exception as e:
            logger.error(f"Error loading config {config_path}: {e}")
            return {}
    
    def _get_secret_key(self) -> str:
        """Get secret key securely from environment or generate one."""
        # Try environment variable first
        secret_key = os.environ.get("SECRET_KEY")
        if secret_key and len(secret_key) >= 32:
            return secret_key
            
        # Try config file
        config = self._load_yaml_config("api_keys.yaml")
        secret_key = config.get("secret_key")
        if secret_key and len(secret_key) >= 32:
            return secret_key
            
        # Generate secure key for development
        generated_key = secrets.token_hex(32)
        logger.warning("Generated temporary secret key for development!")
        logger.warning("Set SECRET_KEY environment variable for production!")
        return generated_key
    
    def _get_openai_key(self) -> Optional[str]:
        """Get OpenAI API key securely."""
        # Try environment variable first
        api_key = os.environ.get("OPENAI_API_KEY")
        if api_key:
            return api_key
            
        # Try config file
        config = self._load_yaml_config("api_keys.yaml")
        openai_config = config.get("openai", {})
        api_key = openai_config.get("api_key")
        
        if api_key and api_key.startswith("sk-"):
            return api_key
            
        logger.warning("OpenAI API key not found. AI features will be disabled.")
        return None
    
    def get_security_config(self) -> SecurityConfig:
        """Get security configuration."""
        if "security" in self._config_cache:
            return self._config_cache["security"]
            
        # Load from environment and config files
        config = self._load_yaml_config("security.yaml")
        
        security_config = SecurityConfig(
            secret_key=self._get_secret_key(),
            max_upload_size=int(os.environ.get("MAX_UPLOAD_SIZE", 
                                              config.get("max_upload_size", 32 * 1024 * 1024))),
            allowed_extensions=set(config.get("allowed_extensions", 
                                            ["dxf", "jpg", "jpeg", "png", "bmp", "tiff"])),
            enable_cors=config.get("enable_cors", False),
            cors_origins=config.get("cors_origins", ["http://localhost:5000"])
        )
        
        self._config_cache["security"] = security_config
        return security_config
    
    def get_ai_config(self) -> AIConfig:
        """Get AI configuration."""
        if "ai" in self._config_cache:
            return self._config_cache["ai"]
            
        config = self._load_yaml_config("ai_config.yaml")
        
        ai_config = AIConfig(
            openai_api_key=self._get_openai_key(),
            openai_model=os.environ.get("OPENAI_MODEL", 
                                      config.get("openai", {}).get("model", "gpt-4o-mini")),
            openai_image_model=os.environ.get("OPENAI_IMAGE_MODEL",
                                            config.get("openai", {}).get("image_model", "dall-e-3")),
            ollama_base_url=os.environ.get("OLLAMA_BASE_URL",
                                         config.get("ollama", {}).get("base_url", "http://localhost:11434")),
            ollama_model=os.environ.get("OLLAMA_MODEL",
                                      config.get("ollama", {}).get("model", "waterjet:latest")),
            ai_timeout=int(os.environ.get("AI_TIMEOUT",
                                        config.get("ai_timeout", 120)))
        )
        
        self._config_cache["ai"] = ai_config
        return ai_config
    
    def get_app_config(self) -> AppConfig:
        """Get application configuration."""
        if "app" in self._config_cache:
            return self._config_cache["app"]
            
        config = self._load_yaml_config("app_config.yaml")
        
        app_config = AppConfig(
            debug=os.environ.get("DEBUG", "false").lower() == "true",
            host=os.environ.get("HOST", config.get("host", "127.0.0.1")),
            port=int(os.environ.get("PORT", config.get("port", 5000))),
            upload_folder=os.environ.get("UPLOAD_FOLDER", config.get("upload_folder", "output/temp")),
            output_folder=os.environ.get("OUTPUT_FOLDER", config.get("output_folder", "output")),
            log_level=os.environ.get("LOG_LEVEL", config.get("log_level", "INFO"))
        )
        
        self._config_cache["app"] = app_config
        return app_config
    
    def validate_config(self) -> bool:
        """Validate configuration and log any issues."""
        issues = []
        
        # Validate security config
        security = self.get_security_config()
        if len(security.secret_key) < 32:
            issues.append("Secret key is too short (minimum 32 characters)")
            
        # Validate AI config (only warn, don't fail)
        ai = self.get_ai_config()
        if not ai.openai_api_key:
            logger.warning("OpenAI API key not configured - AI features disabled")
            # Don't add to issues list as this is optional
            
        # Log issues
        if issues:
            for issue in issues:
                logger.warning(f"Configuration issue: {issue}")
            return False
            
        logger.info("Configuration validation passed")
        return True


# Global config manager instance
config_manager = SecureConfigManager()


def get_security_config() -> SecurityConfig:
    """Get security configuration."""
    return config_manager.get_security_config()


def get_ai_config() -> AIConfig:
    """Get AI configuration."""
    return config_manager.get_ai_config()


def get_app_config() -> AppConfig:
    """Get application configuration."""
    return config_manager.get_app_config()


def validate_config() -> bool:
    """Validate configuration."""
    return config_manager.validate_config()
