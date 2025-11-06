"""
API Key Encryption Manager
==========================

Handles encryption and decryption of API keys and other sensitive configuration data.
"""

import os
import base64
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


class APIKeyEncryptionManager:
    """Manages encryption and decryption of API keys and sensitive data."""
    
    def __init__(self, master_password: Optional[str] = None):
        """
        Initialize encryption manager.
        
        Args:
            master_password: Master password for encryption. If None, uses environment variable.
        """
        self.master_password = master_password or os.getenv('WJP_MASTER_PASSWORD')
        if not self.master_password:
            # Generate a default master password (should be changed in production)
            self.master_password = "wjp-default-master-password-change-in-production"
            logger.warning("Using default master password. Set WJP_MASTER_PASSWORD environment variable.")
        
        self.salt = b'wjp_salt_2024'  # In production, use a random salt per installation
        self._fernet = self._create_fernet()
    
    def _create_fernet(self) -> Fernet:
        """Create Fernet cipher instance."""
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=self.salt,
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(self.master_password.encode()))
        return Fernet(key)
    
    def encrypt_api_key(self, api_key: str) -> str:
        """
        Encrypt an API key.
        
        Args:
            api_key: Plain text API key
            
        Returns:
            Encrypted API key (base64 encoded)
        """
        try:
            encrypted_data = self._fernet.encrypt(api_key.encode())
            encrypted_key = base64.urlsafe_b64encode(encrypted_data).decode()
            logger.info("API key encrypted successfully")
            return encrypted_key
        except Exception as e:
            logger.error(f"Failed to encrypt API key: {e}")
            raise
    
    def decrypt_api_key(self, encrypted_api_key: str) -> str:
        """
        Decrypt an API key.
        
        Args:
            encrypted_api_key: Encrypted API key (base64 encoded)
            
        Returns:
            Plain text API key
            
        Raises:
            ValueError: If decryption fails
        """
        try:
            encrypted_data = base64.urlsafe_b64decode(encrypted_api_key.encode())
            decrypted_data = self._fernet.decrypt(encrypted_data)
            api_key = decrypted_data.decode()
            logger.info("API key decrypted successfully")
            return api_key
        except Exception as e:
            logger.error(f"Failed to decrypt API key: {e}")
            raise ValueError("Invalid encrypted API key")
    
    def encrypt_config_value(self, key: str, value: str) -> str:
        """
        Encrypt a configuration value.
        
        Args:
            key: Configuration key name
            value: Plain text value
            
        Returns:
            Encrypted value (base64 encoded)
        """
        try:
            # Include key name in encryption for additional security
            data_to_encrypt = f"{key}:{value}"
            encrypted_data = self._fernet.encrypt(data_to_encrypt.encode())
            encrypted_value = base64.urlsafe_b64encode(encrypted_data).decode()
            logger.debug(f"Configuration value encrypted for key: {key}")
            return encrypted_value
        except Exception as e:
            logger.error(f"Failed to encrypt config value for key {key}: {e}")
            raise
    
    def decrypt_config_value(self, key: str, encrypted_value: str) -> str:
        """
        Decrypt a configuration value.
        
        Args:
            key: Configuration key name
            encrypted_value: Encrypted value (base64 encoded)
            
        Returns:
            Plain text value
            
        Raises:
            ValueError: If decryption fails or key doesn't match
        """
        try:
            encrypted_data = base64.urlsafe_b64decode(encrypted_value.encode())
            decrypted_data = self._fernet.decrypt(encrypted_data)
            decrypted_string = decrypted_data.decode()
            
            # Verify key matches
            if not decrypted_string.startswith(f"{key}:"):
                raise ValueError("Key mismatch in decrypted data")
            
            value = decrypted_string[len(f"{key}:"):]
            logger.debug(f"Configuration value decrypted for key: {key}")
            return value
        except Exception as e:
            logger.error(f"Failed to decrypt config value for key {key}: {e}")
            raise ValueError("Invalid encrypted configuration value")
    
    def encrypt_sensitive_data(self, data: Dict[str, Any]) -> Dict[str, str]:
        """
        Encrypt all sensitive data in a dictionary.
        
        Args:
            data: Dictionary containing sensitive data
            
        Returns:
            Dictionary with encrypted values
        """
        encrypted_data = {}
        
        for key, value in data.items():
            if isinstance(value, str) and value.strip():
                try:
                    encrypted_data[key] = self.encrypt_config_value(key, value)
                except Exception as e:
                    logger.error(f"Failed to encrypt {key}: {e}")
                    # Keep original value if encryption fails
                    encrypted_data[key] = value
            else:
                encrypted_data[key] = value
        
        return encrypted_data
    
    def decrypt_sensitive_data(self, encrypted_data: Dict[str, str]) -> Dict[str, str]:
        """
        Decrypt all sensitive data in a dictionary.
        
        Args:
            encrypted_data: Dictionary containing encrypted values
            
        Returns:
            Dictionary with decrypted values
        """
        decrypted_data = {}
        
        for key, value in encrypted_data.items():
            if isinstance(value, str) and value.strip():
                try:
                    decrypted_data[key] = self.decrypt_config_value(key, value)
                except ValueError:
                    # If decryption fails, assume it's not encrypted
                    decrypted_data[key] = value
                except Exception as e:
                    logger.error(f"Failed to decrypt {key}: {e}")
                    decrypted_data[key] = value
            else:
                decrypted_data[key] = value
        
        return decrypted_data
    
    def is_encrypted(self, value: str) -> bool:
        """
        Check if a value appears to be encrypted.
        
        Args:
            value: Value to check
            
        Returns:
            True if value appears to be encrypted, False otherwise
        """
        try:
            # Try to decode as base64
            base64.urlsafe_b64decode(value.encode())
            return True
        except Exception:
            return False
    
    def generate_new_master_password(self) -> str:
        """
        Generate a new secure master password.
        
        Returns:
            New secure master password
        """
        import secrets
        import string
        
        # Generate a 32-character password with mixed case, digits, and symbols
        characters = string.ascii_letters + string.digits + "!@#$%^&*"
        password = ''.join(secrets.choice(characters) for _ in range(32))
        
        logger.info("Generated new master password")
        return password
    
    def update_master_password(self, new_master_password: str) -> None:
        """
        Update the master password and re-encrypt all data.
        
        Args:
            new_master_password: New master password
        """
        # This would need to be implemented with a database to re-encrypt all stored data
        # For now, we'll just update the current instance
        self.master_password = new_master_password
        self._fernet = self._create_fernet()
        logger.info("Master password updated")
