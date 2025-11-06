"""
Secure API Key Management
=========================

Encrypted storage and management of API keys.
"""

import os
import base64
import json
from typing import Dict, Any, Optional
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class APIKey:
    """API Key data structure."""
    name: str
    encrypted_key: str
    service: str
    created_at: str
    last_used: Optional[str] = None
    is_active: bool = True


class APIKeyManager:
    """Secure API key management with encryption."""
    
    def __init__(self, master_password: Optional[str] = None):
        self.master_password = master_password or os.getenv('WJP_MASTER_PASSWORD', 'change-this-in-production')
        self.fernet = self._create_fernet()
        self.keys_file = 'config/encrypted_api_keys.json'
        self.keys: Dict[str, APIKey] = {}
        self._load_keys()
    
    def _create_fernet(self) -> Fernet:
        """Create Fernet encryption instance."""
        # Derive key from master password
        password = self.master_password.encode()
        salt = b'wjp_analyser_salt'  # In production, use random salt
        
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(password))
        return Fernet(key)
    
    def _load_keys(self):
        """Load encrypted keys from file."""
        try:
            if os.path.exists(self.keys_file):
                with open(self.keys_file, 'r') as f:
                    data = json.load(f)
                
                for key_name, key_data in data.items():
                    self.keys[key_name] = APIKey(**key_data)
                
                logger.info(f"Loaded {len(self.keys)} encrypted API keys")
        except Exception as e:
            logger.error(f"Failed to load API keys: {e}")
            self.keys = {}
    
    def _save_keys(self):
        """Save encrypted keys to file."""
        try:
            os.makedirs(os.path.dirname(self.keys_file), exist_ok=True)
            
            data = {}
            for key_name, api_key in self.keys.items():
                data[key_name] = {
                    'name': api_key.name,
                    'encrypted_key': api_key.encrypted_key,
                    'service': api_key.service,
                    'created_at': api_key.created_at,
                    'last_used': api_key.last_used,
                    'is_active': api_key.is_active
                }
            
            with open(self.keys_file, 'w') as f:
                json.dump(data, f, indent=2)
            
            logger.info("API keys saved successfully")
        except Exception as e:
            logger.error(f"Failed to save API keys: {e}")
    
    def add_api_key(self, name: str, api_key: str, service: str) -> bool:
        """Add new API key."""
        try:
            # Encrypt the API key
            encrypted_key = self.fernet.encrypt(api_key.encode()).decode()
            
            # Create APIKey object
            api_key_obj = APIKey(
                name=name,
                encrypted_key=encrypted_key,
                service=service,
                created_at=str(os.path.getctime(self.keys_file) if os.path.exists(self.keys_file) else 0)
            )
            
            self.keys[name] = api_key_obj
            self._save_keys()
            
            logger.info(f"Added API key: {name} for service: {service}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add API key {name}: {e}")
            return False
    
    def get_api_key(self, name: str) -> Optional[str]:
        """Get decrypted API key."""
        if name not in self.keys:
            logger.warning(f"API key not found: {name}")
            return None
        
        api_key_obj = self.keys[name]
        
        if not api_key_obj.is_active:
            logger.warning(f"API key is inactive: {name}")
            return None
        
        try:
            # Decrypt the API key
            decrypted_key = self.fernet.decrypt(api_key_obj.encrypted_key.encode()).decode()
            
            # Update last used timestamp
            api_key_obj.last_used = str(os.path.getctime(self.keys_file))
            self._save_keys()
            
            logger.info(f"Retrieved API key: {name}")
            return decrypted_key
            
        except Exception as e:
            logger.error(f"Failed to decrypt API key {name}: {e}")
            return None
    
    def list_api_keys(self) -> Dict[str, Dict[str, Any]]:
        """List all API keys (without decrypted values)."""
        result = {}
        for name, api_key_obj in self.keys.items():
            result[name] = {
                'name': api_key_obj.name,
                'service': api_key_obj.service,
                'created_at': api_key_obj.created_at,
                'last_used': api_key_obj.last_used,
                'is_active': api_key_obj.is_active
            }
        return result
    
    def update_api_key(self, name: str, new_api_key: str) -> bool:
        """Update existing API key."""
        if name not in self.keys:
            logger.warning(f"API key not found for update: {name}")
            return False
        
        try:
            # Encrypt the new API key
            encrypted_key = self.fernet.encrypt(new_api_key.encode()).decode()
            
            # Update the key
            self.keys[name].encrypted_key = encrypted_key
            self._save_keys()
            
            logger.info(f"Updated API key: {name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update API key {name}: {e}")
            return False
    
    def delete_api_key(self, name: str) -> bool:
        """Delete API key."""
        if name not in self.keys:
            logger.warning(f"API key not found for deletion: {name}")
            return False
        
        try:
            del self.keys[name]
            self._save_keys()
            
            logger.info(f"Deleted API key: {name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete API key {name}: {e}")
            return False
    
    def deactivate_api_key(self, name: str) -> bool:
        """Deactivate API key."""
        if name not in self.keys:
            logger.warning(f"API key not found for deactivation: {name}")
            return False
        
        try:
            self.keys[name].is_active = False
            self._save_keys()
            
            logger.info(f"Deactivated API key: {name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to deactivate API key {name}: {e}")
            return False
    
    def activate_api_key(self, name: str) -> bool:
        """Activate API key."""
        if name not in self.keys:
            logger.warning(f"API key not found for activation: {name}")
            return False
        
        try:
            self.keys[name].is_active = True
            self._save_keys()
            
            logger.info(f"Activated API key: {name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to activate API key {name}: {e}")
            return False
    
    def migrate_from_plaintext(self, plaintext_file: str) -> bool:
        """Migrate from plaintext API keys file."""
        try:
            if not os.path.exists(plaintext_file):
                logger.warning(f"Plaintext file not found: {plaintext_file}")
                return False
            
            import yaml
            with open(plaintext_file, 'r') as f:
                data = yaml.safe_load(f)
            
            migrated_count = 0
            for service, config in data.items():
                if isinstance(config, dict) and 'api_key' in config:
                    api_key = config['api_key']
                    if api_key and api_key != "your-api-key-here":
                        if self.add_api_key(f"{service}_key", api_key, service):
                            migrated_count += 1
            
            logger.info(f"Migrated {migrated_count} API keys from plaintext")
            return migrated_count > 0
            
        except Exception as e:
            logger.error(f"Failed to migrate from plaintext: {e}")
            return False


# Global API key manager instance
api_key_manager = APIKeyManager()


def get_api_key(service: str) -> Optional[str]:
    """Get API key for service."""
    return api_key_manager.get_api_key(f"{service}_key")


def set_api_key(service: str, api_key: str) -> bool:
    """Set API key for service."""
    return api_key_manager.add_api_key(f"{service}_key", api_key, service)


# Migration function
def migrate_existing_keys():
    """Migrate existing plaintext keys to encrypted storage."""
    plaintext_file = "config/api_keys.yaml"
    if os.path.exists(plaintext_file):
        logger.info("Migrating existing API keys to encrypted storage...")
        success = api_key_manager.migrate_from_plaintext(plaintext_file)
        if success:
            logger.info("Migration completed successfully")
            # Backup the original file
            backup_file = f"{plaintext_file}.backup"
            os.rename(plaintext_file, backup_file)
            logger.info(f"Original file backed up to: {backup_file}")
        else:
            logger.warning("Migration failed or no keys to migrate")
    else:
        logger.info("No existing API keys file found")
