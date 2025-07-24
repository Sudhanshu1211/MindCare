"""
Utilities for AES encryption and decryption (shared by modules).
"""

from cryptography.fernet import Fernet, InvalidToken
import base64
import hashlib

def _get_fernet(key: str) -> Fernet:
    # Derive a 32-byte base64 key from the provided string
    digest = hashlib.sha256(key.encode()).digest()
    b64key = base64.urlsafe_b64encode(digest)
    return Fernet(b64key)

def encrypt_data(data: bytes, key: str) -> bytes:
    """
    Encrypts bytes using AES (Fernet).
    """
    f = _get_fernet(key)
    return f.encrypt(data)

def decrypt_data(token: bytes, key: str) -> bytes:
    """
    Decrypts bytes using AES (Fernet).
    """
    f = _get_fernet(key)
    try:
        return f.decrypt(token)
    except InvalidToken as e:
        raise ValueError("Decryption failed: Invalid token or wrong key") from e
