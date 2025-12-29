"""
Feature engineering module for MLOps homework.
Implements deterministic feature hashing for string inputs.
"""
import hashlib


def hash_feature(input_string: str, num_buckets: int = 1000) -> int:
    """
    Hash a string input to a deterministic integer bucket index.
    
    Args:
        input_string: The string to hash
        num_buckets: Number of buckets for the hash (default: 1000)
    
    Returns:
        An integer bucket index between 0 and num_buckets - 1
    
    Example:
        >>> hash_feature("user_123")
        456
    """
    # Create a deterministic hash using MD5
    hash_object = hashlib.md5(input_string.encode('utf-8'))
    hash_hex = hash_object.hexdigest()
    
    # Convert hex string to integer and take modulo for bucket index
    hash_int = int(hash_hex, 16)
    bucket_index = hash_int % num_buckets
    
    return bucket_index

