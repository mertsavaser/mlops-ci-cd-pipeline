"""
Unit tests for feature engineering logic.
These tests are fast, isolated, and have no external dependencies.
"""
import pytest
from app.features import hash_feature


def test_hash_feature_deterministic():
    """
    Test that the hash function is deterministic.
    Same input should always produce the same output.
    """
    input_string = "test_user_123"
    result1 = hash_feature(input_string)
    result2 = hash_feature(input_string)
    result3 = hash_feature(input_string)
    
    assert result1 == result2 == result3, "Hash function must be deterministic"


def test_hash_feature_bucket_range_default():
    """
    Test that the output is always within the default bucket range [0, 999].
    """
    test_inputs = [
        "user_1",
        "user_2",
        "user_999",
        "different_string",
        "another_test_case",
        "",
        "a" * 100,  # Long string
    ]
    
    for input_str in test_inputs:
        result = hash_feature(input_str)
        assert 0 <= result < 1000, f"Result {result} must be in range [0, 999] for input '{input_str}'"


def test_hash_feature_bucket_range_custom():
    """
    Test that the output respects custom bucket sizes.
    """
    test_cases = [
        ("test_string", 10),
        ("another_string", 100),
        ("yet_another", 5),
        ("input", 1),
    ]
    
    for input_str, num_buckets in test_cases:
        result = hash_feature(input_str, num_buckets=num_buckets)
        assert 0 <= result < num_buckets, \
            f"Result {result} must be in range [0, {num_buckets - 1}] for {num_buckets} buckets"


def test_hash_feature_different_inputs_different_outputs():
    """
    Test that different inputs produce (likely) different outputs.
    Note: This tests the hash distribution property.
    """
    input1 = "user_123"
    input2 = "user_456"
    input3 = "completely_different"
    
    result1 = hash_feature(input1)
    result2 = hash_feature(input2)
    result3 = hash_feature(input3)
    
    # At least two of the three should be different (very high probability)
    results = {result1, result2, result3}
    assert len(results) >= 2, "Different inputs should produce different hash values"


def test_hash_feature_empty_string():
    """
    Test that empty string is handled correctly.
    """
    result = hash_feature("")
    assert isinstance(result, int)
    assert 0 <= result < 1000


def test_hash_feature_unicode_strings():
    """
    Test that unicode strings are handled correctly.
    """
    unicode_inputs = [
        "café",
        "用户_123",
        "🚀_emoji_test",
        "ñoño",
    ]
    
    for input_str in unicode_inputs:
        result = hash_feature(input_str)
        assert 0 <= result < 1000, f"Unicode input '{input_str}' should produce valid bucket index"

