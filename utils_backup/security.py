"""
Security utilities for input validation and sanitization.
"""

import html
import re


def validate_user_input(text: str, max_length: int = 1000) -> str:
    """
    Validates and sanitizes user input to prevent security issues.

    Args:
        text: User input text to validate
        max_length: Maximum allowed length (default: 1000)

    Returns:
        Sanitized and validated text

    Raises:
        ValueError: If input is invalid or too long
    """
    if not isinstance(text, str):
        raise ValueError("Input must be a string")

    if len(text) > max_length:
        raise ValueError(f"Input too long: {len(text)} > {max_length} characters")

    # Remove potentially dangerous characters
    # Prevent SQL injection attempts
    dangerous_patterns = [
        r'[;\'"\\]',  # SQL injection characters
        r"<script[^>]*>.*?</script>",  # XSS scripts
        r"javascript:",  # JavaScript URLs
        r"data:",  # Data URLs
        r"vbscript:",  # VBScript URLs
    ]

    cleaned_text = text
    for pattern in dangerous_patterns:
        cleaned_text = re.sub(pattern, "", cleaned_text, flags=re.IGNORECASE)

    # HTML escape for safety
    cleaned_text = html.escape(cleaned_text)

    # Remove excessive whitespace
    cleaned_text = re.sub(r"\s+", " ", cleaned_text).strip()

    return cleaned_text


def validate_file_path(file_path: str) -> str:
    """
    Validates file paths to prevent directory traversal attacks.

    Args:
        file_path: File path to validate

    Returns:
        Safe file path

    Raises:
        ValueError: If path contains dangerous elements
    """
    if not isinstance(file_path, str):
        raise ValueError("File path must be a string")

    # Prevent directory traversal
    dangerous_path_patterns = [
        r"\.\.",  # Parent directory access
        r'[<>:"|?*]',  # Invalid filename characters on Windows
        r"^/",  # Absolute paths (force relative)
        r"\\",  # Backslashes (normalize to forward slash)
    ]

    for pattern in dangerous_path_patterns:
        if re.search(pattern, file_path):
            raise ValueError(f"Dangerous path pattern detected: {pattern}")

    # Normalize path
    safe_path = file_path.replace("\\", "/").strip("/")

    return safe_path


def sanitize_log_message(message: str) -> str:
    """
    Sanitizes log messages to prevent log injection attacks.

    Args:
        message: Log message to sanitize

    Returns:
        Sanitized log message
    """
    if not isinstance(message, str):
        return str(message)

    # Remove newlines and control characters that could break log format
    sanitized = re.sub(r"[\r\n\t\x00-\x1f\x7f-\x9f]", " ", message)

    # Limit length to prevent log flooding
    if len(sanitized) > 500:
        sanitized = sanitized[:497] + "..."

    return sanitized
