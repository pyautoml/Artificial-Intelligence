import re
import random
from typing import List
from functools import lru_cache


REGEX_DOMAIN: re.compile = re.compile(r'https?://([^/]+)')
SPECIAL_CHARS: List[str] = [
    ",", ".", "'", '"', 
    "!", "#", "$", "%", 
    "&", "*", "+", "-", 
    "/", "=", "?", "^", 
    "_", "`", "{", "|", 
    "}", "~"
]

HEADERS: List[dict] = [
    {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"},
    {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"},
    {"User-Agent": "Mozilla/5.0 (Linux; Android 10; SM-G973F) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Mobile Safari/537.36"},
    {"User-Agent": "Mozilla/5.0 (iPhone; CPU iPhone OS 14_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0 Mobile/15E148 Safari/604.1"},
]


@lru_cache(maxsize=1000)
def extract_domain(url: str) -> str:
    """Extract domain url from url."""
    match = REGEX_DOMAIN.search(url)
    return f"https://{match.group(1)}" if match else ""

@lru_cache(maxsize=200)
def remove_special_chars(text: str) -> str:
    """Remove unwanted character from a string"""
    try:
        for char in SPECIAL_CHARS:
            text = text.replace(char, "")
        return text
    except Exception as e:
        raise e
    
@lru_cache(maxsize=300)
def get_random_headers() -> dict:
   """Provide random headers for http/https requests."""
   return random.choice(HEADERS)
