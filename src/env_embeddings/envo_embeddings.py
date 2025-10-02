"""ENVO term embedding functionality using OLS."""

import re
from pathlib import Path
from typing import List, Optional

from diskcache import Cache
from ols_client import EBIClient

# Disk cache using diskcache library
_CACHE_DIR = Path.home() / ".cache" / "env-embeddings" / "envo"
_cache = Cache(str(_CACHE_DIR))


def extract_first_envo_term(envo_text: Optional[str]) -> Optional[str]:
    """Extract the first ENVO term from a string containing multiple ENVO terms.

    Args:
        envo_text: String containing ENVO terms like "ENVO:00000428 | ENVO:01001055 | ENVO_00002003"

    Returns:
        First ENVO term in standardized format (ENVO:XXXXXXX) or None if no valid term found

    Examples:
        >>> extract_first_envo_term("ENVO:00000428 | ENVO:01001055 | ENVO_00002003")
        'ENVO:00000428'
        >>> extract_first_envo_term("polar biome [ENVO:01000339] | island [ENVO_00000098]")
        'ENVO:01000339'
        >>> extract_first_envo_term("ENVO_00000114 | ENVO_00000115")
        'ENVO:00000114'
    """
    if not envo_text or not isinstance(envo_text, str):
        return None

    # Pattern to match ENVO terms in both formats: ENVO:XXXXXXX or ENVO_XXXXXXX
    pattern = r"ENVO[_:](\d{8})"

    match = re.search(pattern, envo_text)
    if match:
        # Return in standardized colon format
        term_id = match.group(1)
        return f"ENVO:{term_id}"

    return None


def get_envo_embedding(envo_term: str, use_cache: bool = True) -> Optional[List[float]]:
    """Get OLS embedding for an ENVO term with disk-backed caching.

    Uses diskcache for automatic persistent caching to SQLite.

    Args:
        envo_term: ENVO term in format "ENVO:XXXXXXX"
        use_cache: Whether to use cached embeddings (default: True)

    Returns:
        List of floats representing the embedding, or None if failed

    Examples:
        >>> embedding = get_envo_embedding("ENVO:00000428")
        >>> len(embedding) if embedding else 0
        1536
    """
    if not envo_term or not isinstance(envo_term, str):
        return None

    # Validate ENVO term format
    if not re.match(r"^ENVO:\d{8}$", envo_term):
        return None

    # Check cache first
    if use_cache and envo_term in _cache:
        return _cache[envo_term]

    try:
        client = EBIClient()

        # Convert ENVO:XXXXXXX to IRI format expected by OLS
        term_number = envo_term.split(":")[1]
        iri = f"http://purl.obolibrary.org/obo/ENVO_{term_number}"

        # Get embedding from OLS
        embedding = client.get_embedding("envo", iri)

        # Cache the result (diskcache automatically persists to disk)
        if use_cache:
            _cache[envo_term] = embedding

        return embedding

    except Exception as e:
        print(f"Error getting embedding for {envo_term}: {e}")
        # Cache the failure
        if use_cache:
            _cache[envo_term] = None
        return None


def get_envo_embedding_from_text(envo_text: str) -> Optional[List[float]]:
    """Extract first ENVO term from text and get its embedding.

    Args:
        envo_text: String containing ENVO terms

    Returns:
        List of floats representing the embedding of the first ENVO term, or None if failed

    Examples:
        >>> embedding = get_envo_embedding_from_text("ENVO:00000428 | ENVO:01001055")
        >>> len(embedding) if embedding else 0
        1536
    """
    envo_term = extract_first_envo_term(envo_text)
    if envo_term is None:
        return None

    return get_envo_embedding(envo_term)


def get_cache_stats() -> dict:
    """Get statistics about the ENVO embedding cache.

    Returns:
        Dictionary with cache statistics
    """
    total = len(_cache)
    successful = sum(1 for k in _cache if _cache[k] is not None)
    failed = total - successful

    return {
        "total_cached": total,
        "successful": successful,
        "failed": failed,
        "cache_dir": str(_CACHE_DIR),
    }


def clear_cache() -> None:
    """Clear the ENVO embedding cache."""
    _cache.clear()
    print(f"Cleared ENVO cache at {_CACHE_DIR}")


if __name__ == "__main__":
    # Test ENVO term extraction
    test_texts = [
        "ENVO:00000428 | ENVO:01001055 | ENVO_00002003",
        "polar biome [ENVO:01000339] | island [ENVO_00000098]|coast [ENVO_01000687] | soil [ENVO_00001998]",
        "ENVO:00000114 | ENVO_00000115 | ENVO_00000115",
        "no envo terms here",
        "",
        None,
    ]

    for text in test_texts:
        result = extract_first_envo_term(text)
        print(f"'{text}' -> '{result}'")
