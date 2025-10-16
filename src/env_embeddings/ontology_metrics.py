"""Ontology-aware evaluation metrics for ENVO predictions.

This module provides hierarchical scoring functions that evaluate predictions
against true environmental triad labels using ENVO (Environment Ontology)
structure. Instead of binary exact-match scoring, partial credit is awarded
for semantically related terms based on ontological distance.

Scoring Rules:
    1. Exact match → 1.0
    2. 1 hop (parent/child) → 0.75
    3. 2 hops (grandparent/sibling) → 0.50
    4. 3+ hops → 0.25
    5. Unrelated (no path) → 0.0

Optional constraint validation ensures predictions follow ENVO guidelines:
    - env_broad_scale: must be subclass of ENVO:00000428 (biome)
    - env_local_scale: must be subclass of ENVO:01000813 (astronomical body part)
    - env_medium: must be subclass of ENVO:00010483 (environmental medium)
"""

from typing import Optional, List, Tuple
import logging

try:
    from oaklib import get_adapter  # type: ignore[import-not-found]
except ImportError:
    raise ImportError("oaklib is required. Install with: pip install oaklib")

logger = logging.getLogger(__name__)

# ENVO constraint CURIEs (parent classes for each scale)
ENVO_CONSTRAINTS = {
    "env_broad_scale": "ENVO:00000428",  # biome
    "env_local_scale": "ENVO:01000813",  # astronomical body part
    "env_medium": "ENVO:00010483",  # environmental medium
}


class ONTOLOGYDistance:
    """Utility class for computing ontological distances."""

    @staticmethod
    def exact_match(term1: str, term2: str) -> int:
        """Check if two terms are identical.

        Args:
            term1: First ENVO term (CURIE format)
            term2: Second ENVO term (CURIE format)

        Returns:
            0 if exact match, 1 if different

        Examples:
            >>> ONTOLOGYDistance.exact_match("ENVO:X", "ENVO:X")
            0
            >>> ONTOLOGYDistance.exact_match("ENVO:X", "ENVO:Y")
            1
        """
        if term1 is None or term2 is None:
            return 1
        return 0 if term1 == term2 else 1

    @staticmethod
    def calculate(
        term1: Optional[str], term2: Optional[str], ontology_adapter
    ) -> float:
        """Calculate shortest path distance between two terms in ontology.

        Uses oaklib's paths() method to find all paths between terms,
        restricted to is_a (subclass) relationships only. Returns the length
        of the shortest path.

        Args:
            term1: First ENVO term (CURIE)
            term2: Second ENVO term (CURIE)
            ontology_adapter: Initialized oaklib adapter for ENVO

        Returns:
            Shortest path distance as integer (number of edges), or float('inf') if no path

        Note:
            Only follows is_a (rdfs:subClassOf) relationships. Does NOT follow
            part_of or other relationship types.

        Examples:
            >>> # With real ENVO ontology loaded
            >>> adapter = get_adapter("sqlite:obo:envo")  # doctest: +SKIP
            >>> dist = ONTOLOGYDistance.calculate("ENVO:X", "ENVO:Y", adapter)  # doctest: +SKIP
            >>> isinstance(dist, (int, float))  # doctest: +SKIP
            True
        """
        if term1 is None or term2 is None:
            return float("inf")

        try:
            # oaklib's paths() returns all paths between two terms
            # Each path is a tuple of nodes (term1, ..., term2)
            # Restrict to only is_a (subclass) relationships for hierarchical scoring
            paths = list(ontology_adapter.paths(term1, term2, predicates=["is_a"]))
        except (AttributeError, TypeError, ValueError) as e:
            # Only catch actual oaklib query failures, not logic errors
            logger.debug(f"Error querying ontology for distance: {e}")
            return float("inf")

        # Logic operations unguarded - let logic errors propagate
        if not paths:
            return float("inf")

        # Find shortest path (minimum length)
        shortest_path = min(paths, key=len)

        # Distance is number of edges (nodes - 1)
        distance = len(shortest_path) - 1
        return distance if distance >= 0 else float("inf")


class ENVOHierarchy:
    """Load and cache ENVO ontology for hierarchical operations.

    This class provides efficient access to ENVO ontology structure,
    including distance calculations and subclass checking. Distances
    are cached to avoid repeated graph traversals.

    Attributes:
        adapter: Initialized oaklib adapter for ENVO ontology
        _distance_cache: Cache for computed distances (term1, term2) -> distance
    """

    _instance: Optional["ENVOHierarchy"] = None

    def __new__(cls):
        """Implement singleton pattern to cache ontology."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        """Initialize oaklib adapter for ENVO.

        Uses sqlite backend for local caching of ENVO ontology.
        """
        try:
            self.adapter = get_adapter("sqlite:obo:envo")
            self._distance_cache: dict = {}  # Cache for (term1, term2) -> distance
            logger.info("ENVO ontology loaded via oaklib")
        except Exception as e:
            logger.error(f"Failed to load ENVO ontology: {e}")
            raise

    def distance(self, term1: Optional[str], term2: Optional[str]) -> float:
        """Calculate shortest path distance between two ENVO terms with caching.

        Results are cached to avoid repeated graph traversals.

        Args:
            term1: First ENVO term (CURIE format, e.g., 'ENVO:00000428')
            term2: Second ENVO term (CURIE format)

        Returns:
            Distance as integer, or float('inf') if no path exists

        Examples:
            >>> hierarchy = ENVOHierarchy()  # doctest: +SKIP
            >>> hierarchy.distance("ENVO:X", "ENVO:X")  # doctest: +SKIP
            0
        """
        # Normalize cache key (order doesn't matter for undirected distance)
        if term1 is None or term2 is None:
            return float("inf")

        cache_key = tuple(sorted([term1, term2]))

        # Check cache first
        if cache_key in self._distance_cache:
            return self._distance_cache[cache_key]

        # Compute and cache
        distance = ONTOLOGYDistance.calculate(term1, term2, self.adapter)
        self._distance_cache[cache_key] = distance

        return distance

    def is_subclass_of(self, term: Optional[str], parent: Optional[str]) -> bool:
        """Check if term is a subclass of parent in ENVO.

        Args:
            term: Term to check (CURIE)
            parent: Parent term (CURIE)

        Returns:
            True if term is subclass of parent, False otherwise

        Examples:
            >>> hierarchy = ENVOHierarchy()  # doctest: +SKIP
            >>> # marine biome is subclass of biome
            >>> hierarchy.is_subclass_of("ENVO:00000446", "ENVO:00000428")  # doctest: +SKIP
            True
        """
        if term is None or parent is None:
            return False

        if term == parent:
            return True

        try:
            # Use oaklib's ancestor checking
            ancestors = self.adapter.ancestors(term, reflexive=True)
            return parent in ancestors
        except (AttributeError, TypeError, ValueError) as e:
            # Only catch actual oaklib query failures, not logic errors
            logger.debug(f"Error checking subclass in ontology: {e}")
            return False


def hierarchical_score(
    predicted: Optional[str],
    true: Optional[str],
    ontology: ENVOHierarchy,
) -> float:
    """Score a prediction using hierarchical ontology distance.

    Scoring rules:
        1.0 - Exact match
        0.75 - 1 hop (parent/child)
        0.50 - 2 hops (grandparent/sibling)
        0.25 - 3+ hops (distant relative)
        0.0 - Unrelated or None

    Args:
        predicted: Predicted ENVO term (or None)
        true: True ENVO term (or None)
        ontology: ENVOHierarchy instance

    Returns:
        Score between 0.0 and 1.0

    Examples:
        >>> ontology = ENVOHierarchy()  # doctest: +SKIP
        >>> hierarchical_score("ENVO:X", "ENVO:X", ontology)  # doctest: +SKIP
        1.0
    """
    if predicted is None or true is None:
        return 0.0

    if predicted == true:
        return 1.0

    distance = ontology.distance(predicted, true)

    # Check for infinity first, as float('inf') >= 3 is True
    if distance == float("inf"):
        return 0.0
    elif distance == 0:
        return 1.0
    elif distance == 1:
        return 0.75
    elif distance == 2:
        return 0.50
    elif distance >= 3:
        return 0.25
    else:
        return 0.0


def is_valid_term_for_scale(
    term: Optional[str],
    scale: str,
    ontology: ENVOHierarchy,
) -> bool:
    """Check if term satisfies constraint for ENVO scale.

    Optional validation ensures terms follow ENVO guidelines.

    Args:
        term: ENVO term to validate (or None)
        scale: ENVO scale name ('env_broad_scale', 'env_local_scale', 'env_medium')
        ontology: ENVOHierarchy instance

    Returns:
        True if term satisfies constraint, False otherwise

    Examples:
        >>> ontology = ENVOHierarchy()  # doctest: +SKIP
        >>> # marine biome should be valid for broad_scale
        >>> is_valid_term_for_scale("ENVO:00000446", "env_broad_scale", ontology)  # doctest: +SKIP
        True
    """
    if term is None:
        return False

    if scale not in ENVO_CONSTRAINTS:
        logger.warning(f"Unknown scale: {scale}")
        return False

    parent = ENVO_CONSTRAINTS[scale]
    return ontology.is_subclass_of(term, parent)


def hierarchical_accuracy(
    predictions: List[Optional[str]],
    true_labels: List[Optional[str]],
    ontology: ENVOHierarchy,
) -> float:
    """Compute mean hierarchical accuracy across predictions.

    Args:
        predictions: List of predicted ENVO terms
        true_labels: List of true ENVO terms
        ontology: ENVOHierarchy instance

    Returns:
        Mean hierarchical score (0.0 to 1.0)

    Raises:
        ValueError: If lengths don't match or lists are empty

    Examples:
        >>> ontology = ENVOHierarchy()  # doctest: +SKIP
        >>> predictions = ["ENVO:A", "ENVO:B", "ENVO:C"]  # doctest: +SKIP
        >>> true_labels = ["ENVO:A", "ENVO:B", "ENVO:D"]  # doctest: +SKIP
        >>> acc = hierarchical_accuracy(predictions, true_labels, ontology)  # doctest: +SKIP
        >>> 0.0 <= acc <= 1.0  # doctest: +SKIP
        True
    """
    if len(predictions) != len(true_labels):
        raise ValueError(
            f"Length mismatch: {len(predictions)} predictions "
            f"vs {len(true_labels)} labels"
        )

    if len(predictions) == 0:
        raise ValueError("Cannot compute accuracy on empty predictions")

    scores = [
        hierarchical_score(pred, true, ontology)
        for pred, true in zip(predictions, true_labels)
    ]

    return sum(scores) / len(scores)


def constrained_hierarchical_accuracy(
    predictions: List[Optional[str]],
    true_labels: List[Optional[str]],
    scale: str,
    ontology: ENVOHierarchy,
) -> Tuple[float, float]:
    """Compute hierarchical accuracy with constraint validation.

    Returns both hierarchical accuracy and percentage of predictions
    satisfying the ontology constraint for the scale.

    Args:
        predictions: List of predicted ENVO terms
        true_labels: List of true ENVO terms
        scale: ENVO scale name
        ontology: ENVOHierarchy instance

    Returns:
        Tuple of (hierarchical_accuracy, constraint_satisfaction_rate)

    Examples:
        >>> ontology = ENVOHierarchy()  # doctest: +SKIP
        >>> predictions = ["ENVO:A", "ENVO:B"]  # doctest: +SKIP
        >>> true_labels = ["ENVO:A", "ENVO:C"]  # doctest: +SKIP
        >>> acc, constraint = constrained_hierarchical_accuracy(  # doctest: +SKIP
        ...     predictions, true_labels, "env_broad_scale", ontology
        ... )
        >>> 0.0 <= acc <= 1.0 and 0.0 <= constraint <= 1.0  # doctest: +SKIP
        True
    """
    acc = hierarchical_accuracy(predictions, true_labels, ontology)

    # Check constraint satisfaction
    valid_count = sum(
        1 for pred in predictions if is_valid_term_for_scale(pred, scale, ontology)
    )
    constraint_rate = valid_count / len(predictions)

    return acc, constraint_rate
