"""Tests for ontology-aware ENVO prediction metrics.

Tests cover hierarchical scoring, constraint validation, and accuracy metrics
using ENVO (Environment Ontology) for evaluating environmental triad predictions.
"""

import pytest
from unittest.mock import Mock

from env_embeddings.ontology_metrics import (
    ONTOLOGYDistance,
    hierarchical_score,
    is_valid_term_for_scale,
    hierarchical_accuracy,
    ENVOHierarchy,
)


class TestOntologyDistance:
    """Test ontology distance calculation."""

    def test_distance_exact_match(self):
        """Exact match should have distance 0."""
        assert ONTOLOGYDistance.exact_match("ENVO:X", "ENVO:X") == 0

    def test_distance_different_terms(self):
        """Different terms should have non-zero distance."""
        assert ONTOLOGYDistance.exact_match("ENVO:X", "ENVO:Y") > 0

    def test_distance_with_mock_ontology(self):
        """Test distance calculation with mock ontology adapter."""
        # Mock oaklib adapter
        mock_adapter = Mock()
        # shortest_path returns list of nodes including start and end
        mock_adapter.shortest_path = Mock(return_value=["ENVO:child", "ENVO:parent"])

        distance = ONTOLOGYDistance.calculate("ENVO:child", "ENVO:parent", mock_adapter)
        assert distance == 1  # 1 hop

    def test_distance_unrelated_terms(self):
        """Unrelated terms should return infinity/max distance."""
        mock_adapter = Mock()
        # No path between unrelated terms
        mock_adapter.shortest_path = Mock(return_value=None)

        distance = ONTOLOGYDistance.calculate("ENVO:A", "ENVO:X", mock_adapter)
        assert distance == float("inf")


class TestHierarchicalScore:
    """Test hierarchical scoring logic."""

    def test_score_exact_match(self):
        """Exact match should score 1.0."""
        mock_ontology = Mock()
        score = hierarchical_score(
            predicted="ENVO:X",
            true="ENVO:X",
            ontology=mock_ontology,
        )
        assert score == 1.0

    def test_score_parent_child_distance_1(self):
        """1 hop distance (parent/child) should score 0.75."""
        mock_ontology = Mock()
        mock_ontology.distance = Mock(return_value=1)

        score = hierarchical_score(
            predicted="ENVO:child",
            true="ENVO:parent",
            ontology=mock_ontology,
        )
        assert score == 0.75

    def test_score_distance_2_hops(self):
        """2 hops (grandparent/sibling) should score 0.50."""
        mock_ontology = Mock()
        mock_ontology.distance = Mock(return_value=2)

        score = hierarchical_score(
            predicted="ENVO:grandchild",
            true="ENVO:grandparent",
            ontology=mock_ontology,
        )
        assert score == 0.50

    def test_score_distance_3_plus_hops(self):
        """3+ hops should score 0.25."""
        mock_ontology = Mock()
        mock_ontology.distance = Mock(return_value=3)

        score = hierarchical_score(
            predicted="ENVO:distant",
            true="ENVO:origin",
            ontology=mock_ontology,
        )
        assert score == 0.25

    def test_score_unrelated_terms(self):
        """Unrelated terms (no path) should score 0.0."""
        mock_ontology = Mock()
        mock_ontology.distance = Mock(return_value=float("inf"))

        score = hierarchical_score(
            predicted="ENVO:unrelated",
            true="ENVO:origin",
            ontology=mock_ontology,
        )
        assert score == 0.0

    def test_score_none_values(self):
        """None or missing values should score 0.0."""
        mock_ontology = Mock()

        assert hierarchical_score(None, "ENVO:X", mock_ontology) == 0.0
        assert hierarchical_score("ENVO:X", None, mock_ontology) == 0.0


class TestConstraintValidation:
    """Test optional constraint validation."""

    def test_is_valid_broad_scale(self):
        """Test validation for env_broad_scale (must be subclass of biome)."""
        mock_ontology = Mock()
        mock_ontology.is_subclass_of = Mock(return_value=True)

        # Valid term (marine biome is subclass of biome)
        result = is_valid_term_for_scale(
            "ENVO:00000446",  # marine biome
            scale="env_broad_scale",
            ontology=mock_ontology,
        )
        assert result is True

    def test_is_invalid_broad_scale(self):
        """Test invalid term for env_broad_scale."""
        mock_ontology = Mock()
        mock_ontology.is_subclass_of = Mock(return_value=False)

        # Invalid term (soil is not subclass of biome)
        result = is_valid_term_for_scale(
            "ENVO:00001998",
            scale="env_broad_scale",
            ontology=mock_ontology,
        )
        assert result is False

    def test_is_valid_local_scale(self):
        """Test validation for env_local_scale."""
        mock_ontology = Mock()
        mock_ontology.is_subclass_of = Mock(return_value=True)

        result = is_valid_term_for_scale(
            "ENVO:00000033",  # coastal zone
            scale="env_local_scale",
            ontology=mock_ontology,
        )
        assert result is True

    def test_is_valid_medium(self):
        """Test validation for env_medium."""
        mock_ontology = Mock()
        mock_ontology.is_subclass_of = Mock(return_value=True)

        result = is_valid_term_for_scale(
            "ENVO:00001998",  # soil
            scale="env_medium",
            ontology=mock_ontology,
        )
        assert result is True

    def test_constraint_validation_with_none(self):
        """Constraint validation with None should return False."""
        mock_ontology = Mock()

        result = is_valid_term_for_scale(
            None, scale="env_broad_scale", ontology=mock_ontology
        )
        assert result is False


class TestHierarchicalAccuracy:
    """Test hierarchical accuracy metric computation."""

    def test_hierarchical_accuracy_all_exact(self):
        """All exact matches should give 1.0 accuracy."""
        mock_ontology = Mock()
        mock_ontology.distance = Mock(return_value=0)

        predictions = ["ENVO:A", "ENVO:B", "ENVO:C"]
        true_labels = ["ENVO:A", "ENVO:B", "ENVO:C"]

        accuracy = hierarchical_accuracy(predictions, true_labels, mock_ontology)
        assert accuracy == 1.0

    def test_hierarchical_accuracy_all_wrong(self):
        """All unrelated terms should give 0.0 accuracy."""
        mock_ontology = Mock()
        mock_ontology.distance = Mock(return_value=float("inf"))

        predictions = ["ENVO:X", "ENVO:Y", "ENVO:Z"]
        true_labels = ["ENVO:A", "ENVO:B", "ENVO:C"]

        accuracy = hierarchical_accuracy(predictions, true_labels, mock_ontology)
        assert accuracy == 0.0

    def test_hierarchical_accuracy_mixed(self):
        """Mixed predictions should give average score."""
        mock_ontology = Mock()

        # Set up distances: 1 hop (parent), inf (unrelated), 2 hops (sibling)
        distances = [1, float("inf"), 2]
        mock_ontology.distance = Mock(side_effect=distances)

        # All pairs are different, so all call distance()
        predictions = ["ENVO:X", "ENVO:Y", "ENVO:Z"]
        true_labels = ["ENVO:A", "ENVO:B", "ENVO:C"]

        # Scores: 0.75 (1 hop), 0.0 (inf), 0.50 (2 hops) → mean = 0.416...
        accuracy = hierarchical_accuracy(predictions, true_labels, mock_ontology)
        expected = (0.75 + 0.0 + 0.50) / 3
        assert abs(accuracy - expected) < 0.01

    def test_hierarchical_accuracy_empty(self):
        """Empty predictions should raise or return 0."""
        mock_ontology = Mock()

        predictions = []
        true_labels = []

        with pytest.raises((ValueError, ZeroDivisionError)):
            hierarchical_accuracy(predictions, true_labels, mock_ontology)

    def test_hierarchical_accuracy_length_mismatch(self):
        """Mismatched lengths should raise ValueError."""
        mock_ontology = Mock()

        predictions = ["ENVO:A", "ENVO:B"]
        true_labels = ["ENVO:A"]

        with pytest.raises(ValueError):
            hierarchical_accuracy(predictions, true_labels, mock_ontology)


class TestENVOHierarchy:
    """Test ENVOHierarchy class."""

    def test_load_envo_ontology(self):
        """Test loading ENVO ontology."""
        hierarchy = ENVOHierarchy()
        # Should successfully initialize without errors
        assert hierarchy is not None

    def test_envo_hierarchy_has_methods(self):
        """Test that ENVOHierarchy has required methods."""
        hierarchy = ENVOHierarchy()

        assert hasattr(hierarchy, "distance")
        assert hasattr(hierarchy, "is_subclass_of")
        assert callable(hierarchy.distance)
        assert callable(hierarchy.is_subclass_of)

    def test_distance_method_returns_int_or_inf(self):
        """Distance method should return int or infinity."""
        hierarchy = ENVOHierarchy()

        # These should not raise errors
        result = hierarchy.distance("ENVO:00000428", "ENVO:00000428")
        assert isinstance(result, (int, float))
        assert result >= 0 or result == float("inf")

    def test_is_subclass_of_same_term(self):
        """A term should be considered a subclass of itself."""
        hierarchy = ENVOHierarchy()

        result = hierarchy.is_subclass_of("ENVO:00000428", "ENVO:00000428")
        assert result is True

    def test_ontology_cache(self):
        """ENVOHierarchy should cache ontology after loading (singleton)."""
        hierarchy1 = ENVOHierarchy()
        hierarchy2 = ENVOHierarchy()

        # Should be the same instance (singleton pattern)
        assert hierarchy1 is hierarchy2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
