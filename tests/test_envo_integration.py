"""Integration tests with real ENVO ontology data.

These tests verify that the hierarchical scoring logic works correctly
with actual ENVO terms from the ontology.
"""

import pytest
from env_embeddings.ontology_metrics import ENVOHierarchy, hierarchical_score


class TestRealENVODistances:
    """Test distance calculations with real ENVO terms."""

    def test_marine_biome_and_forest_biome_distance(self):
        """Test distance calculation between marine biome and forest biome."""
        hierarchy = ENVOHierarchy()

        # ENVO:00000446 = marine biome
        # ENVO:01000174 = forest biome
        distance = hierarchy.distance("ENVO:00000446", "ENVO:01000174")

        print(f"\nDistance between marine biome and forest biome: {distance}")

        # Verify it's a reasonable distance (not infinity, not 0)
        assert 0 < distance < 10, f"Expected distance between 0 and 10, got {distance}"

    def test_marine_biome_to_biome_parent(self):
        """Marine biome to biome parent should be 1 hop."""
        hierarchy = ENVOHierarchy()

        # ENVO:00000446 = marine biome
        # ENVO:00000428 = biome (parent)
        distance = hierarchy.distance("ENVO:00000446", "ENVO:00000428")

        print(f"\nDistance from marine biome to biome: {distance}")

        assert distance == 1.0, f"Expected distance=1.0, got {distance}"

    def test_hierarchical_score_with_real_terms_exact_match(self):
        """Test hierarchical scoring with exact match."""
        hierarchy = ENVOHierarchy()

        score = hierarchical_score("ENVO:00000446", "ENVO:00000446", hierarchy)
        assert score == 1.0

    def test_hierarchical_score_with_real_terms_1_hop(self):
        """Test hierarchical scoring with 1 hop distance."""
        hierarchy = ENVOHierarchy()

        # Marine biome to biome = 1 hop
        score = hierarchical_score("ENVO:00000446", "ENVO:00000428", hierarchy)
        print(f"\nScore for 1-hop (marine biome -> biome): {score}")
        assert score == 0.75, f"Expected score=0.75, got {score}"

    def test_hierarchical_score_with_real_terms_close_distance(self):
        """Test hierarchical scoring with close distance."""
        hierarchy = ENVOHierarchy()

        # Marine biome to forest biome - close terms
        score = hierarchical_score("ENVO:00000446", "ENVO:01000174", hierarchy)
        print(f"\nScore for close terms (marine biome <-> forest biome): {score}")
        # Should give partial credit (between 0.25 and 1.0)
        assert 0.25 <= score < 1.0, f"Expected score between 0.25 and 1.0, got {score}"

    def test_debug_ancestors(self):
        """Debug test to inspect ancestor paths."""
        hierarchy = ENVOHierarchy()

        # Check ancestors of marine biome
        marine = "ENVO:00000446"
        ancestors = list(
            hierarchy.adapter.ancestors(
                marine, reflexive=True, predicates=["rdfs:subClassOf"]
            )
        )

        print(f"\nAncestors of {marine} (marine biome) with rdfs:subClassOf:")
        for i, ancestor in enumerate(ancestors[:10]):  # Show first 10
            print(f"  {i}: {ancestor}")

        # Try without predicates filter
        ancestors_all = list(hierarchy.adapter.ancestors(marine, reflexive=True))
        print(f"\nAncestors of {marine} (marine biome) without predicate filter:")
        for i, ancestor in enumerate(ancestors_all[:10]):
            print(f"  {i}: {ancestor}")

        # Check if biome is in there
        biome = "ENVO:00000428"
        print(f"\nIs {biome} (biome) in ancestors? {biome in ancestors_all}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
