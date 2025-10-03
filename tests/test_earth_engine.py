"""Tests for Earth Engine functionality."""

import pytest
from unittest.mock import Mock, patch
from env_embeddings.earth_engine import initialize_ee, get_embedding


class TestInitializeEE:
    """Test Earth Engine initialization."""

    @patch("env_embeddings.earth_engine.ee")
    def test_initialize_ee(self, mock_ee):
        """Test that initialize_ee calls authentication and initialization."""
        initialize_ee()

        mock_ee.Authenticate.assert_called_once()
        mock_ee.Initialize.assert_called_once()


class TestGetEmbedding:
    """Test getting embeddings from Earth Engine."""

    @patch("env_embeddings.earth_engine.ee")
    def test_get_embedding_success(self, mock_ee):
        """Test successful embedding retrieval."""
        # Mock the Earth Engine components
        mock_point = Mock()
        mock_ee.Geometry.Point.return_value = mock_point

        mock_collection = Mock()
        mock_ee.ImageCollection.return_value = mock_collection

        mock_filtered = Mock()
        mock_collection.filterDate.return_value = mock_filtered

        mock_bounds_filtered = Mock()
        mock_filtered.filterBounds.return_value = mock_bounds_filtered

        # Mock size() to return 1 (collection has images)
        mock_size = Mock()
        mock_size.getInfo.return_value = 1
        mock_bounds_filtered.size.return_value = mock_size

        mock_image = Mock()
        mock_bounds_filtered.first.return_value = mock_image

        mock_sampled = Mock()
        mock_image.sample.return_value = mock_sampled

        mock_feature = Mock()
        mock_sampled.first.return_value = mock_feature

        mock_dict = Mock()
        mock_feature.toDictionary.return_value = mock_dict

        # Create test embedding data
        test_embedding = {f"A{str(i).zfill(2)}": float(i) for i in range(64)}
        mock_dict.getInfo.return_value = test_embedding

        # Test the function (disable cache for testing)
        result = get_embedding(39.0372, -121.8036, 2024, use_cache=False)

        # Verify the result
        assert len(result) == 64
        assert result == [float(i) for i in range(64)]

        # Verify the calls
        mock_ee.Geometry.Point.assert_called_once_with(-121.8036, 39.0372)
        mock_ee.ImageCollection.assert_called_once_with(
            "GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL"
        )
        mock_collection.filterDate.assert_called_once_with("2024-01-01", "2025-01-01")
        mock_filtered.filterBounds.assert_called_once_with(mock_point)
        mock_image.sample.assert_called_once_with(region=mock_point, scale=10)

    @patch("env_embeddings.earth_engine.ee")
    def test_get_embedding_no_image_found(self, mock_ee):
        """Test error when no image is found for the given parameters."""
        # Mock the Earth Engine components to return None for first()
        mock_point = Mock()
        mock_ee.Geometry.Point.return_value = mock_point

        mock_collection = Mock()
        mock_ee.ImageCollection.return_value = mock_collection

        mock_filtered = Mock()
        mock_collection.filterDate.return_value = mock_filtered

        mock_bounds_filtered = Mock()
        mock_filtered.filterBounds.return_value = mock_bounds_filtered

        # Mock size() to return 0 (no images in collection)
        mock_size = Mock()
        mock_size.getInfo.return_value = 0
        mock_bounds_filtered.size.return_value = mock_size

        # Test that ValueError is raised (disable cache for testing)
        with pytest.raises(
            ValueError, match="No embedding found for 39.0372,-121.8036 at year 2024"
        ):
            get_embedding(39.0372, -121.8036, 2024, use_cache=False)

    @patch("env_embeddings.earth_engine.ee")
    def test_get_embedding_with_different_coordinates(self, mock_ee):
        """Test embedding retrieval with different coordinates."""
        # Setup mocks similar to success test
        mock_point = Mock()
        mock_ee.Geometry.Point.return_value = mock_point

        mock_collection = Mock()
        mock_ee.ImageCollection.return_value = mock_collection

        mock_filtered = Mock()
        mock_collection.filterDate.return_value = mock_filtered

        mock_bounds_filtered = Mock()
        mock_filtered.filterBounds.return_value = mock_bounds_filtered

        # Mock size() to return 1 (collection has images)
        mock_size = Mock()
        mock_size.getInfo.return_value = 1
        mock_bounds_filtered.size.return_value = mock_size

        mock_image = Mock()
        mock_bounds_filtered.first.return_value = mock_image

        mock_sampled = Mock()
        mock_image.sample.return_value = mock_sampled

        mock_feature = Mock()
        mock_sampled.first.return_value = mock_feature

        mock_dict = Mock()
        mock_feature.toDictionary.return_value = mock_dict

        # Create test data with different values
        test_embedding = {f"A{str(i).zfill(2)}": float(i * 2) for i in range(64)}
        mock_dict.getInfo.return_value = test_embedding

        # Test with different coordinates and year (disable cache for testing)
        result = get_embedding(40.0, -120.0, 2023, use_cache=False)

        # Verify the result
        assert len(result) == 64
        assert result == [float(i * 2) for i in range(64)]

        # Verify correct coordinates were used
        mock_ee.Geometry.Point.assert_called_once_with(-120.0, 40.0)
        mock_collection.filterDate.assert_called_once_with("2023-01-01", "2024-01-01")
