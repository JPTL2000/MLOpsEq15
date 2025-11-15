"""
Unit and Integration Tests for Model Functions
Tests for load_feature_names, load_model, and predict_single functions
"""
import pytest
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.api.model_extraction import load_feature_names, load_model, predict_single

# Test data paths
TEST_METADATA_PATH = Path("src/tests/test_model_metadata.pkl")
TEST_MODEL_PATH = Path("src/tests/test_online_news_model.pkl")


# -----------------------------
#  Tests
# -----------------------------

def test_load_feature_names_success():
    feature_names = load_feature_names(TEST_METADATA_PATH)
    
    assert feature_names is not None
    assert isinstance(feature_names, (list, tuple)) or hasattr(feature_names, '__iter__')
    assert len(feature_names) > 0


def test_load_feature_names_file_not_found():
    fake_path = Path("tests/nonexistent_metadata.pkl")
    
    with pytest.raises(FileNotFoundError):
        load_feature_names(fake_path)


def test_load_model_success():
    model = load_model(TEST_MODEL_PATH)
    assert model is not None
    assert hasattr(model, 'predict')


def test_load_model_file_not_found():
    fake_path = Path("tests/nonexistent_model.pkl")
    
    with pytest.raises(FileNotFoundError):
        load_model(fake_path)


def test_predict_single_returns_prediction():
    model = load_model(TEST_MODEL_PATH)
    
    feature_names = load_feature_names(TEST_METADATA_PATH)
    
    features = {name: 0 for name in feature_names}
    
    prediction = predict_single(model, features)
    
    assert prediction is not None
    assert isinstance(prediction, (int, float, np.integer, np.floating))


def test_predict_single_output_type():
    model = load_model(TEST_MODEL_PATH)
    feature_names = load_feature_names(TEST_METADATA_PATH)
    
    features = {name: 0 for name in feature_names}
    
    prediction = predict_single(model, features)
    
    assert isinstance(prediction, (int, float, np.integer, np.floating))


# -----------------------------
#  Integration tests
# -----------------------------

def test_full_pipeline_integration():
    # Load feature names from metadata
    feature_names = load_feature_names(TEST_METADATA_PATH)
    assert feature_names is not None
    assert len(feature_names) > 0
    
    # Load the model
    model = load_model(TEST_MODEL_PATH)
    assert model is not None
    assert hasattr(model, 'predict')
    
    # Create sample features
    features = {name: 0 for name in feature_names}
    
    # Make prediction
    prediction = predict_single(model, features)
    
    # Verify final prediction
    assert prediction is not None
    assert isinstance(prediction, (int, float, np.integer, np.floating))


def test_multiple_predictions():
    model = load_model(TEST_MODEL_PATH)
    feature_names = load_feature_names(TEST_METADATA_PATH)
    
    # Create two different feature sets
    features_1 = {name: 0 for name in feature_names}
    features_2 = {name: 1 for name in feature_names}
    
    # Make predictions
    prediction_1 = predict_single(model, features_1)
    prediction_2 = predict_single(model, features_2)
    
    # Verify both predictions are valid
    assert prediction_1 is not None
    assert prediction_2 is not None
    assert isinstance(prediction_1, (int, float, np.integer, np.floating))
    assert isinstance(prediction_2, (int, float, np.integer, np.floating))


def test_prediction_consistency():
    model = load_model(TEST_MODEL_PATH)
    feature_names = load_feature_names(TEST_METADATA_PATH)
    
    # Create sample features
    features = {name: 0.5 for name in feature_names}
    
    # Make two predictions with same input
    prediction_1 = predict_single(model, features)
    prediction_2 = predict_single(model, features)
    
    # Verify predictions are identical
    assert prediction_1 == prediction_2

# -----------------------------
#  Parametrized tests for robustness
# -----------------------------

@pytest.mark.parametrize("feature_value", [0, 0.5, 1, -1, 100])
def test_predict_with_different_values(feature_value):
    model = load_model(TEST_MODEL_PATH)
    feature_names = load_feature_names(TEST_METADATA_PATH)
    
    features = {name: feature_value for name in feature_names}
    prediction = predict_single(model, features)
    
    assert prediction is not None
    assert isinstance(prediction, (int, float, np.integer, np.floating))
