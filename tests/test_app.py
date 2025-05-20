import pytest
import pandas as pd
import numpy as np
from src.model import make_features, load_clean_data

def test_make_features():
    # Create sample input data
    data = {
        'neighbourhood_group': ['Manhattan', 'Brooklyn'],
        'room_type': ['Entire home/apt', 'Private room'],
        'minimum_nights': [1, 2],
        'availability_365': [365, 180],
        'latitude': [40.7128, 40.6782],
        'longitude': [-74.0060, -73.9442],
        'price': [100, 50]
    }
    df = pd.DataFrame(data)
    
    # Test feature creation
    X, y = make_features(df)
    
    # Check that features are created correctly
    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, pd.Series)
    assert len(X) == len(df)
    assert len(y) == len(df)
    
    # Check that all expected columns are present
    expected_columns = [
        'neighbourhood_Manhattan', 'neighbourhood_Brooklyn',
        'room_Entire home/apt', 'room_Private room',
        'min_nights_1', 'min_nights_2_6', 'min_nights_7plus',
        'availability_365', 'latitude', 'longitude'
    ]
    for col in expected_columns:
        assert col in X.columns

def test_make_features_without_price():
    # Create sample input data without price
    data = {
        'neighbourhood_group': ['Manhattan'],
        'room_type': ['Entire home/apt'],
        'minimum_nights': [1],
        'availability_365': [365],
        'latitude': [40.7128],
        'longitude': [-74.0060]
    }
    df = pd.DataFrame(data)
    
    # Test feature creation without price
    X, y = make_features(df)
    
    # Check that features are created correctly
    assert isinstance(X, pd.DataFrame)
    assert y is None
    assert len(X) == len(df)
    
    # Check that all expected columns are present
    expected_columns = [
        'neighbourhood_Manhattan',
        'room_Entire home/apt',
        'min_nights_1', 'min_nights_2_6', 'min_nights_7plus',
        'availability_365', 'latitude', 'longitude'
    ]
    for col in expected_columns:
        assert col in X.columns 