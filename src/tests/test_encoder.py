import pytest
import pandas as pd
import numpy as np
from src.preprocessing.encoder import Encoders


@pytest.fixture
def sample_df():
    """Create a sample DataFrame for testing"""
    return pd.DataFrame({
        'cat_1': ['A', 'B', 'A', 'C', 'B', 'A', 'C', 'B', 'A', 'C', 'B', 'A'],
        'cat_2': ['X', 'Y', 'X', 'Z', 'Y', 'Z', 'X', 'Y', 'X', 'Z', 'Y', 'Z'],
        'cat_3': ['High', 'Low', 'Medium', 'High', 'Low', 'Medium', 'High', 'Low', 'Medium', 'High', 'Low', 'Medium'],
        'numeric': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        'target': [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
    })


def test_label_encoder(sample_df):
    """Test label encoding for a single column"""
    encoder_dict = {'label': ['cat_1']}
    encoder = Encoders(sample_df, encoder_dict)

    # Test fit_transform
    transformed_df = encoder.fit_transform()
    assert 'cat_1' in transformed_df.columns
    assert transformed_df['cat_1'].dtype == np.int64
    assert len(transformed_df['cat_1'].unique()) == 3

    # Test transform on new data
    new_data = pd.DataFrame({'cat_1': ['A', 'B', 'C']})
    transformed_new = encoder.transform(new_data)
    assert transformed_new['cat_1'].dtype == np.int64


def test_onehot_encoder(sample_df):
    """Test one-hot encoding for a single column"""
    encoder_dict = {'one_hot': ['cat_2']}
    encoder = Encoders(sample_df, encoder_dict)

    transformed_df = encoder.fit_transform()
    expected_columns = ['cat_2_X', 'cat_2_Y', 'cat_2_Z']

    # Check if one-hot encoded columns exist
    assert all(col in transformed_df.columns for col in expected_columns)
    # Check if original column is dropped
    assert 'cat_2' not in transformed_df.columns
    # Check if values are binary
    assert transformed_df[expected_columns].isin([0, 1]).all().all()


def test_target_encoder(sample_df):
    """Test target encoding for a single column"""
    encoder_dict = {'target': ['cat_1']}
    encoder = Encoders(sample_df, encoder_dict)

    transformed_df = encoder.fit_transform(target_column='target')
    assert 'cat_1' in transformed_df.columns
    assert transformed_df['cat_1'].dtype == np.float64

    # Test transform on new data
    new_data = pd.DataFrame({'cat_1': ['A', 'B', 'C']})
    transformed_new = encoder.transform(new_data)
    assert transformed_new['cat_1'].dtype == np.float64


def test_ordinal_encoder(sample_df):
    """Test ordinal encoding for a single column"""
    encoder_dict = {'ordinal': ['cat_3']}
    encoder = Encoders(sample_df, encoder_dict)

    transformed_df = encoder.fit_transform()
    assert 'cat_3' in transformed_df.columns
    assert transformed_df['cat_3'].dtype == np.float64
    assert len(transformed_df['cat_3'].unique()) == 3


def test_binary_encoder(sample_df):
    """Test binary encoding for a single column"""
    encoder_dict = {'binary': ['cat_1']}
    encoder = Encoders(sample_df, encoder_dict)

    transformed_df = encoder.fit_transform()
    # Binary encoder typically creates multiple binary columns
    assert 'cat_1' not in transformed_df.columns
    # Check if any binary encoded columns exist
    assert any('cat_1_' in col for col in transformed_df.columns)


def test_frequency_encoder(sample_df):
    """Test frequency encoding for a single column"""
    encoder_dict = {'frequency': ['cat_2']}
    encoder = Encoders(sample_df, encoder_dict)

    transformed_df = encoder.fit_transform()
    assert 'cat_2' in transformed_df.columns
    assert transformed_df['cat_2'].dtype == np.float64


def test_multiple_encoders(sample_df):
    """Test multiple encoders together"""
    encoder_dict = {
        'label': ['cat_1'],
        'one_hot': ['cat_2'],
        'ordinal': ['cat_3']
    }
    encoder = Encoders(sample_df, encoder_dict)

    transformed_df = encoder.fit_transform()

    # Check label encoding
    assert 'cat_1' in transformed_df.columns
    assert transformed_df['cat_1'].dtype == np.int64

    # Check one-hot encoding
    assert 'cat_2' not in transformed_df.columns
    assert all(col in transformed_df.columns for col in ['cat_2_X', 'cat_2_Y', 'cat_2_Z'])

    # Check ordinal encoding
    assert 'cat_3' in transformed_df.columns
    assert transformed_df['cat_3'].dtype == np.float64
