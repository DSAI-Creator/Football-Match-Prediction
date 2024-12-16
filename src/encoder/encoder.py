import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.base import BaseEstimator, TransformerMixin

class DataEncoding(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.label_encoders = {}
        self.one_hot_encoder = None
        self.target_mean_map = {}
        self.ordinal_maps = {}
        self.binary_encoders = {}

    def label_encode(self, df, columns):
        """
        Apply label encoding to specified columns.
        :param df: Input DataFrame
        :param columns: List of column names to label encode
        :return: DataFrame with label-encoded columns
        """
        for col in columns:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
            df[col] = self.label_encoders[col].fit_transform(df[col])
        return df

    def one_hot_encode(self, df, columns):
        """
        Apply one-hot encoding to specified columns.
        :param df: Input DataFrame
        :param columns: List of column names to one-hot encode
        :return: DataFrame with one-hot encoded columns
        """
        return pd.get_dummies(df, columns=columns)

    def target_encode(self, df, columns, target_column):
        """
        Apply target encoding to specified columns based on a target column.
        :param df: Input DataFrame
        :param columns: List of column names to target encode
        :param target_column: Target column for calculating mean encoding
        :return: DataFrame with target-encoded columns
        """
        for col in columns:
            if col not in self.target_mean_map:
                self.target_mean_map[col] = df.groupby(col)[target_column].mean()
            df[col] = df[col].map(self.target_mean_map[col])
        return df

    def ordinal_encode(self, df, columns, order_dict):
        """
        Apply ordinal encoding to specified columns based on a provided order.
        :param df: Input DataFrame
        :param columns: List of column names to ordinal encode
        :param order_dict: Dictionary mapping column names to ordered categories
        :return: DataFrame with ordinal-encoded columns
        """
        for col in columns:
            if col not in self.ordinal_maps:
                self.ordinal_maps[col] = {k: i for i, k in enumerate(order_dict[col])}
            df[col] = df[col].map(self.ordinal_maps[col])
        return df

    def binary_encode(self, df, columns):
        """
        Apply binary encoding to specified columns.
        :param df: Input DataFrame
        :param columns: List of column names to binary encode
        :return: DataFrame with binary-encoded columns
        """
        for col in columns:
            if col not in self.binary_encoders:
                unique_vals = sorted(df[col].unique())
                self.binary_encoders[col] = {val: format(i, f'0{len(bin(len(unique_vals) - 1)[2:])}b') for i, val in enumerate(unique_vals)}
            binary_cols = df[col].map(self.binary_encoders[col]).apply(lambda x: list(map(int, x))).tolist()
            binary_df = pd.DataFrame(binary_cols, columns=[f"{col}_bin_{i}" for i in range(len(binary_cols[0]))], index=df.index)
            df = pd.concat([df.drop(columns=[col]), binary_df], axis=1)
        return df

    def frequency_encode(self, df, columns):
        """
        Apply frequency encoding to specified columns.
        :param df: Input DataFrame
        :param columns: List of column names to frequency encode
        :return: DataFrame with frequency-encoded columns
        """
        for col in columns:
            freq_map = df[col].value_counts(normalize=True)
            df[col] = df[col].map(freq_map)
        return df

    def fit(self, X, y=None):
        """
        Placeholder for scikit-learn compatibility.
        :param X: Input features (DataFrame)
        :param y: Target column (optional)
        :return: self
        """
        return self

    def transform(self, X):
        """
        Placeholder for scikit-learn compatibility. Returns the DataFrame unchanged.
        :param X: Input features (DataFrame)
        :return: Transformed DataFrame
        """
        return X

# Example usage:
if __name__ == "__main__":
    data = {
        'Category': ['A', 'B', 'A', 'C', 'B'],
        'City': ['Hanoi', 'Saigon', 'Hanoi', 'Danang', 'Saigon'],
        'Target': [1, 0, 1, 0, 1]
    }
    df = pd.DataFrame(data)

    encoder = DataEncoding()

    # Label Encoding
    df = encoder.label_encode(df, columns=['Category'])
    print("Label Encoded DataFrame:\n", df)

    # One-Hot Encoding
    df = encoder.one_hot_encode(df, columns=['City'])
    print("One-Hot Encoded DataFrame:\n", df)

    # Target Encoding
    df = encoder.target_encode(df, columns=['Category'], target_column='Target')
    print("Target Encoded DataFrame:\n", df)

    # Ordinal Encoding
    ordinal_order = {'Category': ['A', 'B', 'C']}
    df = encoder.ordinal_encode(df, columns=['Category'], order_dict=ordinal_order)
    print("Ordinal Encoded DataFrame:\n", df)

    # Binary Encoding
    df['BinaryCategory'] = ['A', 'B', 'A', 'C', 'B']
    df = encoder.binary_encode(df, columns=['BinaryCategory'])
    print("Binary Encoded DataFrame:\n", df)

    # Frequency Encoding
    df['FrequencyCategory'] = ['A', 'B', 'A', 'C', 'B']
    df = encoder.frequency_encode(df, columns=['FrequencyCategory'])
    print("Frequency Encoded DataFrame:\n", df)
