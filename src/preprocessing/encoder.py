import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder, TargetEncoder
from category_encoders import BinaryEncoder, CountEncoder


class Encoders:
    def __init__(self, df, encoder_columns_dict):
        """
        Initialize the encoder with a dataframe and encoding specifications
        encoder_columns_dict format: {
            'label': ['col1', 'col2'],
            'one_hot': ['col3'],
            'target': ['col4'],
            etc.
        }
        """
        self.df = df
        self.encoder_list = ['label', 'one_hot', 'target', 'ordinal', 'binary', 'frequency']
        self.encoder_columns_dict = encoder_columns_dict
        
        # Store encoders separately for each column
        self.label_encoders = {col: LabelEncoder() for col in encoder_columns_dict.get('label', [])}
        self.one_hot_encoder = OneHotEncoder(sparse_output=False)
        self.ordinal_encoder = OrdinalEncoder()
        self.target_encoder = TargetEncoder()
        self.binary_encoder = BinaryEncoder()
        self.count_encoder = CountEncoder(normalize=True)

    def fit(self, target_column=None):
        """
        Fit encoders to the data according to encoder_columns_dict
        """
        for encoder_type, columns in self.encoder_columns_dict.items():
            if encoder_type == 'label':
                # Fit each column separately with its own LabelEncoder
                for col in columns:
                    self.label_encoders[col].fit(self.df[col])
            elif encoder_type == 'one_hot':
                self.one_hot_encoder.fit(self.df[columns])
            elif encoder_type == 'target':
                if target_column is None:
                    raise ValueError("target_column must be specified for target encoding")
                self.target_encoder.fit(self.df[columns], self.df[target_column])
            elif encoder_type == 'ordinal':
                self.ordinal_encoder.fit(self.df[columns])
            elif encoder_type == 'binary':
                self.binary_encoder.fit(self.df[columns])
            elif encoder_type == 'frequency':
                self.count_encoder.fit(self.df[columns])
            else:
                raise ValueError(f"Unknown encoder type: {encoder_type}")
        return self

    def transform(self, df):
        """
        Transform the data using fitted encoders
        """
        df = df.copy()  # Create a copy to avoid modifying the original dataframe
        
        for encoder_type, columns in self.encoder_columns_dict.items():
            if encoder_type == 'label':
                # Transform each column separately
                for col in columns:
                    df[col] = self.label_encoders[col].transform(df[col])
            
            elif encoder_type == 'one_hot':
                # Handle one-hot encoding with proper column names
                encoded_array = self.one_hot_encoder.transform(df[columns])
                feature_names = self.one_hot_encoder.get_feature_names_out(columns)
                encoded_df = pd.DataFrame(encoded_array, columns=feature_names, index=df.index)
                
                # Replace original columns with encoded ones
                df = df.drop(columns=columns)
                df = pd.concat([df, encoded_df], axis=1)
            
            elif encoder_type == 'target':
                df[columns] = self.target_encoder.transform(df[columns])
            elif encoder_type == 'ordinal':
                df[columns] = self.ordinal_encoder.transform(df[columns])
            elif encoder_type == 'binary':
                encoded_df = self.binary_encoder.transform(df[columns])
                df = df.drop(columns=columns)
                df = pd.concat([df, encoded_df], axis=1)
            elif encoder_type == 'frequency':
                df[columns] = self.count_encoder.transform(df[columns])

        return df

    def fit_transform(self, target_column=None):
        """
        Fit and transform the data in one step using the encoders' native fit_transform methods
        """
        df = self.df.copy()  # Create a copy to avoid modifying the original dataframe
        
        for encoder_type, columns in self.encoder_columns_dict.items():
            if encoder_type == 'label':
                # Fit_transform each column separately
                for col in columns:
                    df[col] = self.label_encoders[col].fit_transform(df[col])
            
            elif encoder_type == 'one_hot':
                # Handle one-hot encoding with proper column names
                encoded_array = self.one_hot_encoder.fit_transform(df[columns])
                feature_names = self.one_hot_encoder.get_feature_names_out(columns)
                encoded_df = pd.DataFrame(encoded_array, columns=feature_names, index=df.index)
                
                # Replace original columns with encoded ones
                df = df.drop(columns=columns)
                df = pd.concat([df, encoded_df], axis=1)
            
            elif encoder_type == 'target':
                if target_column is None:
                    raise ValueError("target_column must be specified for target encoding")
                df[columns] = self.target_encoder.fit_transform(df[columns], df[target_column])
            
            elif encoder_type == 'ordinal':
                df[columns] = self.ordinal_encoder.fit_transform(df[columns])
            
            elif encoder_type == 'binary':
                encoded_df = self.binary_encoder.fit_transform(df[columns])
                df = df.drop(columns=columns)
                df = pd.concat([df, encoded_df], axis=1)
            
            elif encoder_type == 'frequency':
                df[columns] = self.count_encoder.fit_transform(df[columns])
            
            else:
                raise ValueError(f"Unknown encoder type: {encoder_type}")

        return df
