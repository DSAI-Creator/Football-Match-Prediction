import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import KNNImputer, IterativeImputer


def _check_imputer_kwargs(imputer_name, **kwargs):
    if imputer_name == 'knn':
        try:
            n_neighbors = kwargs['n_neighbors']
        except KeyError:
            n_neighbors = 5
        try:
            metric = kwargs['metric']
        except KeyError:
            metric = 'nan_euclidean'
        return n_neighbors, metric
    elif imputer_name == 'mice':
        try:
            estimator = kwargs['estimator']
        except KeyError:
            estimator = None
        try:
            max_iter = kwargs['max_iter']
        except KeyError:
            max_iter = 10
        try:
            n_nearest_features = kwargs['n_nearest_features']
        except KeyError:
            n_nearest_features = None
        try:
            verbose = kwargs['verbose']
        except KeyError:
            verbose = 0
        try:
            random_state = kwargs['random_state']
        except KeyError:
            random_state = None
        return estimator, max_iter, n_nearest_features, verbose, random_state


class HandleMissingValues:
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.missing_values_per_col = df.isnull().sum()
        self.percent_missing_values = self.missing_values_per_col * 100 / len(self.df)

    def get_na(self, col_names: list = None, display: str = 'number') -> pd.Series:
        """
        Function for Display NA values of given columns.
        :param col_names: List of columns
        :param display: Str of way to display missing values
        :return: Series about Number or Percent of Missing Values of each column
        """
        if display not in ['number', 'percent']:
            raise ValueError('get_na(): "display" must be in ["number", "percent"]')
        missing_values_per_col = self.df[col_names].isnull().sum() if col_names else self.df.isnull().sum()
        if display == 'percent':
            missing_values_per_col = round(missing_values_per_col * 100 / len(self.df), 3)
        return missing_values_per_col

    def remove_na(self, col_names: list = None, technique: str = 'row', inplace: bool = False) -> pd.DataFrame:
        """
        Function for Removing Rows with NA values of given columns (method = 'row'),
        and Removing given Columns (method = 'column').
        :param inplace: Bool for whether to Modify the DataFrame or not
        :param col_names: List of columns
        :param technique: Str of way to remove missing values
        :return: None (inplace=True), DataFrame (inplace=False)
        """
        if technique not in ['row', 'column']:
            raise ValueError('remove_na(): "method" must be in ["row", "column"]')
        df = None
        if inplace:
            if technique == 'row':
                self.df.dropna(subset=col_names, inplace=True)
            else:
                self.df.drop(columns=col_names, axis=1, inplace=True)
        else:
            if technique == 'row':
                df = self.df.dropna(subset=col_names)
            else:
                df = self.df.drop(columns=col_names)
        return df

    def fill_na(self, col_names: list = None, technique: str = 'mean', inplace: bool = False, **kwargs) -> pd.DataFrame:
        """
        Function for Filling NaN values of given columns with different techniques
        :param col_names: List of columns
        :param technique: Str of way to fill NaN values
        :param inplace: Bool for whether to Modify the DataFrame or not
        :param kwargs: Args for different imputer
        :return: None (inplace=True), DataFrame (inplace=False)
        """
        if technique not in ['mean', 'median', 'mode', 'interpolation', 'knn', 'mice']:
            raise ValueError('fill_na(): "method" must be in '
                             '["mean", "median", "mode", "interpolation", "knn", "mice"]')
        df = None if inplace else self.df.copy()
        if inplace:
            if technique == 'mean':
                self.df[col_names] = self.df[col_names].fillna(self.df[col_names].mean())
            elif technique == 'median':
                self.df[col_names] = self.df[col_names].fillna(self.df[col_names].median())
            elif technique == 'mode':
                self.df[col_names] = self.df[col_names].fillna(self.df[col_names].mode()[0])
            elif technique == 'interpolation':
                self.df[col_names] = self.df[col_names].interpolate(method=kwargs['method'],
                                                                    order=kwargs['order'])
            elif technique == 'knn':
                n_neighbors, metric = _check_imputer_kwargs('knn', **kwargs)
                imputer = KNNImputer(n_neighbors=n_neighbors,
                                     metric=metric)
                self.df[col_names] = imputer.fit_transform(self.df[col_names])
            else:
                estimator, max_iter, n_nearest_features, verbose, random_state = (
                    _check_imputer_kwargs('mice', **kwargs))
                imputer = IterativeImputer(estimator=estimator,
                                           max_iter=max_iter,
                                           n_nearest_features=n_nearest_features,
                                           verbose=verbose,
                                           random_state=random_state)
                self.df[col_names] = imputer.fit_transform(self.df[col_names])
        else:
            if technique == 'mean':
                df[col_names] = df[col_names].fillna(df[col_names].mean())
            elif technique == 'median':
                df[col_names] = df[col_names].fillna(df[col_names].median())
            elif technique == 'mode':
                df[col_names] = df[col_names].fillna(df[col_names].mode()[0])
            elif technique == 'interpolation':
                df[col_names] = df[col_names].interpolate(method=kwargs['method'],
                                                          order=kwargs['order'])
            elif technique == 'knn':
                n_neighbors, metric = _check_imputer_kwargs('knn', **kwargs)
                imputer = KNNImputer(n_neighbors=n_neighbors,
                                     metric=metric)
                df[col_names] = imputer.fit_transform(df[col_names])
            else:
                estimator, max_iter, n_nearest_features, verbose, random_state = (
                    _check_imputer_kwargs('mice', **kwargs))
                imputer = IterativeImputer(estimator=estimator,
                                           max_iter=max_iter,
                                           n_nearest_features=n_nearest_features,
                                           verbose=verbose,
                                           random_state=random_state)
                df[col_names] = imputer.fit_transform(df[col_names])
        return df


def test():
    DATA_PATH = '../../data/raw/all_teams_data.csv'
    df = pd.read_csv(DATA_PATH)
    tool = HandleMissingValues(df)
    print("--------------------Fill Missing Values--------------------")
    print(df['xG'].mean())
    print(tool.fill_na(['xG'], technique='mice', inplace=True))
    print(tool.df['xG'].isnull().sum())
    print(tool.df['xG'].mean())


if __name__ == '__main__':
    test()
