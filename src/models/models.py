from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor, \
    RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from xgboost import XGBRegressor, XGBClassifier, plot_importance
import optuna
from sklearn.model_selection import cross_val_score
import numpy as np
import matplotlib.pyplot as plt
import yaml


class BoostingRegressionOptimize:
    def __init__(self, X, y, n_trials=100):
        """
        Use get_model() method to retrieve optimized model
        :param X: Input features
        :param y: Target variable
        :param n_trials: Number of optimization trials
        """
        self.models_list = ['DecisionTreeRegressor', 'RandomForestRegressor', 'AdaBoostRegressor',
                            'GradientBoostingRegressor', 'XGBRegressor']
        self.X = X
        self.y = y
        self.n_trials = n_trials
        self.model = None
        self.model_name = None

    def _optuna_DTRegressor(self):
        """
        Optimize hyperparameters for Decision Tree Regressor
        :return: Optimized Decision Tree Regressor
        """

        def objective(trial):
            dt = DecisionTreeRegressor(
                max_depth=trial.suggest_int('max_depth', 1, 50),
                min_samples_leaf=trial.suggest_int('min_samples_leaf', 1, 20),
                ccp_alpha=trial.suggest_float('ccp_alpha', 1e-8, 1e-2, log=True),
                random_state=42
            )
            return np.mean(cross_val_score(dt, self.X, self.y, scoring='r2', cv=5))

        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=self.n_trials)

        dt = DecisionTreeRegressor(
            max_depth=study.best_params['max_depth'],
            min_samples_leaf=study.best_params['min_samples_leaf'],
            ccp_alpha=study.best_params['ccp_alpha'],
            random_state=42
        )
        return dt

    def _optuna_RFRegressor(self):
        """
        Optimize hyperparameters for Random Forest Regressor
        :return: Optimized Random Forest Regressor
        """

        def objective(trial):
            rf = RandomForestRegressor(
                n_estimators=trial.suggest_int('n_estimators', 100, 1000),
                max_depth=trial.suggest_int('max_depth', 5, 50),
                min_samples_leaf=trial.suggest_int('min_samples_leaf', 1, 10),
                max_features=trial.suggest_int('max_features', 1, self.X.shape[1]),
                max_samples=trial.suggest_float('max_samples', 0.5, 1.0),
                bootstrap=True,
                oob_score=True,
                random_state=42
            )
            return np.mean(cross_val_score(rf, self.X, self.y, scoring='r2', cv=5))

        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=self.n_trials)

        rf = RandomForestRegressor(
            n_estimators=study.best_params['n_estimators'],
            max_depth=study.best_params['max_depth'],
            min_samples_leaf=study.best_params['min_samples_leaf'],
            max_features=study.best_params['max_features'],
            max_samples=study.best_params['max_samples'],
            bootstrap=True,
            oob_score=True,
            random_state=42
        )
        return rf

    def _optuna_ABRegressor(self):
        """
        Optimize hyperparameters for AdaBoost Regressor
        :return: Optimized AdaBoost Regressor
        """

        def objective(trial):
            ab = AdaBoostRegressor(
                n_estimators=trial.suggest_int('n_estimators', 50, 500),
                learning_rate=trial.suggest_float('learning_rate', 0.01, 1.0, log=True),
                loss=trial.suggest_categorical('loss', ['linear', 'square', 'exponential']),
                estimator=DecisionTreeRegressor(
                    max_depth=trial.suggest_int('base_estimator_max_depth', 1, 20)
                ),
                random_state=42
            )
            return np.mean(cross_val_score(ab, self.X, self.y, scoring='r2', cv=5))

        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=self.n_trials)

        ab = AdaBoostRegressor(
            n_estimators=study.best_params['n_estimators'],
            learning_rate=study.best_params['learning_rate'],
            loss=study.best_params['loss'],
            estimator=DecisionTreeRegressor(
                max_depth=study.best_params['base_estimator_max_depth']
            ),
            random_state=42
        )
        return ab

    def _optuna_GBRegressor(self):
        """
        Optimize hyperparameters for Gradient Boosting Regressor
        :return: Optimized Gradient Boosting Regressor
        """

        def objective(trial):
            gb = GradientBoostingRegressor(
                n_estimators=trial.suggest_int('n_estimators', 100, 2000),
                max_depth=trial.suggest_int('max_depth', 3, 20),
                min_samples_split=trial.suggest_int('min_samples_split', 2, 50),
                learning_rate=trial.suggest_float('learning_rate', 0.001, 0.3, log=True),
                random_state=42
            )
            return np.mean(cross_val_score(gb, self.X, self.y, scoring='r2', cv=5))

        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=self.n_trials)

        gb = GradientBoostingRegressor(
            n_estimators=study.best_params['n_estimators'],
            max_depth=study.best_params['max_depth'],
            min_samples_split=study.best_params['min_samples_split'],
            learning_rate=study.best_params['learning_rate'],
            random_state=42
        )
        return gb

    def _optuna_XGBRegressor(self):
        """
        Optimize hyperparameters for XGBoost Regressor
        :return: Optimized XGBoost Regressor
        """

        def objective(trial):
            xgb = XGBRegressor(
                n_estimators=trial.suggest_int('n_estimators', 100, 2000),
                max_depth=trial.suggest_int('max_depth', 3, 20),
                eta=trial.suggest_float('eta', 0.001, 0.3, log=True),
                subsample=trial.suggest_float('subsample', 0.5, 1.0),
                colsample_bytree=trial.suggest_float('colsample_bytree', 0.3, 1.0),
                gamma=trial.suggest_float('gamma', 1e-8, 1.0, log=True),
                reg_lambda=trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
                random_state=42
            )
            return np.mean(cross_val_score(xgb, self.X, self.y, scoring='r2', cv=5))

        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=self.n_trials)

        xgb = XGBRegressor(
            n_estimators=study.best_params['n_estimators'],
            max_depth=study.best_params['max_depth'],
            eta=study.best_params['eta'],
            subsample=study.best_params['subsample'],
            colsample_bytree=study.best_params['colsample_bytree'],
            gamma=study.best_params['gamma'],
            reg_lambda=study.best_params['reg_lambda'],
            random_state=42
        )
        return xgb

    def get_pretrained(self, model):
        with open('D:/HUST/_Intro to DS/Capstone Project/Football-Match-Prediction/pretrained_models.yaml', 'r') as file:
            config = yaml.safe_load(file)
        if model not in self.models_list:
            raise ValueError(f'Model must be in {self.models_list}')
        elif model == 'DecisionTreeRegressor':
            self.model = DecisionTreeRegressor(
                max_depth=config['REGRESSION']['DT']['MAX_DEPTH'],
                min_samples_leaf=config['REGRESSION']['DT']['MIN_SAMPLES_LEAF'],
                ccp_alpha=config['REGRESSION']['DT']['CCP_ALPHA'],
                random_state=42
            )
            self.model_name = 'DecisionTreeRegressor'
        elif model == 'RandomForestRegressor':
            self.model = RandomForestRegressor(
                n_estimators=config['REGRESSION']['RF']['N_ESTIMATORS'],
                max_depth=config['REGRESSION']['RF']['MAX_DEPTH'],
                min_samples_leaf=config['REGRESSION']['RF']['MIN_SAMPLES_LEAF'],
                max_features=config['REGRESSION']['RF']['MAX_FEATURES'],
                max_samples=config['REGRESSION']['RF']['MAX_SAMPLES'],
                bootstrap=True,
                oob_score=True,
                random_state=42
            )
            self.model_name = 'RandomForestRegressor'
        elif model == 'AdaBoostRegressor':
            self.model = AdaBoostRegressor(
                n_estimators=config['REGRESSION']['AB']['N_ESTIMATORS'],
                learning_rate=config['REGRESSION']['AB']['LEARNING_RATE'],
                loss=config['REGRESSION']['AB']['LOSS'],
                estimator=DecisionTreeRegressor(
                    max_depth=config['REGRESSION']['AB']['BASE_ESTIMATOR_MAX_DEPTH']
                ),
                random_state=42
            )
            self.model_name = 'AdaBoostRegressor'
        elif model == 'GradientBoostingRegressor':
            self.model = GradientBoostingRegressor(
                n_estimators=config['REGRESSION']['GB']['N_ESTIMATORS'],
                max_depth=config['REGRESSION']['GB']['MAX_DEPTH'],
                min_samples_split=config['REGRESSION']['GB']['MIN_SAMPLES_SPLIT'],
                learning_rate=config['REGRESSION']['GB']['LEARNING_RATE'],
                random_state=42
            )
            self.model_name = 'GradientBoostingRegressor'
        elif model == 'XGBRegressor':
            self.model = XGBRegressor(
                n_estimators=config['REGRESSION']['XGB']['N_ESTIMATORS'],
                max_depth=config['REGRESSION']['XGB']['MAX_DEPTH'],
                eta=config['REGRESSION']['XGB']['ETA'],
                subsample=config['REGRESSION']['XGB']['SUBSAMPLE'],
                colsample_bytree=config['REGRESSION']['XGB']['COLSAMPLE_BYTREE'],
                gamma=config['REGRESSION']['XGB']['GAMMA'],
                reg_lambda=config['REGRESSION']['XGB']['REG_LAMBDA'],
                random_state=42
            )
            self.model_name = 'XGBRegressor'
        file.close()
        return self.model

    def get_model(self, model) -> (DecisionTreeRegressor or RandomForestRegressor or AdaBoostRegressor or
                                   GradientBoostingRegressor or XGBRegressor):
        """
        Retrieve optimized model
        :param model: Str model name to optimize. It should be in ['DecisionTreeRegressor', 'RandomForestRegressor',
        'AdaBoostRegressor', 'GradientBoostingRegressor', 'XGBRegressor']
        :return: Optimized regression model
        """
        if model not in self.models_list:
            raise ValueError(f'Model must be in {self.models_list}')
        elif model == 'DecisionTreeRegressor':
            self.model = self._optuna_DTRegressor()
            self.model_name = 'DecisionTreeRegressor'
        elif model == 'RandomForestRegressor':
            self.model = self._optuna_RFRegressor()
            self.model_name = 'RandomForestRegressor'
        elif model == 'AdaBoostRegressor':
            self.model = self._optuna_ABRegressor()
            self.model_name = 'AdaBoostRegressor'
        elif model == 'GradientBoostingRegressor':
            self.model = self._optuna_GBRegressor()
            self.model_name = 'GradientBoostingRegressor'
        elif model == 'XGBRegressor':
            self.model = self._optuna_XGBRegressor()
            self.model_name = 'XGBRegressor'
        return self.model

    def plot_importance_feats(self, feature_limits=None):
        if not self.model:
            return None

        # Get feature importances and names
        if self.model_name == 'XGBRegressor':
            # Retrieve feature importances for XGBoost
            feature_importances = self.model.get_booster().get_score(importance_type='weight')
            feature_names = list(feature_importances.keys())
            feature_importances = np.array(list(feature_importances.values()))
        else:
            # For scikit-learn models
            feature_importances = self.model.feature_importances_
            feature_names = self.X.columns

        # Sort feature importances in descending order
        sorted_indices = np.argsort(feature_importances)[::-1]
        sorted_importances = feature_importances[sorted_indices]
        sorted_feature_names = [feature_names[i] for i in sorted_indices]

        # Apply feature limits if specified
        if feature_limits is not None and feature_limits < len(feature_names):
            sorted_importances = sorted_importances[:feature_limits]
            sorted_feature_names = sorted_feature_names[:feature_limits]

        # Plot feature importances
        plt.figure(figsize=(10, 6))
        plt.barh(range(len(sorted_importances)), sorted_importances, align='center')
        plt.yticks(range(len(sorted_importances)), sorted_feature_names)
        plt.xlabel('Feature Importance')
        plt.title(f'Top {len(sorted_importances)} Feature Importance for {self.model_name}')
        plt.tight_layout()
        plt.show()


class BoostingClassificationOptimize:
    def __init__(self, X, y, n_trials=100):
        """
        Use get_model() method to retrieve optimized model
        :param X: Input features
        :param y: Target variable
        :param n_trials: Number of optimization trials
        """
        self.models_list = ['DecisionTreeClassifier', 'RandomForestClassifier', 'AdaBoostClassifier',
                            'GradientBoostingClassifier', 'XGBClassifier']
        self.X = X
        self.y = y
        self.n_trials = n_trials
        self.model = None
        self.model_name = None

    def _optuna_DTClassifier(self):
        """
        Optimize hyperparameters for Decision Tree Classifier
        :return: Optimized Decision Tree Classifier
        """

        def objective(trial):
            dt = DecisionTreeClassifier(
                criterion=trial.suggest_categorical('criterion', ['gini', 'entropy', 'log_loss']),
                max_depth=trial.suggest_int('max_depth', 1, 50),
                min_samples_leaf=trial.suggest_int('min_samples_leaf', 1, 20),
                ccp_alpha=trial.suggest_float('ccp_alpha', 1e-8, 1e-2, log=True),
                random_state=42
            )
            scores = cross_val_score(dt, self.X, self.y, scoring='f1_macro', cv=5)
            return np.mean(scores)

        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=self.n_trials)

        dt = DecisionTreeClassifier(
            criterion=study.best_params['criterion'],
            max_depth=study.best_params['max_depth'],
            min_samples_leaf=study.best_params['min_samples_leaf'],
            ccp_alpha=study.best_params['ccp_alpha'],
            random_state=42
        )
        return dt

    def _optuna_RFClassifier(self):
        """
        Optimize hyperparameters for Random Forest Classifier
        :return: Optimized Random Forest Classifier
        """

        def objective(trial):
            rf = RandomForestClassifier(
                criterion=trial.suggest_categorical('criterion', ['gini', 'entropy', 'log_loss']),
                n_estimators=trial.suggest_int('n_estimators', 100, 1000),
                max_depth=trial.suggest_int('max_depth', 5, 50),
                min_samples_leaf=trial.suggest_int('min_samples_leaf', 1, 10),
                max_features=trial.suggest_int('max_features', 1, self.X.shape[1]),
                max_samples=trial.suggest_float('max_samples', 0.5, 1.0),
                bootstrap=True,
                oob_score=True,
                random_state=42
            )
            scores = cross_val_score(rf, self.X, self.y, scoring='f1_macro', cv=5)
            return np.mean(scores)

        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=self.n_trials)

        rf = RandomForestClassifier(
            criterion=study.best_params['criterion'],
            n_estimators=study.best_params['n_estimators'],
            max_depth=study.best_params['max_depth'],
            min_samples_leaf=study.best_params['min_samples_leaf'],
            max_features=study.best_params['max_features'],
            max_samples=study.best_params['max_samples'],
            bootstrap=True,
            oob_score=True,
            random_state=42
        )
        return rf

    def _optuna_ABClassifier(self):
        """
        Optimize hyperparameters for AdaBoost Classifier
        :return: Optimized AdaBoost Classifier
        """

        def objective(trial):
            ab = AdaBoostClassifier(
                n_estimators=trial.suggest_int('n_estimators', 50, 500),
                learning_rate=trial.suggest_float('learning_rate', 0.01, 1.0, log=True),
                estimator=DecisionTreeClassifier(
                    max_depth=trial.suggest_int('base_estimator_max_depth', 1, 20)
                ),
                random_state=42
            )
            scores = cross_val_score(ab, self.X, self.y, scoring='f1_macro', cv=5)
            return np.mean(scores)

        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=self.n_trials)

        ab = AdaBoostClassifier(
            n_estimators=study.best_params['n_estimators'],
            learning_rate=study.best_params['learning_rate'],
            estimator=DecisionTreeClassifier(
                max_depth=study.best_params['base_estimator_max_depth']
            ),
            random_state=42
        )
        return ab

    def _optuna_GBClassifier(self):
        """
        Optimize hyperparameters for Gradient Boosting Classifier
        :return: Optimized Gradient Boosting Classifier
        """

        def objective(trial):
            gb = GradientBoostingClassifier(
                n_estimators=trial.suggest_int('n_estimators', 100, 2000),
                max_depth=trial.suggest_int('max_depth', 3, 20),
                min_samples_split=trial.suggest_int('min_samples_split', 2, 50),
                learning_rate=trial.suggest_float('learning_rate', 0.001, 0.3, log=True),
                random_state=42
            )
            scores = cross_val_score(gb, self.X, self.y, scoring='f1_macro', cv=5)
            return np.mean(scores)

        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=self.n_trials)

        gb = GradientBoostingClassifier(
            n_estimators=study.best_params['n_estimators'],
            max_depth=study.best_params['max_depth'],
            min_samples_split=study.best_params['min_samples_split'],
            learning_rate=study.best_params['learning_rate'],
            random_state=42
        )
        return gb

    def _optuna_XGBClassifier(self):
        """
        Optimize hyperparameters for XGBoost Classifier
        :return: Optimized XGBoost Classifier
        """

        def objective(trial):
            xgb = XGBClassifier(
                objective='multi:softprob',
                n_estimators=trial.suggest_int('n_estimators', 100, 2000),
                max_depth=trial.suggest_int('max_depth', 3, 20),
                eta=trial.suggest_float('eta', 0.001, 0.3, log=True),
                subsample=trial.suggest_float('subsample', 0.5, 1.0),
                colsample_bytree=trial.suggest_float('colsample_bytree', 0.3, 1.0),
                gamma=trial.suggest_float('gamma', 1e-8, 1.0, log=True),
                reg_lambda=trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
                random_state=42
            )
            scores = cross_val_score(xgb, self.X, self.y, scoring='f1_macro', cv=5)
            return np.mean(scores)

        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=self.n_trials)

        xgb = XGBClassifier(
            objective='multi:softprob',
            n_estimators=study.best_params['n_estimators'],
            max_depth=study.best_params['max_depth'],
            eta=study.best_params['eta'],
            subsample=study.best_params['subsample'],
            colsample_bytree=study.best_params['colsample_bytree'],
            gamma=study.best_params['gamma'],
            reg_lambda=study.best_params['reg_lambda'],
            random_state=42
        )
        return xgb
    
    def get_pretrained(self, model):
        with open('D:/HUST/_Intro to DS/Capstone Project/Football-Match-Prediction/pretrained_models.yaml', 'r') as file:
            config = yaml.safe_load(file)
        if model not in self.models_list:
            raise ValueError(f'Model must be in {self.models_list}')
        elif model == 'DecisionTreeClassifier':
            self.model = DecisionTreeClassifier(
                criterion=config['CLASSIFICATION']['DT']['CRITERION'],
                max_depth=config['CLASSIFICATION']['DT']['MAX_DEPTH'],
                min_samples_leaf=config['CLASSIFICATION']['DT']['MIN_SAMPLES_LEAF'],
                ccp_alpha=config['CLASSIFICATION']['DT']['CCP_ALPHA'],
                random_state=42
            )
            self.model_name = 'DecisionTreeClassifier'
        elif model == 'RandomForestClassifier':
            self.model = RandomForestClassifier(
                criterion=config['CLASSIFICATION']['RF']['CRITERION'],
                n_estimators=config['CLASSIFICATION']['RF']['N_ESTIMATORS'],
                max_depth=config['CLASSIFICATION']['RF']['MAX_DEPTH'],
                min_samples_leaf=config['CLASSIFICATION']['RF']['MIN_SAMPLES_LEAF'],
                max_features=config['CLASSIFICATION']['RF']['MAX_FEATURES'],
                max_samples=config['CLASSIFICATION']['RF']['MAX_SAMPLES'],
                bootstrap=True,
                oob_score=True,
                random_state=42
            )
            self.model_name = 'RandomForestClassifier'
        elif model == 'AdaBoostClassifier':
            self.model = AdaBoostClassifier(
                n_estimators=config['CLASSIFICATION']['AB']['N_ESTIMATORS'],
                learning_rate=config['CLASSIFICATION']['AB']['LEARNING_RATE'],
                estimator=DecisionTreeClassifier(
                    max_depth=config['CLASSIFICATION']['AB']['BASE_ESTIMATOR_MAX_DEPTH']
                ),
                random_state=42
            )
            self.model_name = 'AdaBoostClassifier'
        elif model == 'GradientBoostingClassifier':
            self.model = GradientBoostingClassifier(
                n_estimators=config['CLASSIFICATION']['GB']['N_ESTIMATORS'],
                max_depth=config['CLASSIFICATION']['GB']['MAX_DEPTH'],
                min_samples_split=config['CLASSIFICATION']['GB']['MIN_SAMPLES_SPLIT'],
                learning_rate=config['CLASSIFICATION']['GB']['LEARNING_RATE'],
                random_state=42
            )
            self.model_name = 'GradientBoostingClassifier'
        elif model == 'XGBClassifier':
            self.model = XGBClassifier(
                objective='multi:softprob',
                n_estimators=config['CLASSIFICATION']['XGB']['N_ESTIMATORS'],
                max_depth=config['CLASSIFICATION']['XGB']['MAX_DEPTH'],
                eta=config['CLASSIFICATION']['XGB']['ETA'],
                subsample=config['CLASSIFICATION']['XGB']['SUBSAMPLE'],
                colsample_bytree=config['CLASSIFICATION']['XGB']['COLSAMPLE_BYTREE'],
                gamma=config['CLASSIFICATION']['XGB']['GAMMA'],
                reg_lambda=config['CLASSIFICATION']['XGB']['REG_LAMBDA'],
                random_state=42
            )
            self.model_name = 'XGBClassifier'
        file.close()
        return self.model

    def get_model(self, model):
        """
        Retrieve optimized model
        :param model: Str model name to optimize. It should be in ['DecisionTreeClassifier', 'RandomForestClassifier',
        'AdaBoostClassifier', 'GradientBoostingClassifier', 'XGBClassifier']
        :return: Optimized classification model
        """
        if model not in self.models_list:
            raise ValueError(f'Model must be in {self.models_list}')
        elif model == 'DecisionTreeClassifier':
            self.model = self._optuna_DTClassifier()
            self.model_name = 'DecisionTreeClassifier'
        elif model == 'RandomForestClassifier':
            self.model = self._optuna_RFClassifier()
            self.model_name = 'RandomForestClassifier'
        elif model == 'AdaBoostClassifier':
            self.model = self._optuna_ABClassifier()
            self.model_name = 'AdaBoostClassifier'
        elif model == 'GradientBoostingClassifier':
            self.model = self._optuna_GBClassifier()
            self.model_name = 'GradientBoostingClassifier'
        elif model == 'XGBClassifier':
            self.model = self._optuna_XGBClassifier()
            self.model_name = 'XGBClassifier'
        return self.model

    def plot_importance_feats(self, feature_limits=None):
        """
        Plot feature importances for the optimized model
        :param feature_limits: Number of top features to plot (optional)
        :return: None (displays plot)
        """
        if not self.model:
            return None

        # Get feature importances and names
        if self.model_name == 'XGBClassifier':
            # Retrieve feature importances for XGBoost
            feature_importances = self.model.get_booster().get_score(importance_type='weight')
            feature_names = list(feature_importances.keys())
            feature_importances = np.array(list(feature_importances.values()))
        else:
            # For scikit-learn models
            feature_importances = self.model.feature_importances_
            feature_names = self.X.columns

        # Sort feature importances in descending order
        sorted_indices = np.argsort(feature_importances)[::-1]
        sorted_importances = feature_importances[sorted_indices]
        sorted_feature_names = [feature_names[i] for i in sorted_indices]

        # Apply feature limits if specified
        if feature_limits is not None and feature_limits < len(feature_names):
            sorted_importances = sorted_importances[:feature_limits]
            sorted_feature_names = sorted_feature_names[:feature_limits]

        # Plot feature importances
        plt.figure(figsize=(10, 6))
        plt.barh(range(len(sorted_importances)), sorted_importances, align='center')
        plt.yticks(range(len(sorted_importances)), sorted_feature_names)
        plt.xlabel('Feature Importance')
        plt.title(f'Top {len(sorted_importances)} Feature Importance for {self.model_name}')
        plt.tight_layout()
        plt.show()