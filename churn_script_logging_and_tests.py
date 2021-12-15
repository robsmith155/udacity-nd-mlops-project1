"""
This module contains tests for some of the methods in the CustomerChurn class.

@author: Rob Smith
date: 10th December 2021

"""

import os
import logging
import yaml
import pandas as pd
from box import Box
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score
import churn_library as cls

logging.basicConfig(
    filename='./logs/churn_library.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')

# Load config file
with open("config.yaml", "r", encoding='utf-8') as ymlfile:
    config = Box(yaml.safe_load(ymlfile))


class TestCustomerChurn(cls.CustomerChurn):
    """
    Run tests for the methods in CustomerChurn.
    """

    def test_import_data(self) -> None:
        '''
        Test that the import_data method correctly loads the data.
        '''
        try:
            self.import_data(filepath=config.testing.data_filepath)
            logging.info("Testing import_data: File loaded successfully")
        except FileNotFoundError as err:
            logging.error(
                f"Testing import_data: The file {config.testing.data_filepath} wasn't found")
            raise err

        try:
            assert self.df.shape[0] > 0
            assert self.df.shape[1] > 0
            logging.info("Testing import_data: The file is not empty")
        except AssertionError as err:
            logging.error(
                "Testing import_data: The file doesn't appear to have rows and columns")
            raise err

    def test_create_churn_feature(self) -> None:
        '''
        Test that the create_churn_feature method creates the Churn variable.
        '''''
        try:
            self.create_churn_feature()
            assert 'Churn' in self.df.columns
            logging.info(
                "Testing create_churn_feature: Feature successfully created")
        except AssertionError as err:
            logging.error(
                "Testing create_churn_feature: A feature named Churn does not exist in self.df")
            raise err

        try:
            assert self.df['Churn'].min() == 0.0
            assert self.df['Churn'].max() == 1.0
            logging.info(
                "Testing create_churn_feature: Verified min and max value of Churn column are 0.0 and 1.0 respectively")
        except AssertionError as err:
            logging.error(
                f"Testing create_churn_feature: Expected min and max values of Churn column to be 0.0 and 1.0, but got {self.df['Churn'].min()} and {self.df['Churn'].max()}")
            raise err

    def test_perform_eda(self) -> None:
        '''
        Test that the perform_eda method outputs all the EDA images
        '''
        # Remove any files already in output directory
        output_dir = config.eda.output_dir
        num_files = len(os.listdir(output_dir))
        if num_files != 0:
            for f in os.listdir(output_dir):
                os.remove(os.path.join(output_dir, f))
        try:
            self.perform_eda(
                histogram_features=config.eda.plot_histograms,
                output_dir=output_dir)
            expected_num_plots = len(config.eda.plot_histograms) + 1
            assert len(os.listdir(output_dir)) == expected_num_plots
            logging.info(
                f"Testing perform_eda: A total of {len(os.listdir(output_dir))} plots were successfully saved to {output_dir}")
        except AssertionError as err:
            logging.error(
                f"Testing perform_eda: Expected {expected_num_plots} plots in {output_dir}, but only found {len(os.listdir(output_dir))}")
            raise err

    def test_perform_feature_engineering(self) -> None:
        '''
        Test that the perform_feature_engineering method created the encoded
        categorical features.
        '''
        if isinstance(config.feature_engineering.encoded_cat_names, list):
            encoded_feature_names = config.feature_engineering.encoded_cat_names
        else:
            encoded_feature_names = []
            for feature in config.feature_engineering.categorical_columns:
                encoded_feature_names.append(f'{feature}_Churn')
        logging.info(
            f"Testing perform_feature_engineering: Encoded feature names {encoded_feature_names}")
        try:
            self.perform_feature_engineering(
                category_lst=config.feature_engineering.categorical_columns)
            intersection = set(encoded_feature_names).intersection(
                set(self.df.columns))
            assert len(encoded_feature_names) - len(intersection) == 0
            logging.info(
                "Testing perform_feature_engineering: Encoded categorical features successfully created")
        except AssertionError as err:
            logging.error(
                f"Testing perform_feature_engineering: Expected {len(encoded_feature_names)} new columns but only got {len(intersection)}.")
            raise err

    def test_prepare_training_data(self) -> None:
        '''
        Test that the prepare_training_data method creates the training and
        test datasets
        '''
        try:
            self.prepare_training_data(
                training_features_lst=config.prepare_data.training_features,
                target=config.prepare_data.target,
                test_size=config.prepare_data.test_size,
                random_state=config.prepare_data.random_state)

            assert isinstance(self.X_train, pd.DataFrame)
            assert isinstance(self.X_test, pd.DataFrame)
            assert isinstance(self.y_train, pd.Series)
            assert isinstance(self.y_test, pd.Series)
            logging.info(
                "Testing prepare_training_data: Training and test datasets successfully created")
        except AssertionError as err:
            logging.error(
                f"Testing prepare_training_data: Unexpected output data types. X_train type is {type(self.X_train)}, expected pd.DataFrame. X_test type is {type(self.X_test)}, expected pd.DataFrame. y_train type is {type(self.y_train)}, expected pd.Series. y_test type is {type(self.y_test)}, expected pd.Series.")
            raise err

    def test_train_models(self) -> None:
        '''
        Test that the prepare_train_models method creates the Random Forest
        and Logistic Regression models
        '''
        try:
            self.train_models(
                rf_param_grid=config.models.random_forest.param_grid,
                num_cv_folds=config.models.random_forest.num_cv_folds,
                random_state=config.models.random_state)

            assert isinstance(self.random_forest_gridcv, GridSearchCV)
            logging.info(
                "Testing train_models: Random Forest model created and of correct type")
        except AssertionError as err:
            logging.error(
                f"Expected Random Forest model to be of the type GridSearchCV, but returned {type(self.random_forest_gridcv)}")
            raise err

        try:
            assert isinstance(self.logistic_classifier, LogisticRegression)
            logging.info(
                "Testing train_models: Logistic Regression model created and of correct type")
        except AssertionError as err:
            logging.error(
                f"Expected Logistic Regression model to be of the type LogisticRegression, but returned {type(self.logistic_classifier)}")
            raise err
        try:
            y_test_preds_rf = self.random_forest_gridcv.best_estimator_.predict(
                self.X_test)
            best_rf_auc_score = roc_auc_score(
                y_true=self.y_test, y_score=y_test_preds_rf)
            assert best_rf_auc_score > 0.85
            logging.info(
                f"Testing train models: Random Forest best AUC score of {best_rf_auc_score}. Performance okay.")
        except AssertionError as err:
            logging.error(
                f"Testing train models: Best Random Forest AUC score of {best_rf_auc_score} does not meet expectations.")
            raise err

        try:
            y_test_preds_logistic = self.logistic_classifier.predict(
                self.X_test)
            logistic_auc_score = roc_auc_score(
                y_true=self.y_test, y_score=y_test_preds_logistic)
            assert logistic_auc_score > 0.65
            logging.info(
                f"Testing train models: Logistic Regression model AUC score of {logistic_auc_score}. Performance okay.")
        except AssertionError as err:
            logging.error(
                f"Testing train models: Logistic Regression model AUC score of {logistic_auc_score} does not meet expectations.")
            raise err


if __name__ == "__main__":

    churn_test = TestCustomerChurn()
    # Run tests
    churn_test.test_import_data()
    churn_test.test_create_churn_feature()
    churn_test.test_perform_eda()
    churn_test.test_perform_feature_engineering()
    churn_test.test_prepare_training_data()
    churn_test.test_train_models()
