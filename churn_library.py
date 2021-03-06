"""
This module contains a class named CustomerChurn, which can be used for
to load data and then preprocess, train and evaluate a Random Forest and
Logistic Regression model.

@author: Rob Smith
date: 10th December 2021

"""

import os
from typing import Optional
import joblib
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import yaml
from box import Box
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import plot_roc_curve, classification_report
import dataframe_image as dfi


class CustomerChurn:
    """
    This class enables a model to be trained from scratch for predicting customer churn.

    Attributes:
        df (pd.DataFrame): DataFrame storing data
        X_train (pd.DataFrame): DataFrame storing trainng input features
        X_test (pd.DataFrame): DataFrame storing test input features
        y_train (pd.Series): Training response values
        y_test (pd.Series): Test response values
        random_forest_gridcv (GridSearchCV): Output of grid search for random forest
        logistc_classifier (LogisticRegression): Logistic regression model
        y_train_preds_lr (np.ndarray): Training predictions from logistic regression
        y_train_preds_rf (np.ndarray): Training predictions from random forest
        y_test_preds_lr (np.ndarray): Test predictions from logistic regression
        y_test_preds_rf (np.ndarray): Test predictions from random forest
    """

    def __init__(self):
        self.df = None
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.random_forest_gridcv = None
        self.logistic_classifier = None
        self.y_train_preds_rf = None
        self.y_train_preds_lr = None
        self.y_test_preds_rf = None
        self.y_test_preds_lr = None

    def import_data(self, filepath: str) -> None:
        '''
        Returns DataFrame for the csv found at filepath.

        Args:
            filepath (str): A path to the csv containing the data
        '''
        self.df = pd.read_csv(filepath)

    def create_churn_feature(self) -> None:
        '''
        Adds feature named 'Churn' to the loaded data (stored in self.df).

        Args:
            None
        '''
        self.df['Churn'] = self.df['Attrition_Flag'].apply(
            lambda val: 0 if val == "Existing Customer" else 1)

    def plot_distribution(self, feature: str, output_dir: str) -> None:
        '''
        Plot and save histogram of specified feature in output_dir

        Args:
            feature (str): Name of feature to plot
        '''
        plt.figure(figsize=(20, 10))
        ax = sns.histplot(data=self.df[feature])
        ax.set_title(f'Histogram of {feature}')
        filename = f'{feature}_distribution.png'.lower()
        img_path = os.path.join(output_dir, filename)
        plt.savefig(fname=img_path, dpi=200, format='png')

    def perform_eda(self, histogram_features: list, output_dir: str) -> None:
        '''
        Perform EDA on the input data and save figures to the output_dir

        Args:
            histogram_features (list): List of features to plot as histograms
            output_dir (str): Directory to store the output images
        '''
        # Histograms
        for feature in histogram_features:
            self.plot_distribution(feature=feature, output_dir=output_dir)

        # Heatmap
        plt.figure(figsize=(20, 10))
        sns.heatmap(self.df.corr(), annot=False, cmap='Dark2_r', linewidths=2)
        img_path = os.path.join(output_dir, 'heatmap.png')
        plt.savefig(fname=img_path, dpi=200, format='png')

    def perform_feature_engineering(
            self,
            category_lst: list,
            encoded_name_lst: Optional[list] = None) -> None:
        '''
        Turn each categorical column into a new feature with propotion of churn for each category

        Args:
            category_lst (list): List of columns containing categorical features
            encoded_name_lst (Optional: list): List of names for encoded features from category_lst.
                If passed must be the same length as category_lst and in corresponding order.
        '''
        for i, feature in enumerate(category_lst):
            feature_groups = self.df.groupby(feature).mean()['Churn']
            if encoded_name_lst is None:
                encoded_name = f'{feature}_Churn'
            else:
                encoded_name = encoded_name_lst[i]

            encoded_feature_lst = []
            for val in self.df[feature]:
                encoded_feature_lst.append(feature_groups.loc[val])
            self.df[encoded_name] = encoded_feature_lst

    def prepare_training_data(
            self,
            training_features_lst: list,
            target: str,
            test_size: float,
            random_state: int) -> None:
        '''
        Split data into training and test datasets ready for training.

        Args:
            training_features_lst (list): List of columns used as input features for training
            target (str): Name of column to use as target in training
            test_size (float): Proportion of data to set aside for testing
            random_state (int): Set the random state for reproducible workflow
        '''
        X = self.df[training_features_lst]
        y = self.df[target]

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state)

    def train_models(
            self,
            rf_param_grid: dict,
            num_cv_folds: int,
            random_state: int) -> None:
        '''
        Train the random forest and logistic regression models.

        Args:
            rf_param_grid (dict): Parameter grid for grid search of random forest hyperparameters
            num_cv_folds (int): Number of cross-validation folds in training
            random_state (int): Set the random state for reproducible workflow
        '''
        rf_classifier = RandomForestClassifier(random_state=random_state)
        self.random_forest_gridcv = GridSearchCV(
            estimator=rf_classifier, param_grid=rf_param_grid, cv=num_cv_folds)
        # The job failed with the default Logistic Regression solver, so was
        # set to liblinear below
        self.logistic_classifier = LogisticRegression(
            random_state=random_state, solver='liblinear')
        print('Starting Random Forest grid search...')
        self.random_forest_gridcv.fit(X=self.X_train, y=self.y_train)
        print('Finished Random Forest grid search. Starting training Logistic Regression', end="")
        print(' classifier...')
        self.logistic_classifier.fit(X=self.X_train, y=self.y_train)
        print('Finished training Logistic Regression classifier')

    def make_predictions(self) -> None:
        '''
        Use trained models to predict training and test data.

        Args:
            None
        '''
        self.y_train_preds_rf = self.random_forest_gridcv.best_estimator_.predict(
            self.X_train)
        self.y_test_preds_rf = self.random_forest_gridcv.best_estimator_.predict(
            self.X_test)
        self.y_train_preds_lr = self.logistic_classifier.predict(self.X_train)
        self.y_test_preds_lr = self.logistic_classifier.predict(self.X_test)

    def classification_report_image(self, output_path: str) -> None:
        '''
        Produces classification reports for training and testing results and stores report as image
        in images folder.

        Args:
            output_path (str): Path of directory to save the outputs.
        '''
        report_rf_train = classification_report(
            y_true=self.y_train,
            y_pred=self.y_train_preds_rf,
            output_dict=True)
        output_filepath = f'{output_path}/rf_train_report.png'
        dfi.export(pd.DataFrame(report_rf_train).T, output_filepath)

        report_rf_test = classification_report(
            y_true=self.y_test,
            y_pred=self.y_test_preds_rf,
            output_dict=True)
        output_filepath = f'{output_path}/rf_test_report.png'
        dfi.export(pd.DataFrame(report_rf_test).T, output_filepath)

        report_lr_train = classification_report(
            y_true=self.y_train,
            y_pred=self.y_train_preds_lr,
            output_dict=True)
        output_filepath = f'{output_path}/lr_train_report.png'
        dfi.export(pd.DataFrame(report_lr_train).T, output_filepath)

        report_lr_test = classification_report(
            y_true=self.y_test,
            y_pred=self.y_test_preds_lr,
            output_dict=True)
        output_filepath = f'{output_path}/lr_test_report.png'
        dfi.export(pd.DataFrame(report_lr_test).T, output_filepath)

    def output_roc_curves(self, output_path: str) -> None:
        '''
        Produces receiver operating characteristic (ROC) curve plots for training and testing
        results and stores figures as images in the output directory.

        Args:
            output_path (str): Path of directory to save the outputs.
        '''
        # Training data
        plt.figure(figsize=(15, 8))
        ax = plt.gca()
        plot_roc_curve(self.random_forest_gridcv.best_estimator_,
                       self.X_train, self.y_train, ax=ax, alpha=0.8)
        plot_roc_curve(
            self.logistic_classifier,
            self.X_train,
            self.y_train,
            ax=ax,
            alpha=0.8)
        img_path = f'{output_path}/training_roc_curve_result.png'
        plt.savefig(fname=img_path, dpi=200, format='png')

        # Test data
        plt.figure(figsize=(15, 8))
        ax = plt.gca()
        plot_roc_curve(self.random_forest_gridcv.best_estimator_,
                       self.X_test, self.y_test, ax=ax, alpha=0.8)
        plot_roc_curve(
            self.logistic_classifier,
            self.X_test,
            self.y_test,
            ax=ax,
            alpha=0.8)
        img_path = f'{output_path}/test_roc_curve_result.png'
        plt.savefig(fname=img_path, dpi=200, format='png')

    def output_feature_importance_plot(self, output_path: str) -> None:
        '''
        Creates and stores the feature importances from the random forest best model in output_path

        Args:
            output_path (str): Path of directory to save the outputs.
        '''
        importances = self.random_forest_gridcv.best_estimator_.feature_importances_
        # Sort feature importances in descending order
        indices = np.argsort(importances)[::-1]

        # Rearrange feature names so they match the sorted feature importances
        names = [self.X_train.columns[i] for i in indices]

        # Create plot
        plt.figure(figsize=(20, 10))
        plt.title("Feature Importance")
        plt.ylabel('Importance')
        plt.bar(range(self.X_train.shape[1]), importances[indices])
        plt.xticks(range(self.X_train.shape[1]), names, rotation=90)

        img_path = f'{output_path}/rf_feature_importances.png'
        plt.savefig(fname=img_path, dpi=200, format='png')

    def save_models(self, output_path: str) -> None:
        '''
        Saves the models in output_path

        Args:
            output_path (str): Path of directory to save the models.
        '''
        joblib.dump(
            self.random_forest_gridcv.best_estimator_,
            f'{output_path}/rfc_model.pkl')
        joblib.dump(
            self.logistic_classifier,
            f'{output_path}/logistic_model.pkl')


if __name__ == '__main__':
    # Load config file
    with open("config.yaml", "r", encoding='utf-8') as ymlfile:
        config = Box(yaml.safe_load(ymlfile))

    # Plot settings
    matplotlib.rc('font', size=config.matplotlib.small_size)
    matplotlib.rc('axes', titlesize=config.matplotlib.medium_size)
    matplotlib.rc('axes', labelsize=config.matplotlib.medium_size)
    matplotlib.rc('xtick', labelsize=config.matplotlib.small_size)
    matplotlib.rc('ytick', labelsize=config.matplotlib.small_size)
    matplotlib.rc('figure', titlesize=config.matplotlib.large_size)

    churn = CustomerChurn()
    churn.import_data(filepath=config.data.raw_filepath)
    churn.create_churn_feature()
    churn.perform_eda(histogram_features=config.eda.plot_histograms,
                      output_dir=config.eda.output_dir)
    churn.perform_feature_engineering(
        category_lst=config.feature_engineering.categorical_columns)
    churn.prepare_training_data(
        training_features_lst=config.prepare_data.training_features,
        target=config.prepare_data.target,
        test_size=config.prepare_data.test_size,
        random_state=config.prepare_data.random_state)
    churn.train_models(rf_param_grid=config.models.random_forest.param_grid,
                       num_cv_folds=config.models.random_forest.num_cv_folds,
                       random_state=config.models.random_state)
    churn.make_predictions()
    churn.classification_report_image(
        output_path=config.classification_report.output_path)
    churn.output_roc_curves(
        output_path=config.classification_report.output_path)
    churn.output_feature_importance_plot(
        output_path=config.classification_report.output_path)
    churn.save_models(output_path=config.models.output_path)
