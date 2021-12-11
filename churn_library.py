# library doc string
"""
This module

@author: Rob Smith
date: 10th December 2021

"""

# import libraries
import os
import shap
import joblib
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import yaml
from typing import Optional
from box import Box
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import plot_roc_curve, classification_report

class CustomerChurn:
    """
    This class enables a model to be trained from scratch for predicting customer churn.
    
    Attributes:
        df (pd.DataFrame): DataFrame storing data
    """
    def __init__(self):
        self.df = None
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None

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
        self.df['Churn'] = self.df['Attrition_Flag'].apply(lambda val: 0 if val == "Existing Customer" else 1)
    
    def plot_distribution(self, feature: str, output_dir: str) -> None:
        '''
        Plot and save histogram of specified feature in output_dir

        Args:
            feature (str): Name of feature to plot
        '''
        fig = plt.figure(figsize=(20,10))
        ax = sns.histplot(data=self.df[feature])
        ax.set_title(f'Histogram of {feature}')
        filename = f'{feature}_distribution.png'.lower()
        img_path = os.path.join(output_dir, filename)
        plt.savefig(fname=img_path, dpi=200, format='png')
    
    def perform_eda(self, histogram_features: list, output_dir: str) -> None:
        '''
        Perform EDA on the input data and save figfures to the output_dir

        Args:
            histogram_features (list): List of features to plot as histograms
            output_dir (str): Directory to store the output images
        '''
        # Histograms
        for feature in histogram_features:
            self.plot_distribution(feature=feature, output_dir=output_dir)
            
        # Heatmap
        plt.figure(figsize=(20,10)) 
        sns.heatmap(self.df.corr(), annot=False, cmap='Dark2_r', linewidths = 2)
        img_path = os.path.join(output_dir, 'heatmap.png')
        plt.savefig(fname=img_path, dpi=200, format='png')
        
    def perform_feature_engineering(self, category_lst: list, encoded_name_lst: Optional[list] =  None) -> None:
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







# def perform_feature_engineering(df, response):
#     '''
#     input:
#               df: pandas dataframe
#               response: string of response name [optional argument that could be used for naming variables or index y column]

#     output:
#               X_train: X training data
#               X_test: X testing data
#               y_train: y training data
#               y_test: y testing data
#     '''

# def classification_report_image(y_train,
#                                 y_test,
#                                 y_train_preds_lr,
#                                 y_train_preds_rf,
#                                 y_test_preds_lr,
#                                 y_test_preds_rf):
#     '''
#     produces classification report for training and testing results and stores report as image
#     in images folder
#     input:
#             y_train: training response values
#             y_test:  test response values
#             y_train_preds_lr: training predictions from logistic regression
#             y_train_preds_rf: training predictions from random forest
#             y_test_preds_lr: test predictions from logistic regression
#             y_test_preds_rf: test predictions from random forest

#     output:
#              None
#     '''
#     pass


# def feature_importance_plot(model, X_data, output_pth):
#     '''
#     creates and stores the feature importances in pth
#     input:
#             model: model object containing feature_importances_
#             X_data: pandas dataframe of X values
#             output_pth: path to store the figure

#     output:
#              None
#     '''
#     pass

# def train_models(X_train, X_test, y_train, y_test):
#     '''
#     train, store model results: images + scores, and store models
#     input:
#               X_train: X training data
#               X_test: X testing data
#               y_train: y training data
#               y_test: y testing data
#     output:
#               None
#     '''
#     pass

if __name__ == '__main__':
    # Load config file
    with open("config.yaml", "r") as ymlfile:
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
    churn.perform_feature_engineering(category_lst=config.feature_engineering.categorical_columns)
    print(churn.df.columns)