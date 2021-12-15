# udacity-nd-mlops-project1
This repo contains the code developed for the project from the first course in the [Udacity Machine Learning DevOps Engineer nanodegree](https://www.udacity.com/course/machine-learning-dev-ops-engineer-nanodegree--nd0821).

## Project: Predict Customer Churn
The project assigned to us is to write clean code to predict customer churn (the rate at which customers stop doing buesiness with a company). We are provided wth some initial code in the notebook named **churn_notebook.ipynb** which takes us from raw data to predictions made by a Random Forest and Logistic Regression model. Our task is to take the code from this notebook and turn it into clean code in the form of a script in the **churn.py** file. The code should follow PEP8 standards as far as possible

**Note:** We could improve on things that are done in the preprocessing and modeling, but that is not the main objective of this project. Here the goal is to turn what has been given to us into clean code.

The main files in this repo are:
-**churn_library.py**: Contains the main code which has been structured into a class named `CustomerChurn`. This has a number of methods that allow us to load the raw data, process it, train the models and evaluate teh results. It can be run as a script.
-**churn_script_logging_and_tests.py**: Has a number of tests for the methods contained in `churn.py`.

## Running Files
You can follow the instructions below to run these scripts yourself:

1. Clone repository
Make a clone of this repo to your local workspace
`git clone https://github.com/robsmith155/udacity-nd-mlops-project1.git`

2. Create Python virtual environment
You need to create a virtual enviornment using something like Conda or virtualenv. Note that my tests were run using Python 3.8.10.

3. Install dependencies
Install the packages required by this code (make sure you have first activated your virtual environment):
`pip install -r requirements.txt`

4. Run scripts
Change directory to the cloned repo. To run the `churn.py` script, simply execute the following in the command prompt/terminal:
`python3 churn.py`

This will load the raw data, run the preprocessing, train and save the models as well as output some QC plots of the data and evaluation performance. These are stores in the `images` directory.

To run the test and logging script:
`python3 churn_scriptlogging_and_tests.py`


