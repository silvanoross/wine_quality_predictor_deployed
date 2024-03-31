# The data set used in this example is from http://archive.ics.uci.edu/ml/datasets/Wine+Quality
# P. Cortez, A. Cerdeira, F. Almeida, T. Matos and J. Reis.
# Modeling wine preferences by data mining from physicochemical properties. In Decision Support Systems, Elsevier, 47(4):547-553, 2009.

# import all required modules
import os
import warnings
import sys

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet # most simple linear model (hybrid of L1 and L2 penalties)
from urllib.parse import urlparse
import mlflow
import mlflow.sklearn

# how we will log our results for use later
import logging

logging.basicConfig(level = logging.WARN)
logger = logging.getLogger(__name__)

# must write methods for each task
# here, the inputs are the actual value and predictions and we get the output of prediction metrics
# treating as regression because quality is 1-5
def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2

# main script to run in command line
if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)

    # read the wine-quality csv file from the URL
    csv_url = (
        "http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv" # reading in the same dataset
    )
    try:
        data = pd.read_csv(csv_url, sep = ";")
    except Exception as e:
        logger.exception(
            "Unable to download training & test CSV, check your internet connection. Error: %s", e
        )

    # split the data into training and test sets. (0.75, 0.25) split.
    train, test = train_test_split(data)

    # the predicted column is "quality" which is a scalar from [3, 9]
    train_x = train.drop(["quality"], axis = 1)
    test_x = test.drop(["quality"], axis = 1)
    train_y = train[["quality"]]
    test_y = test[["quality"]]

    # what kind of parameters do we want to supply, when we want to call this python code, for the elastic net
    alpha = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5 # if user supplies argument, but default is .5 if not supplied
    l1_ratio = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5
    experiment_name = str(sys.argv[3]) if len(sys.argv) > 3 else "predict_wine_quality" # user can specify their own otherwise, give them default

    mlflow.set_experiment(experiment_name)
    # mlflow.autolog()
    with mlflow.start_run():
        
        # first activate run process and store
        run = mlflow.active_run()
        experiment = mlflow.get_experiment(run.info.experiment_id)
        print("Experiment ID: \"{}\"".format(run.info.experiment_id))
        print("Experiment name: \"{}\"".format(experiment.name))
        print("Run ID: \"{}\"".format(run.info.run_id))
        
        # training portion we use an elastic net linear regression alogrithm
        lr = ElasticNet(alpha = alpha, l1_ratio = l1_ratio, random_state = 42)
        lr.fit(train_x, train_y)
        
        # make predictions
        predicted_qualities = lr.predict(test_x)
        
        # call eval metrics functions and get results
        (rmse, mae, r2) = eval_metrics(test_y, predicted_qualities)

        # print info to show metrics
        print("Using alpha = {:0.2f}, l1_ratio = {:0.2f} we get the following metrics:".format(alpha, l1_ratio))
        print("  metric RMSE: {:6.2f}".format(rmse))
        print("  metric MAE: {:6.2f}".format(mae))
        print("  metric R-squared: {:0.2f}".format(r2))

        # log the parameters, store all the run information, and record which parameters for each run
        # after all tries of machine learning training, under what parameters is the model accuracy
        mlflow.log_param("alpha", alpha) # parameter, controls regularization amount
        mlflow.log_param("l1_ratio", l1_ratio) # split between L1 and L2 penalties
        
        # log metrics
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("mae", mae)

        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        # model registry does not work with file store
        if tracking_url_type_store != "file":

            # register the model
            mlflow.sklearn.log_model(lr, "model", registered_model_name = "ElasticnetWineModel")
        else:
            mlflow.sklearn.log_model(lr, "model")
