import os
import pandas as pd
import logging
import mlflow

from mlflow.entities import ViewType
from mlflow.tracking import MlflowClient

def predict(test_file_path, preprocessing_pipeline_name, logistic_reg_model_name, prediction_file_path):
    """
    Make prediction on test data with saved model.

    :param test_file_path: path to test data.
    :param preprocessing_pipeline_name: name of the preprocessing pipeline saved with mlflow.
    :param logistic_reg_model_name: name of logistic reg model to be saved with mlflow.
    :param prediction_file_path: path to prediction data.

    :return: None
    """
    test_df = pd.read_csv(test_file_path)

    current_run = mlflow.active_run()

    logging.info("Loading preprocessing pipeline")
    # TODO 4 : charger le preprocessing pipeline dans le dossier du run actif (allez chercher l’artifact_uri dans
    #  les infos du run actif). Utilisez os.path.join pour une gestion de fichiers cross-OS.
    #preprocessing_pipeline = NotImplementedError()
    preprocessing_pipeline, pp_run_id = load_latest_preprocessing_pipeline(
        preprocessing_pipeline_name, files.MLFLOW_EXPERIMENT_NAME)
    if pp_run_id != current_run_id:
        logging.info("Using preprocessing pipeline saved in a previous run")
        mlflow.log_param("preprocessing_pipeline_run_id", pp_run_id)
        
    preprocessed_test = preprocessing_pipeline.transform(test_df)

    logging.info("Loading trained model")
    # TODO 5 : charger le modèle dans le dossier du run actif.
    #logistic_reg = NotImplementedError()
    logistic_reg = mlflow.sklearn.load_model(
        os.path.join(mlflow.active_run().info.artifact_uri, logistic_reg_model_name))

    logging.info(f"Make predictions with {logistic_reg_model_name}")
    y_pred = logistic_reg.predict(preprocessed_test)

    test_df["prediction"] = y_pred

    logging.info("Saving prediction results")
    test_df.to_csv(prediction_file_path, index=False)
