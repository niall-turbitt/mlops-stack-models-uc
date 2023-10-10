# Databricks notebook source
##################################################################################
# Batch Inference Notebook
#
# This notebook is an example of applying a model for batch inference against an input delta table,
# It is configured and can be executed as the batch_inference_job in the batch_inference_job workflow defined under
# ``mlops_stack_models_uc/databricks-resources/batch-inference-workflow-resource.yml``
#
# Parameters:
#
#  * env (optional)  - String name of the current environment (dev, staging, or prod). Defaults to "dev"
#  * input_table_name (required)  - Delta table name containing your input data.
#  * output_table_name (required) - Delta table name where the predictions will be written to.
#                                   Note that this will create a new version of the Delta table if
#                                   the table already exists
#  * model_name (required) - The name of the model to be used in batch inference.
##################################################################################


# List of input args needed to run the notebook as a job.
# Provide them via DB widgets or notebook arguments.
#
# Name of the current environment
dbutils.widgets.dropdown("env", "dev", ["dev", "staging", "prod"], "Environment Name")
# A Hive-registered Delta table containing the input features.
dbutils.widgets.text("input_table_name", "", label="Input Table Name")
# Delta table to store the output predictions.
dbutils.widgets.text("output_table_name", "", label="Output Table Name")
# Batch inference model name
dbutils.widgets.text("model_name", "", label="Model Name")

# COMMAND ----------

import os

notebook_path =  '/Workspace/' + os.path.dirname(dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get())
%cd $notebook_path

# COMMAND ----------

# MAGIC %pip install -r ../../../requirements.txt

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import sys
import os
notebook_path =  '/Workspace/' + os.path.dirname(dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get())
%cd $notebook_path
%cd ..
sys.path.append("../..")

# COMMAND ----------

# DBTITLE 1,Define input and output variables
# from utils import get_deployed_model_stage_for_env
from utils import get_deployed_model_alias_for_env

env = dbutils.widgets.get("env")
input_table_name = dbutils.widgets.get("input_table_name")
input_table_name = "niall_dev.schema_name.taxi_scoring_sample"

output_table_name = dbutils.widgets.get("output_table_name")
output_table_name = "niall_dev.schema_name.taxi_batch_inference"

model_name = dbutils.widgets.get("model_name")
model_name = "niall_dev.schema_name.mlops-stack-models-uc-model"

assert input_table_name != "", "input_table_name notebook parameter must be specified"
assert output_table_name != "", "output_table_name notebook parameter must be specified"
assert model_name != "", "model_name notebook parameter must be specified"
# stage = get_deployed_model_stage_for_env(env)
alias = get_deployed_model_alias_for_env(env)
# model_uri = f"models:/{model_name}/{stage}"
model_uri = f"models:/{model_name}@{alias}"

# COMMAND ----------

# # Get model version from stage
# from mlflow import MlflowClient

# model_version_infos = MlflowClient().search_model_versions("name = '%s'" % model_name)
# model_version = max(
#     int(version.version)
#     for version in model_version_infos
#     if version.current_stage == stage
# )

# Get model version from alias
from mlflow import MlflowClient

client = MlflowClient(registry_uri="databricks-uc")
model_version = client.get_model_version_by_alias(model_name, alias).version

# COMMAND ----------

# Get datetime
from datetime import datetime

ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# COMMAND ----------
# DBTITLE 1,Load model and run inference

from predict import predict_batch

predict_batch(spark, model_uri, input_table_name, output_table_name, model_version, ts)
dbutils.notebook.exit(output_table_name)
