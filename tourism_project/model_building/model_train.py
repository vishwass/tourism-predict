import pandas as pd
import sklearn
import os
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import classification_report,precision_score,accuracy_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import RandomizedSearchCV
from sklearn.compose import make_column_transformer
import joblib
import mlflow
from huggingface_hub import login, HfApi, hf_hub_download
tok = os.getenv("HF_TOKEN")
#from google.colab import userdata
#tok = userdata.get('HF_TOKEN')

api = HfApi(token = tok)

mlflow.set_tracking_uri("https://supranaturalistic-sook-validatory.ngrok-free.dev/")
mlflow.set_experiment("Tourism Project-mlops1")


numerical_features = [
    'Age',
    'DurationOfPitch',
    'NumberOfPersonVisiting',
    'NumberOfFollowups' ,
    'PreferredPropertyStar' ,
    'NumberOfTrips',
    'Passport',
    'PitchSatisfactionScore',
    'OwnCar',
    'NumberOfChildrenVisiting',
    'MonthlyIncome'
]
categorical_features = [
  'CityTier',
  'Occupation',
  'TypeofContact',
  'Gender',
  'ProductPitched'
]
target = 'ProdTaken'
x_train_fname = "x_train.csv"
x_test_fname= "x_test.csv"
y_train_fname= "y_train.csv"
y_test_fname= "y_test.csv"
repo_id = "Cruise949/tourism-predict"
repo_type = "dataset"
x_train  = pd.read_csv(hf_hub_download(repo_id=repo_id, filename=x_train_fname, repo_type=repo_type))
x_test = pd.read_csv(hf_hub_download(repo_id=repo_id, filename=x_test_fname, repo_type=repo_type))
y_train = pd.read_csv(hf_hub_download(repo_id=repo_id, filename=y_train_fname, repo_type=repo_type))
y_test = pd.read_csv(hf_hub_download(repo_id=repo_id, filename=y_test_fname, repo_type=repo_type))

preprocessor = make_column_transformer(
    (StandardScaler(), numerical_features), (OneHotEncoder(handle_unknown='ignore'), categorical_features)
)

param_grid = {
    'xgbclassifier__n_estimators': [50, 75, 100, 200, 300],
    'xgbclassifier__learning_rate': [0.01, 0.1, 0.2,0.3],
    'xgbclassifier__max_depth': [3, 4, 5,8, 10,  12],
    'xgbclassifier__colsample_bytree': [0.3, 0.5, 0.7, 1.0],
    'xgbclassifier__colsample_bylevel': [0.3, 0.5, 0.7, 1.0],
    'xgbclassifier__reg_lambda': [0.1, 0.4, 0.7, 1.0],
}
class_weight = y_train.value_counts()[0]/ y_train.value_counts()[1]
print("class weight:",class_weight)

model = xgb.XGBClassifier(scale_pos_weight=class_weight, random_state=1)

pipeline = make_pipeline( preprocessor, model)

#print(pipeline.get_feature_names_out())

with mlflow.start_run():
    random_cv= RandomizedSearchCV(pipeline, param_grid, cv=5, n_jobs=-1)
    random_cv.fit(x_train, y_train)
    #pipeline.fit(x_train, y_train)
    #print(pipeline[:-1].get_feature_names_out())

    best_params = random_cv.best_params_
    mlflow.log_params(best_params)
    best_model = random_cv.best_estimator_
    print(best_model.feature_names_in_)
    #print(best_model.get_params())
    y_pred = best_model.predict(x_test)
    train_report = classification_report(y_train, best_model.predict(x_train), output_dict=True)
    test_report = classification_report(y_test, y_pred, output_dict=True)
    print("Train Classification Report:\n", train_report)
    print("Test Classification Report:\n", test_report)

    mlflow.log_metrics({
        "train_accuracy": train_report['accuracy'],
        "train_precision": train_report['1']['precision'],
        "test_accuracy": test_report['accuracy'],
        "test_precision": test_report['1']['precision']

    })

    mlflow.sklearn.log_model(best_model, "model")

    model_path = "tourism_project/model_building/model.joblib"
    joblib.dump(best_model, model_path)

    api.upload_file(
      path_or_fileobj= model_path,
      path_in_repo= model_path.split("/")[-1],
      repo_id="Cruise949/tourism-predict",
      repo_type="model",
    )

