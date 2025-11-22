import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from huggingface_hub import login, HfApi, hf_hub_download
#from google.colab import userdata
#tok = userdata.get('HF_TOKEN')
tok = os.getenv("HF_TOKEN")
api = HfApi(token=tok)
repo_id = "Cruise949/tourism-predict"
filename = "tour.csv"
repo_type = "dataset"
dataset = "hf://datasets/Cruise949/tourism-predict/tour.csv"
df = pd.read_csv(hf_hub_download(repo_id=repo_id, filename=filename, repo_type=repo_type))

y = df['ProdTaken']
X = df.drop('ProdTaken', axis=1)
# Remove unwanted columns
X.drop('Unnamed: 0', axis=1, inplace=True)
X.drop('CustomerID', axis=1, inplace=True)
# Correct data
X['Gender'].replace("Fe Male","Female", inplace=True)
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
x_train.to_csv("tourism_project/data/x_train.csv", index=False)
x_test.to_csv("tourism_project/data/x_test.csv", index=False)
y_train.to_csv("tourism_project/data/y_train.csv", index=False)
y_test.to_csv("tourism_project/data/y_test.csv", index=False)
files = ['x_train.csv', 'x_test.csv', 'y_train.csv', 'y_test.csv']
for file in files:
    file = "tourism_project/data/" + file
    api.upload_file(
        path_or_fileobj= file,
        path_in_repo= file.split("/")[-1],
        repo_id="Cruise949/tourism-predict",
        repo_type="dataset",
    )
