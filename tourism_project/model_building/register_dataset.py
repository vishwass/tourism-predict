from huggingface_hub.utils import RepositoryNotFoundError
from huggingface_hub import HfApi
import os
from huggingface_hub import HfApi
#from google.colab import userdata
#tok = userdata.get('HF_TOKEN')
tok = os.getenv("HF_TOKEN")
api = HfApi(token=tok)

api.upload_folder(
    folder_path="tourism_project/data",
    repo_id="Cruise949/tourism-predict",
    repo_type="dataset",
)

