from huggingface_hub import HfApi
import os

#from google.colab import userdata
#tok = userdata.get('HF_TOKEN')
tok = os.getenv("HF_TOKEN")
api = HfApi(token=tok)

repo_id = "Cruise949/tourism-predict"
repo_type = "space"
foldername = "tourism_project/deployment"
api.upload_folder(repo_id=repo_id,
                repo_type=repo_type,
                folder_path=foldername)


