from   huggingface_hub import HfApi
import os
import sys

try:
  path = sys.argv[1].rstrip('/')
  if len(sys.argv) == 3:
    model_name = sys.argv[2]
  else:
    model_name = os.path.basename(path)
except:
    print('You should run this with [path-to-upload] [optional-model-name]')
    sys.exit(1)


api = HfApi()
try:
    api.create_repo(
        repo_id=f"augmxnt/{model_name}",
        # repo_type="model",
        # private=True,
    )
                  
except:
    pass

os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = "1"
api.upload_folder(
    folder_path=path,
    repo_id=f"augmxnt/{model_name}",
    repo_type='model',
)
