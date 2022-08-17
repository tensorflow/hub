import tensorflow as tf
from keras_layer import KerasLayer
from huggingface_hub import snapshot_download, create_repo, HfApi, HfFolder


def pull_from_hub(repo_id: str,
    revision=None,
    local_files_only=False,
    allow_regex=None,
    ignore_regex=None,
    trainable=False,
):
    model_path = snapshot_download(repo_id=repo_id,
                                   revision=revision,
                                   local_files_only=local_files_only,
                                   allow_regex=allow_regex,
                                   ignore_regex=ignore_regex,
                                )
    return KerasLayer(handle=model_path, 
                      trainable=trainable,
                    )

def push_to_hub(model,
                repo_id, 
                folder_path,
                path_in_repo="./",
                token=None,
              ):
  
    if token is None:
        token = HfFolder.get_token()

    if token is None:
        raise ValueError(
            "You must login to the Hugging Face hub on this computer by typing"
            " `huggingface-cli login` and entering your credentials, "
            " if you are on a notebook environment you may use the `notebook_login` function "
            " Alternatively, you can pass your own token as the `token` argument."
        )
  
    # Persist model to local file system
    tf.saved_model.save(obj=model, 
                        export_dir=folder_path)
    
    # Create HF Hub repository if it doesn't exists
    create_repo(repo_id=repo_id,
                token=token,
                exist_ok=True)

    # Upload all relevant model files to HF Hub
    HfApi().upload_folder(repo_id=repo_id,
                          folder_path=folder_path,
                          path_in_repo=path_in_repo,
                          token=token
                        )