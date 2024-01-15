import boto3
from botocore import UNSIGNED
from botocore.client import Config
import os


"""
Folders:
/train_data - ~14TB, ~90k wav files
/val_data - ~137GB, 864 wav files

/train_data_semantic_tokenized - ~56GB, ~90k pt files
/val_data_semantic_tokenized - ~573MB, 864 pt files

/train_data_encodec_tokenized - ~687GB, ~90k pt files
/val_data_encodec_tokenized - ~7GB, 864 pt files

"""

folders = ['val_data_encodec_tokenized'] # ADD FOLDERS HERE



s3 = boto3.client('s3')

def download_s3_folder(bucket_name, s3_folder, local_dir=None):
    """
    Download the contents of a folder directory
    Args:
        bucket_name: the name of the s3 bucket
        s3_folder: the folder path in the s3 bucket
        local_dir: the local directory to download to (optional)
    """
    keys = []
    dirs = []
    next_token = ''
    base_kwargs = {
        'Bucket': bucket_name,
        'Prefix': s3_folder,
    }
    while next_token is not None:
        kwargs = base_kwargs.copy()
        if next_token != '':
            kwargs.update({'ContinuationToken': next_token})
        results = s3.list_objects_v2(**kwargs)
        contents = results.get('Contents')
        for i in contents:
            k = i.get('Key')
            if k[-1] != '/':
                keys.append(k)
            else:
                dirs.append(k)
        next_token = results.get('NextContinuationToken')
    for d in dirs:
        dest_pathname = os.path.join(local_dir, d)
        if not os.path.exists(os.path.dirname(dest_pathname)):
            os.makedirs(os.path.dirname(dest_pathname))
    for k in keys:
        dest_pathname = os.path.join(local_dir, k)
        if not os.path.exists(os.path.dirname(dest_pathname)):
            os.makedirs(os.path.dirname(dest_pathname))
        s3.download_file(bucket_name, k, dest_pathname)


for folder in folders:
    download_s3_folder('tiny-narrations-instruct', folder, f'/YOUR_PATH_HERE/{folder}')
