import os

from huggingface_hub import HfApi, HfFileSystem

hf_client = HfApi(token=os.environ.get('HF_TOKEN'))
hf_fs = HfFileSystem(token=os.environ.get('HF_TOKEN'))

_REPOSITORY = os.environ['REMOTE_REPOSITORY']


class GenericException:
    pass
