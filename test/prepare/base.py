import os

from huggingface_hub import HfApi, HfFileSystem

hf_client = HfApi(token=os.environ.get('HF_TOKEN'))
hf_fs = HfFileSystem(token=os.environ.get('HF_TOKEN'))

_REPOSITORY = os.environ['REMOTE_REPOSITORY']


class GenericException(Exception):
    pass


def _ensure_repository():
    if not hf_client.repo_exists(repo_id=_REPOSITORY, repo_type='dataset'):
        hf_client.create_repo(
            repo_id=_REPOSITORY,
            repo_type='dataset',
            exist_ok=True,
            private=True,
        )
        lines = hf_fs.read_text(f'datasets/{_REPOSITORY}/.gitattributes').splitlines(keepends=False)
        lines = [*filter(bool, lines), 'archived.json filter=lfs diff=lfs merge=lfs -text']
        hf_fs.write_text(f'datasets/{_REPOSITORY}/.gitattributes', os.linesep.join(lines))
