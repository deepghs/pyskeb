import os

from huggingface_hub import HfApi


def repo_squash():
    hf_client = HfApi(token=os.environ.get('HF_TOKEN'))
    remote_repo = os.environ.get('REMOTE_REPOSITORY')
    hf_client.super_squash_history(repo_id=remote_repo, repo_type='dataset', commit_message='Squashed!')


if __name__ == '__main__':
    repo_squash()
