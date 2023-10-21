import logging
import os.path
import zipfile
from contextlib import contextmanager
from datetime import datetime

from hbutils.system import TemporaryDirectory
from huggingface_hub import CommitOperationAdd, CommitOperationCopy, CommitOperationDelete
from huggingface_hub import hf_hub_url
from tqdm.auto import tqdm

from pyskeb.utils.download import download_file
from .base import hf_fs, _REPOSITORY, hf_client


@contextmanager
def repack_zips():
    with TemporaryDirectory() as td:
        dd_dir = os.path.join(td, 'origin')
        os.makedirs(dd_dir, exist_ok=True)

        fns = []
        for file in tqdm(hf_fs.glob(f'datasets/{_REPOSITORY}/unarchived/*.zip')[:4]):
            filename = os.path.basename(file)
            fns.append(filename)

            with TemporaryDirectory() as ctd:
                zip_file = os.path.join(ctd, filename)
                download_file(
                    hf_hub_url(repo_id=_REPOSITORY, repo_type='dataset', filename=f'unarchived/{filename}'),
                    zip_file,
                )
                with zipfile.ZipFile(zip_file, 'r') as zf:
                    zf.extractall(dd_dir)

        zip_file = os.path.join(td, 'package.zip')
        written = False
        with zipfile.ZipFile(zip_file, 'w') as zf:
            for root, dirs, files in os.walk(dd_dir):
                for file in files:
                    filename = os.path.join(dd_dir, root, file)
                    relname = os.path.relpath(filename, dd_dir)
                    zf.write(filename, relname)
                    written = True

        if written:
            yield zip_file, fns
        else:
            yield None, fns


def _timestamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")


def repack_all():
    with repack_zips() as (zip_file, fns):
        if zip_file is None:
            logging.info('No files to repack, skipped.')
            return

        package_name = f'pack_{_timestamp()}.zip'
        logging.info(f'Creating new pack {package_name!r} ...')
        operations = []
        operations.append(CommitOperationAdd(
            path_or_fileobj=zip_file,
            path_in_repo=f'packs/{package_name}'
        ))
        for fn in fns:
            operations.append(CommitOperationCopy(
                src_path_in_repo=f'unarchived/{fn}',
                path_in_repo=f'archived/{fn}',
            ))
            operations.append(CommitOperationDelete(
                path_in_repo=f'unarchived/{fn}',
            ))

        hf_client.create_commit(
            repo_id=_REPOSITORY,
            repo_type='dataset',
            operations=operations,
            commit_message=f'Create new package {package_name!r}.'
        )
