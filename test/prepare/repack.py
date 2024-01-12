import json
import logging
import os.path
import zipfile
from contextlib import contextmanager
from datetime import datetime

import pandas as pd
from hbutils.scale import size_to_bytes_str
from hbutils.system import TemporaryDirectory
from huggingface_hub import CommitOperationAdd, CommitOperationDelete
from huggingface_hub import hf_hub_url
from huggingface_hub.utils import HfHubHTTPError
from tqdm.auto import tqdm

from pyskeb.utils.download import download_file
from .base import _REPOSITORY, hf_client, hf_fs, _ensure_repository


@contextmanager
def repack_zips():
    with TemporaryDirectory() as td:
        dd_dir = os.path.join(td, 'origin')
        os.makedirs(dd_dir, exist_ok=True)

        fns = []
        for file in tqdm(hf_fs.glob(f'datasets/{_REPOSITORY}/unarchived/*.zip')):
            filename = os.path.basename(file)
            fns.append(filename)

            with TemporaryDirectory() as ctd:
                zip_file = os.path.join(ctd, filename)
                download_file(
                    hf_hub_url(repo_id=_REPOSITORY, repo_type='dataset', filename=f'unarchived/{filename}'),
                    zip_file,
                    headers={'Authorization': f'Bearer {os.environ["HF_TOKEN"]}'},
                )
                with zipfile.ZipFile(zip_file, 'r') as zf:
                    try:
                        zf.extractall(dd_dir)
                    except OSError as err:
                        logging.warning(repr(err))

        zip_file = os.path.join(td, 'package.zip')
        written = False
        with zipfile.ZipFile(zip_file, 'w') as zf:
            for root, dirs, files in os.walk(dd_dir):
                for file in files:
                    filename = os.path.join(dd_dir, root, file)
                    relname = os.path.relpath(filename, dd_dir)
                    zf.write(filename, relname)
                    os.remove(filename)
                    written = True

        if written:
            yield zip_file, fns
        else:
            yield None, fns


def _make_records():
    if not hf_fs.exists(f'datasets/{_REPOSITORY}/index.json'):
        retval = []
        for pack in hf_fs.glob(f'datasets/{_REPOSITORY}/packs/*.zip'):
            filename = os.path.basename(pack)
            _info = hf_fs.info(f'datasets/{_REPOSITORY}/packs/{filename}')
            size = _info['size']
            retval.append({'filename': filename, 'size': size})
        return retval
    else:
        return json.loads(hf_fs.read_text(f'datasets/{_REPOSITORY}/index.json'))


def _timestamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")


def repack_all():
    _ensure_repository()
    if hf_fs.exists(f'datasets/{_REPOSITORY}/archived.json'):
        archived_resource_ids = json.loads(hf_fs.read_text(f'datasets/{_REPOSITORY}/archived.json'))
    else:
        archived_resource_ids = []

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
            operations.append(CommitOperationDelete(
                path_in_repo=f'unarchived/{fn}',
            ))
            archived_resource_ids.append(os.path.splitext(fn)[0])

        all_records = _make_records()
        all_records.append({'filename': package_name, 'size': os.path.getsize(zip_file)})
        all_records = sorted(all_records, key=lambda x: x['filename'], reverse=True)

        df_records = []
        for item in all_records:
            url_for_download = hf_hub_url(
                repo_id=_REPOSITORY, repo_type="dataset",
                filename=f"packs/{item['filename']}"
            )
            df_records.append({
                'Filename': item['filename'],
                'Size': size_to_bytes_str(item['size'], precision=3),
                'Link': f'[Download]({url_for_download})'
            })

        df = pd.DataFrame(df_records)

        with TemporaryDirectory() as td:
            md_file = os.path.join(td, 'README.md')
            with open(md_file, 'w') as f:
                print('---', file=f)
                print('license: mit', file=f)
                print('---', file=f)
                print('', file=f)
                print(df.to_markdown(index=False), file=f)

            operations.append(CommitOperationAdd(
                path_or_fileobj=md_file,
                path_in_repo='README.md',
            ))

            index_file = os.path.join(td, 'index.json')
            with open(index_file, 'w') as f:
                json.dump(all_records, f, sort_keys=True, ensure_ascii=False, indent=4)
            operations.append(CommitOperationAdd(
                path_or_fileobj=index_file,
                path_in_repo='index.json',
            ))

            archived_json_file = os.path.join(td, 'archived.json')
            with open(archived_json_file, 'w') as f:
                json.dump(archived_resource_ids, f, indent=4, ensure_ascii=False)
            operations.append(CommitOperationAdd(
                path_or_fileobj=archived_json_file,
                path_in_repo='archived.json',
            ))

            if operations:
                while True:
                    try:
                        hf_client.create_commit(
                            repo_id=_REPOSITORY,
                            repo_type='dataset',
                            operations=operations,
                            commit_message=f'Create new package {package_name!r}.'
                        )
                    except HfHubHTTPError as err:
                        logging.exception(err)
                        logging.warning('Retry to commit ...')
                    else:
                        break
