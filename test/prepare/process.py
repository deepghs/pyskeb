import json
import logging
import os
import re
import zipfile
from contextlib import contextmanager
from functools import lru_cache
from typing import List, Set

from hbutils.system import TemporaryDirectory

from .base import hf_fs, _REPOSITORY, hf_client, _ensure_repository
from .dropbox import is_dropbox, get_dropbox_resource, download_dropbox_to_directory
from .google import is_google_drive, get_google_resource_id, download_google_to_directory
from .imgur import is_imgur, get_imgur_resource, download_imgur_to_directory

KNOWN_SITES = [
    (is_google_drive, get_google_resource_id, download_google_to_directory),
    (is_imgur, get_imgur_resource, download_imgur_to_directory),
    (is_dropbox, get_dropbox_resource, download_dropbox_to_directory),
]


@contextmanager
def url_to_zip(url, prefix: str = ''):
    for fn_check, fn_rid, fn_download in KNOWN_SITES:
        if fn_check(url):
            resource_id = fn_rid(url)
            if resource_id is None:
                logging.info(f'Unknown resource info for URL {url!r}, skipped!')
                continue

            with TemporaryDirectory() as td, TemporaryDirectory() as ztd:
                fn_download(url, td)
                zip_file = os.path.join(ztd, f'{resource_id}.zip')
                os.system(f'tree {td!r}')

                written = False
                with zipfile.ZipFile(zip_file, 'w') as zf:
                    for root, dirs, files in os.walk(td):
                        for file in files:
                            filename = os.path.join(td, root, file)
                            relname = os.path.relpath(filename, td)
                            relname_body, relname_ext = os.path.splitext(relname)
                            underline_name = prefix + re.sub(r'[\W_]+', '_', relname_body).strip('_') + relname_ext
                            zf.write(filename, underline_name)
                            written = True

                yield zip_file if written else None

            break
    else:
        yield None


@lru_cache()
def _get_archived_resource_ids() -> List[str]:
    if hf_fs.exists(f'datasets/{_REPOSITORY}/archived.json'):
        return json.loads(hf_fs.read_text(f'datasets/{_REPOSITORY}/archived.json'))
    else:
        return []


@lru_cache()
def _get_archived_resource_ids_set() -> Set[str]:
    return set(_get_archived_resource_ids())


def _is_resource_exist(resource_id: str) -> bool:
    return (resource_id in _get_archived_resource_ids_set()) or \
        hf_fs.exists(f'datasets/{_REPOSITORY}/unarchived/{resource_id}.zip')


def try_process_url(url, prefix: str = ''):
    _ensure_repository()
    for fn_check, fn_rid, fn_download in KNOWN_SITES:
        if fn_check(url):
            resource_id = fn_rid(url)
            if resource_id is None:
                logging.info(f'Unknown resource info for URL {url!r}, skipped!')
                continue
            else:
                logging.info(f'Resource confirmed as {resource_id!r} (URL: {url!r})')

            if _is_resource_exist(resource_id):
                logging.info(f'URL {url!r} (resource {resource_id!r}) already crawled, skipped!')
                return

            with url_to_zip(url, prefix) as zip_file:
                if zip_file is not None:
                    hf_client.upload_file(
                        path_or_fileobj=zip_file,
                        path_in_repo=f'unarchived/{resource_id}.zip',
                        repo_id=_REPOSITORY,
                        repo_type='dataset',
                    )
                else:
                    logging.info('Empty package detected, skipped!')

            return
    else:
        logging.info(f'URL {url!r} unconfirmed, skipped.')
