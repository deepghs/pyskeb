import json
import os.path
import re
import shutil
import warnings
import zipfile

import pandas as pd
import requests
from PIL import UnidentifiedImageError
from PIL.Image import DecompressionBombError
from ditk import logging
from hbutils.scale import size_to_bytes_str
from hbutils.string import plural_word
from hbutils.system import TemporaryDirectory, urlsplit
from hfutils.operate import download_file_to_file, upload_directory_as_directory
from huggingface_hub import hf_hub_url
from imgutils.validate import anime_real
from tqdm import tqdm

from pyskeb.utils import download_file, get_requests_session, get_random_ua
from ..base import hf_fs, hf_client, hf_token


def mhs_project_order_crawl(repository: str, maxcnt: int = 100):
    session = get_requests_session()
    session.headers.update({
        'User-Agent': get_random_ua(),
        'Referer': 'https://www.mihuashi.com/projects',
    })

    def _name_safe(name_text):
        return re.sub(r'[\W_]+', '_', name_text).strip('_')

    logging.info('Access projects list ...')
    resp = session.get('https://www.mihuashi.com/projects/', params={'page': '1', 'zone': '2'})
    resp.raise_for_status()

    if not hf_client.repo_exists(repo_id=repository, repo_type='dataset'):
        hf_client.create_repo(repo_id=repository, repo_type='dataset', private=True)
        attr_lines = hf_fs.read_text(f'datasets/{repository}/.gitattributes').splitlines(keepends=False)
        attr_lines.append('*.json filter=lfs diff=lfs merge=lfs -text')
        attr_lines.append('*.csv filter=lfs diff=lfs merge=lfs -text')
        hf_fs.write_text(
            f'datasets/{repository}/.gitattributes',
            os.linesep.join(attr_lines),
        )

    if hf_fs.exists(f'datasets/{repository}/exist_sids.json'):
        exist_sids = json.loads(hf_fs.read_text(f'datasets/{repository}/exist_sids.json'))
    else:
        exist_sids = []
    exist_sids = set(exist_sids)
    logging.info(f'{plural_word(len(exist_sids), "exist sid")} detected.')

    pg = tqdm(desc='Max Count', total=maxcnt)

    with TemporaryDirectory() as td:
        if hf_fs.exists(f'datasets/{repository}/records.csv'):
            records_csv = os.path.join(td, 'records.csv')
            download_file_to_file(
                local_file=records_csv,
                repo_id=repository,
                repo_type='dataset',
                file_in_repo='records.csv',
                hf_token=hf_token,
            )
            records = pd.read_csv(records_csv).to_dict('records')
        else:
            records = []

        img_dir = os.path.join(td, 'images')
        os.makedirs(img_dir, exist_ok=True)

        current_count = 0
        has_new = False
        for project_id in range(100, 2556530):
            suit_id = f'project_{project_id}'
            logging.info(f'Resource {suit_id!r} confirmed.')
            if suit_id in exist_sids:
                logging.info(f'Suit item {suit_id!r} already crawled, skipped.')
                continue

            try:
                resp = session.get(f'https://www.mihuashi.com/api/v1/projects/{project_id}')
                resp.raise_for_status()
                project_info = resp.json()['project']
                project_name = project_info['name']
                project_zone = project_info['zone_id']

                owner_info = project_info['owner']
                owner_name = owner_info['name']
                owner_id = owner_info['id']

                example_image_items = project_info['example_images']
                for ei, eitem in enumerate(example_image_items):
                    e_url = eitem['url']
                    _, ext = os.path.splitext(urlsplit(e_url).filename)
                    if ext == '.j':
                        e_url = e_url + 'pg'
                    elif ext == '.p':
                        e_url = e_url + 'ng'

                    e_name = (f'{owner_id}__{_name_safe(owner_name)}__'
                              f'{project_id}_z{project_zone}__{_name_safe(project_name)}__'
                              f'e_{ei}')
                    _, ext = os.path.splitext(urlsplit(e_url).filename)
                    dst_file = os.path.join(img_dir, f'{e_name}{ext}')
                    logging.info(f'Downloading {e_url!r} to {dst_file!r} ...')
                    download_file(e_url, filename=dst_file, session=session)

                    try:
                        real_type, _ = anime_real(dst_file)
                    except UnidentifiedImageError:
                        warnings.warn(f'Resource {e_name!r} unidentified as image, skipped.')
                        os.remove(dst_file)
                    except (IOError, DecompressionBombError) as err:
                        warnings.warn(f'Skipped due to IO error: {err!r}')
                        os.remove(dst_file)
                    else:
                        if real_type != 'anime':
                            logging.warning(f'Resource {e_name!r} not an anime image, skipped.')
                            os.remove(dst_file)

                card_items = project_info['character_cards']
                for ci, citem in enumerate(card_items):
                    c_token = citem['token']
                    c_id = citem['id']

                    resp = session.get(f'https://www.mihuashi.com/api/v1/character_cards/{c_token}')
                    resp.raise_for_status()

                    c_image_url = resp.json()['character_card']['image_url']
                    _, ext = os.path.splitext(urlsplit(c_image_url).filename)
                    if ext == '.j':
                        c_image_url = c_image_url + 'pg'
                    elif ext == '.p':
                        c_image_url = c_image_url + 'ng'

                    c_image_title = resp.json()['character_card']['name']
                    c_name = (f'{owner_id}__{_name_safe(owner_name)}__'
                              f'{project_id}_z{project_zone}__{_name_safe(project_name)}__'
                              f'c_{c_id}__{_name_safe(c_image_title)}')
                    _, ext = os.path.splitext(urlsplit(c_image_url).filename)
                    dst_file = os.path.join(img_dir, f'{c_name}{ext}')
                    logging.info(f'Downloading {c_image_url!r} to {dst_file!r} ...')
                    download_file(c_image_url, filename=dst_file, session=session)

                    try:
                        real_type, _ = anime_real(dst_file)
                    except UnidentifiedImageError:
                        warnings.warn(f'Resource {c_name!r} unidentified as image, skipped.')
                        os.remove(dst_file)
                    except (IOError, DecompressionBombError) as err:
                        warnings.warn(f'Skipped due to IO error: {err!r}')
                        os.remove(dst_file)
                    else:
                        if real_type != 'anime':
                            logging.warning(f'Resource {c_name!r} not an anime image, skipped.')
                            os.remove(dst_file)

            except requests.exceptions.HTTPError as err:
                status_code = err.response.status_code
                if status_code in {401}:
                    logging.warning(f'Login required for Project {project_id!r}.')
                elif status_code in {403}:
                    logging.warning(f'Project {project_id!r} is private or hidden.')
                elif status_code in {404}:
                    logging.warning(f'Project {project_id!r} not exists.')
                elif status_code in {423}:
                    logging.warning(f'Project {project_id!r} is blocked.')
                else:
                    logging.error(f'Project {project_id!r} skipped due to error: {err!r}')
                    continue

            exist_sids.add(suit_id)
            has_new = True
            pg.update()
            current_count += 1
            if current_count >= maxcnt:
                break

        if not has_new:
            logging.info('No update, quit.')

        export_dir = os.path.join(td, 'export')
        os.makedirs(export_dir, exist_ok=True)

        item_cnt = len(os.listdir(img_dir))
        logging.info(f'{plural_word(item_cnt, "image")} in total.')
        if item_cnt:
            from ..repack import _timestamp
            img_pack_file = os.path.join(td, f'mhs_project_pack_{_timestamp()}.zip')
            with zipfile.ZipFile(img_pack_file, 'w') as zf:
                for file in os.listdir(img_dir):
                    zf.write(os.path.join(img_dir, file), file)
                    os.remove(os.path.join(img_dir, file))

            filename = os.path.basename(img_pack_file)
            records.append({
                'Filename': filename,
                'Images': item_cnt,
                'Size': size_to_bytes_str(os.path.getsize(img_pack_file), precision=3),
                'Download': f'[Download]'
                            f'({hf_hub_url(repo_id=repository, repo_type="dataset", filename=f"packs/{filename}")})',
            })

            dst_pack_file = os.path.join(export_dir, f'packs', filename)
            os.makedirs(os.path.dirname(dst_pack_file), exist_ok=True)
            shutil.copy(img_pack_file, dst_pack_file)

        df = pd.DataFrame(records)
        df = df.sort_values(['Filename'], ascending=False)
        df.to_csv(os.path.join(export_dir, 'records.csv'), index=False)
        with open(os.path.join(export_dir, 'exist_sids.json'), 'w') as f:
            json.dump(sorted(exist_sids), f)

        md_file = os.path.join(export_dir, 'README.md')
        with open(md_file, 'w') as f:
            print('---', file=f)
            print('license: mit', file=f)
            print('---', file=f)
            print('', file=f)
            print(df.to_markdown(index=False), file=f)

        upload_directory_as_directory(
            repo_id=repository,
            repo_type='dataset',
            local_directory=export_dir,
            path_in_repo='.',
            hf_token=hf_token,
        )


if __name__ == '__main__':
    logging.try_init_root(logging.INFO)
    mhs_project_order_crawl(
        repository=os.environ['REMOTE_REPOSITORY_MHS_PROJECT'],
        maxcnt=10000,
    )
