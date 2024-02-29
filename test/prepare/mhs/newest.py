import json
import os.path
import re
import shutil
import zipfile

import pandas as pd
from ditk import logging
from hbutils.scale import size_to_bytes_str
from hbutils.string import plural_word
from hbutils.system import TemporaryDirectory, urlsplit
from hfutils.operate import download_file_to_file, upload_directory_as_directory
from huggingface_hub import hf_hub_url
from tqdm import tqdm

from pyskeb.utils import download_file, get_requests_session, get_random_ua
from ..base import hf_fs, hf_client, hf_token


def mhs_newest_crawl(repository: str, maxcnt: int = 500):
    session = get_requests_session()
    session.headers.update({
        'User-Agent': get_random_ua(),
        'Referer': 'https://www.mihuashi.com/artworks',
    })

    def _name_safe(name_text):
        return re.sub(r'[\W_]+', '_', name_text).strip('_')

    logging.info('Access artwork page list ...')
    resp = session.get('https://www.mihuashi.com/artworks')
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

    if hf_fs.exists(f'datasets/{repository}/artworks.json'):
        all_artworks = json.loads(hf_fs.read_text(f'datasets/{repository}/artworks.json'))
    else:
        all_artworks = []

    from ..repack import _timestamp
    pack_name = f'mhs_newest_pack_{_timestamp()}.zip'

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

        if hf_fs.exists(f'datasets/{repository}/tags.csv'):
            tags_csv = os.path.join(td, 'tags.csv')
            download_file_to_file(
                local_file=tags_csv,
                repo_id=repository,
                repo_type='dataset',
                file_in_repo='tags.csv',
                hf_token=hf_token,
            )
            all_tags = pd.read_csv(tags_csv).to_dict('records')
            all_tag_ids = set(pd.read_csv(tags_csv)['id'])
        else:
            all_tags = []
            all_tag_ids = set()

        img_dir = os.path.join(td, 'images')
        os.makedirs(img_dir, exist_ok=True)

        page = 1
        current_count = 0
        while True:
            logging.info(f'Requesting for page {page!r}.')
            resp = session.get(
                'https://www.mihuashi.com/api/v1/artworks/search',
                params={
                    'page': str(page),
                    'type': 'recent',
                }
            )
            resp.raise_for_status()

            for item in resp.json()['artworks']:
                item_id = item['id']
                item_type = item['artwork_type']
                suit_id = f'artwork_{item_id}'
                logging.info(f'Resource {suit_id!r} confirmed.')
                if suit_id in exist_sids:
                    logging.info(f'Resource {suit_id!r} already crawled, skipped.')
                    continue

                resp = session.get(f'https://www.mihuashi.com/api/v1/artworks/{item_id}')
                resp.raise_for_status()

                author_info = resp.json()['artwork']['author']
                author_id = author_info['id']
                author_name = author_info['name']
                artwork_info = resp.json()['artwork']
                item_url = artwork_info['url']

                item_name = f'{author_id}__{_name_safe(author_name)}__{item_id}'

                _, ext = os.path.splitext(urlsplit(item_url).filename)
                dst_file = os.path.join(img_dir, f'{item_name}{ext}')
                logging.info(f'Downloading {item_url!r} to {dst_file!r} ...')
                download_file(item_url, filename=dst_file, session=session)

                artwork_tags = artwork_info['tags']
                for tag_item in artwork_tags:
                    if tag_item['id'] not in all_tag_ids:
                        all_tags.append(tag_item)
                        all_tag_ids.add(tag_item['id'])

                exist_sids.add(suit_id)
                all_artworks.append({
                    'id': item_id,
                    'type': item_type,
                    'filename': os.path.basename(dst_file),
                    'packname': pack_name,
                    'created_at': artwork_info['created_at'],
                    'author_id': author_id,
                    'author_name': author_name,
                    'tag_ids': [tag_item['id'] for tag_item in artwork_tags],
                })
                pg.update()
                current_count += 1
                if current_count >= maxcnt:
                    break

            if current_count >= maxcnt:
                break

            page += 1

        if not os.listdir(img_dir):
            logging.warning('No images found, quit.')
            return

        item_cnt = len(os.listdir(img_dir))
        img_pack_file = os.path.join(td, pack_name)
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

        export_dir = os.path.join(td, 'export')
        os.makedirs(export_dir, exist_ok=True)

        dst_pack_file = os.path.join(export_dir, f'packs', filename)
        os.makedirs(os.path.dirname(dst_pack_file), exist_ok=True)
        shutil.copy(img_pack_file, dst_pack_file)

        df = pd.DataFrame(records)
        df = df.sort_values(['Filename'], ascending=False)
        df.to_csv(os.path.join(export_dir, 'records.csv'), index=False)

        tags_analysis = {}
        for artwork_item in all_artworks:
            for tag_id in artwork_item['tag_ids']:
                tags_analysis[tag_id] = tags_analysis.get(tag_id, 0) + 1
        for tag_item in all_tags:
            tag_item['count'] = tags_analysis.get(tag_item['id'], 0)
        all_tags = sorted(all_tags, key=lambda x: (1 if x['type'] == 'custom_tag' else 0, x['type'], x['id']))
        df_tags = pd.DataFrame(all_tags)
        df_tags.to_csv(os.path.join(export_dir, 'tags.csv'), index=False)
        with open(os.path.join(export_dir, 'exist_sids.json'), 'w') as f:
            json.dump(sorted(exist_sids), f)
        all_artworks = sorted(all_artworks, key=lambda x: x['id'])
        with open(os.path.join(export_dir, 'artworks.json'), 'w') as f:
            json.dump(all_artworks, f, ensure_ascii=False, sort_keys=True, indent=4)

        md_file = os.path.join(export_dir, 'README.md')
        with open(md_file, 'w') as f:
            print('---', file=f)
            print('license: mit', file=f)
            print('---', file=f)
            print('', file=f)
            print('## Packages', file=f)
            print('', file=f)
            print(df.to_markdown(index=False), file=f)
            print('', file=f)
            print('## Tags', file=f)
            print('', file=f)
            print(df_tags.to_markdown(index=False), file=f)

        upload_directory_as_directory(
            repo_id=repository,
            repo_type='dataset',
            local_directory=export_dir,
            path_in_repo='.',
            hf_token=hf_token,
        )


if __name__ == '__main__':
    logging.try_init_root(logging.INFO)
    mhs_newest_crawl(
        repository=os.environ['REMOTE_REPOSITORY_MHS_NEWEST'],
        maxcnt=500,
    )