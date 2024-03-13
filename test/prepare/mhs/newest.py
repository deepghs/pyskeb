import itertools
import json
import math
import os.path
import random
import re
import shutil
import time
import zipfile
from typing import Iterator, Optional

import dateparser
import pandas as pd
import requests
from ditk import logging
from hbutils.scale import size_to_bytes_str
from hbutils.string import plural_word
from hbutils.system import TemporaryDirectory, urlsplit
from hfutils.operate import download_file_to_file, upload_directory_as_directory
from huggingface_hub import hf_hub_url
from pyrate_limiter import Rate, Limiter, Duration
from tqdm import tqdm

from pyskeb.utils import download_file, get_requests_session, get_random_ua
from ..base import hf_fs, hf_client, hf_token


class IpStopped(Exception):
    pass


class MHSSession:
    def __init__(self, proxy_pool: Optional[str] = None):
        self.session: Optional[requests.Session] = None
        self.proxy_pool = proxy_pool
        self.refresh()

    def refresh(self):
        self.session = get_requests_session()
        self.session.headers.update({
            'User-Agent': get_random_ua(),
            'Referer': 'https://www.mihuashi.com/artworks',
            'Accept': 'application/json, text/plain',
            'Accept-Encoding': 'gzip, deflate, br',
            'Accept-Language': 'zh-CN,zh;q=0.9',
        })
        if self.proxy_pool:
            logging.info('Proxy pool enabled!')
            self.session.proxies.update({
                'http': self.proxy_pool,
                'https': self.proxy_pool,
            })

            from .pp import refresh_pp
            refresh_pp(
                bd_token=os.environ['BD_TOKEN'],
                zone=os.environ['BD_MHS_ZONE']
            )
            time.sleep(5.0)

    def iter_artwork_ids_from_page(self, order: Optional[str] = 'recent', q: Optional[str] = None,
                                   max_page_limit: int = 1000) -> Iterator[int]:
        page = 1
        retries = 0
        while True:
            logging.info(f'Requesting for page {page!r}.')
            try:
                params = {
                    'page': str(page),
                }
                if order:
                    params['type'] = order
                if q:
                    params['q'] = q
                resp = self.session.get(
                    'https://www.mihuashi.com/api/v1/artworks/search',
                    params=params
                )
                if resp.status_code == 403:
                    retries += 1
                    if retries > 5:
                        raise IpStopped('Max refresh try exceeded.')
                    else:
                        logging.info('Ip ban detected, just refresh ip pool.')
                        self.refresh()
                        continue

                resp.raise_for_status()
            except requests.exceptions.RequestException as err:
                logging.warning(f'Page {page} skipped due to error: {err!r}')
            else:
                for item in resp.json()['artworks']:
                    item_id = item['id']
                    yield item_id

            page += 1
            retries = 0
            if page > max_page_limit:
                break

    def homepage(self):
        logging.info('Access artwork page list ...')
        retries = 0
        while True:
            resp = self.session.get('https://www.mihuashi.com/artworks', headers={
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,'
                          'application/signed-exchange;v=b3;q=0.7',
            })
            if resp.status_code == 403:
                retries += 1
                if retries > 5:
                    raise IpStopped('Max refresh try exceeded.')
                else:
                    logging.info('Ip ban detected, just refresh ip pool.')
                    self.refresh()
                    continue

            resp.raise_for_status()
            break


min_id, max_id = 5000000, 13560534


def _iter_artwork_ids_randomly(min_id, max_id) -> Iterator[int]:
    while True:
        yield random.randint(min_id, max_id)


def mhs_newest_crawl(repository: str, maxcnt: int = 500, max_time_limit: int = 50 * 60, use_random: bool = True,
                     proxy_pool: Optional[str] = None):
    start_time = time.time()
    rate = Rate(1, int(math.ceil(Duration.SECOND * 3)))
    limiter = Limiter(rate, max_delay=1 << 32)
    client = MHSSession(proxy_pool=proxy_pool)
    client.homepage()

    def _name_safe(name_text):
        return re.sub(r'[\W_]+', '_', name_text).strip('_')

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
            os.remove(records_csv)
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
            d_tags = pd.read_csv(tags_csv)
            all_tags = d_tags.to_dict('records')
            all_tag_ids = set(pd.read_csv(tags_csv)['id'])
            os.remove(tags_csv)
            if random.random() < 1.1:
                min_names = list(d_tags[(d_tags['type'] != 'custom_tag')].
                                 sort_values(['count'], ascending=True)[:3]['name'])
                q = random.choice(min_names)
            else:
                q = None
        else:
            all_tags = []
            all_tag_ids = set()
            q = None
        order = random.choice([None, 'recent', 'hot'])
        if q is None:
            logging.info(f'Ready to crawl newest images (order: {order!r}) ...')
        else:
            logging.info(f'Ready to crawl newest images with tag {q!r} (order: {order!r}) ...')

        img_dir = os.path.join(td, 'images')
        os.makedirs(img_dir, exist_ok=True)

        if use_random:
            current_max_id = max(item['id'] for item in all_artworks)
            logging.info(f'Current max id: {current_max_id!r}.')
            id_source = itertools.chain(
                client.iter_artwork_ids_from_page(q=q, order=order),
                _iter_artwork_ids_randomly(min_id, current_max_id)
            )
        else:
            id_source = client.iter_artwork_ids_from_page(q=q, order=order)

        current_count = 0
        try:
            for item_id in id_source:
                if time.time() - start_time >= max_time_limit:
                    break

                suit_id = f'artwork_{item_id}'
                logging.info(f'Resource {suit_id!r} confirmed.')
                if suit_id in exist_sids:
                    logging.info(f'Resource {suit_id!r} already crawled, skipped.')
                    continue

                limiter.try_acquire('info')
                try:
                    resp = client.session.get(f'https://www.mihuashi.com/api/v1/artworks/{item_id}')
                except requests.exceptions.RequestException as err:
                    logging.info(f'Resource {suit_id!r} skipped due to request error: {err!r}')
                    continue
                if not resp.ok:
                    if resp.status_code in {401}:
                        logging.warning(f'Login required for Resource {suit_id!r}.')
                    elif resp.status_code in {403}:
                        logging.warning(f'Resource {suit_id!r} is private or hidden, or ip get banned.')
                        client.refresh()
                        continue
                    elif resp.status_code in {404}:
                        logging.warning(f'Resource {suit_id!r} not exists.')
                    elif resp.status_code in {423}:
                        logging.warning(f'Resource {suit_id!r} is blocked.')
                    else:
                        logging.error(f'Resource {suit_id!r} skipped due to error: {err!r}')
                        continue

                else:
                    item_type = resp.json()['artwork']['artwork_type']
                    author_info = resp.json()['artwork']['author']
                    author_id = author_info['id']
                    author_name = author_info['name']
                    artwork_info = resp.json()['artwork']
                    item_url = artwork_info['url']

                    item_name = f'{author_id}__{_name_safe(author_name)}__{item_id}'

                    _, ext = os.path.splitext(urlsplit(item_url).filename)
                    dst_file = os.path.join(img_dir, f'{item_name}{ext}')
                    logging.info(f'Downloading {item_url!r} to {dst_file!r} ...')
                    try:
                        download_file(item_url, filename=dst_file, session=client.session)
                    except (AssertionError, requests.exceptions.RequestException) as err:
                        logging.error(f'Download of {item_url!r} skipped due to error: {err!r}')
                        continue

                    artwork_tags = artwork_info['tags']
                    for tag_item in artwork_tags:
                        if tag_item['id'] not in all_tag_ids:
                            all_tags.append(tag_item)
                            all_tag_ids.add(tag_item['id'])

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

                exist_sids.add(suit_id)
                pg.update()
                current_count += 1
                if current_count >= maxcnt:
                    break
                if current_count % 10 == 0:
                    client.refresh()
                if time.time() - start_time >= max_time_limit:
                    break

        except IpStopped:
            logging.info('IP stopped, stop crawling ...')

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
        date_analysis = {}
        for artwork_item in all_artworks:
            date_str = dateparser.parse(artwork_item['created_at']).strftime('%Y-%m-%d')
            date_analysis[date_str] = date_analysis.get(date_str, 0) + 1
            for tag_id in artwork_item['tag_ids']:
                tags_analysis[tag_id] = tags_analysis.get(tag_id, 0) + 1
        all_cnt = len(all_artworks)
        for tag_item in all_tags:
            tag_item['count'] = tags_analysis.get(tag_item['id'], 0)
        all_tags = sorted(
            all_tags,
            key=lambda x: (
                1 if x['type'] == 'custom_tag' else 0,
                (-x['count'], x['id']) if x['type'] == 'custom_tag' else (x['type'], x['id'])
            )
        )
        df_tags = pd.DataFrame(all_tags)
        df_tags.to_csv(os.path.join(export_dir, 'tags.csv'), index=False)
        with open(os.path.join(export_dir, 'exist_sids.json'), 'w') as f:
            json.dump(sorted(exist_sids), f)
        all_artworks = sorted(all_artworks, key=lambda x: x['id'])
        with open(os.path.join(export_dir, 'artworks.json'), 'w') as f:
            json.dump(all_artworks, f, ensure_ascii=False, sort_keys=True, indent=4)

        df_date = pd.DataFrame([{'date': key, 'count': value} for key, value in date_analysis.items()])
        df_date = df_date.sort_values(['date'], ascending=False)
        md_file = os.path.join(export_dir, 'README.md')
        with open(md_file, 'w') as f:
            print('---', file=f)
            print('license: mit', file=f)
            print('---', file=f)
            print('', file=f)
            print('## Packages', file=f)
            print(f'', file=f)
            print(df[:20].to_markdown(index=False), file=f)
            print(f'', file=f)
            print('## Analysis', file=f)
            print('', file=f)
            print(f'{plural_word(all_cnt, "images")} in total.', file=f)
            print('', file=f)
            print(df_date[:30].to_markdown(index=False), file=f)
            print('', file=f)
            print('## Tags', file=f)
            print('', file=f)
            print('Only some selected tags are shown.', file=f)
            print('', file=f)
            t_df_tags = df_tags
            t_df_tags = t_df_tags[(t_df_tags['type'] != 'custom_tag') | (t_df_tags['count'] >= 50)]
            print(t_df_tags.to_markdown(index=False), file=f)

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
        use_random=False,
        maxcnt=2000,
        max_time_limit=45 * 60,
        proxy_pool=os.environ.get('PP_MHS'),
    )
