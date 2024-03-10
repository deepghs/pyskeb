import itertools
import json
import math
import mimetypes
import os
import random
import re
import time
from typing import Iterator

import httpx
import pandas as pd
import requests.exceptions
from ditk import logging
from hbutils.string import plural_word
from hbutils.system import TemporaryDirectory
from hfutils.operate import upload_directory_as_directory
from huggingface_hub import hf_hub_download
from pyrate_limiter import Duration, Rate, Limiter
from tqdm import tqdm
from waifuc.source import ThreeDBooruSource
from waifuc.utils import srequest

from ..base import hf_fs, hf_client, hf_token

_REPLACEMENTS = [
    ('\x10', ''),
]


def _safe_json(text):
    for from_, to_ in _REPLACEMENTS:
        text = text.replace(from_, to_)
    return json.loads(text)


def crawl_bh_index(repository: str, quit_page_when_exist: bool = True,
                   max_cnt: int = 200000, max_time_limit: int = 50 * 60):
    start_time = time.time()

    if not hf_client.repo_exists(repo_id=repository, repo_type='dataset'):
        hf_client.create_repo(repo_id=repository, repo_type='dataset', private=True)
        attr_lines = hf_fs.read_text(f'datasets/{repository}/.gitattributes').splitlines(keepends=False)
        attr_lines.append('*.json filter=lfs diff=lfs merge=lfs -text')
        attr_lines.append('*.csv filter=lfs diff=lfs merge=lfs -text')
        hf_fs.write_text(
            f'datasets/{repository}/.gitattributes',
            os.linesep.join(attr_lines),
        )

    if hf_fs.exists(f'datasets/{repository}/exist_ids.json'):
        exist_ids = set(json.loads(hf_fs.read_text(f'datasets/{repository}/exist_ids.json')))
    else:
        exist_ids = set()

    if hf_fs.exists(f'datasets/{repository}/3dbooru.csv'):
        df_ = pd.read_csv(hf_hub_download(repo_id=repository, repo_type='dataset', filename='3dbooru.csv'))
        records = df_.to_dict('records')
    else:
        records = []

    if hf_fs.exists(f'datasets/{repository}/tags.csv'):
        df_ = pd.read_csv(hf_hub_download(repo_id=repository, repo_type='dataset', filename='tags.csv'))
        tags = {
            titem['name']: titem
            for titem in df_.to_dict('records')
        }
    else:
        tags = {}

    s = ThreeDBooruSource(['rating:safe'], min_size=5000)

    def _iter_items_from_pages() -> Iterator[dict]:
        page_rate = Rate(1, int(math.ceil(Duration.SECOND * 1.0)))
        page_limiter = Limiter(page_rate, max_delay=1 << 32)

        current_page = 1
        while True:
            page_limiter.try_acquire('page')
            resp = srequest(s.session, 'GET', 'http://behoimi.org/post/index.json', params={
                'page': str(current_page),
                'limit': '100',
            }, raise_for_status=False)
            try:
                resp.raise_for_status()
                lst = _safe_json(resp.text)
            except (requests.exceptions.RequestException, json.JSONDecodeError, httpx.HTTPError) as err:
                logging.info(f'Paginate failed on page {current_page!r} due to {err!r}')
                return

            for item in lst:
                if item['id'] in exist_ids:
                    if quit_page_when_exist:
                        return
                else:
                    yield item

            current_page += 1

    def _iter_from_random_ids():
        page_rate = Rate(1, int(math.ceil(Duration.SECOND * 1.0)))
        page_limiter = Limiter(page_rate, max_delay=1 << 32)
        d_tags = pd.read_csv(hf_hub_download(
            repo_id=repository,
            repo_type='dataset',
            filename='index_tags.csv',
        ))
        all_tags_count_map = dict(zip(d_tags['name'], d_tags['count']))
        all_tags = sorted(d_tags[d_tags['count'] >= 1]['name'])

        while True:
            current_tag = random.choice(all_tags)
            if current_tag in tags:
                current_tag_count = tags[current_tag]['count']
            else:
                current_tag_count = 0
            all_tag_count = all_tags_count_map.get(current_tag, 0)
            if current_tag_count > all_tag_count * 0.95 or current_tag_count >= 95000:
                logging.info(f'Current tag {current_tag!r} reach safe limit '
                             f'(current posts: {current_tag_count!r}, total posts: {all_tag_count!r}), skipped.')
                all_tags.remove(current_tag)
                continue

            logging.info(f'Getting posts of {current_tag!r} ({current_tag_count!r}/{all_tag_count}) ...')
            current_page = 1
            while True:
                page_limiter.try_acquire('page')
                resp = srequest(s.session, 'GET', 'http://behoimi.org/post/index.json', params={
                    'page': str(current_page),
                    'limit': '100',
                    'tags': current_tag,
                }, raise_for_status=False)
                try:
                    resp.raise_for_status()
                    lst = _safe_json(resp.text)
                except (requests.exceptions.RequestException, json.JSONDecodeError, httpx.HTTPError) as err:
                    logging.info(f'Paginate failed on page {current_page!r} due to {err!r}')
                    break

                if not lst:
                    break

                for item in lst:
                    if item['id'] not in exist_ids:
                        yield item

                current_page += 1

            all_tags.remove(current_tag)

    source = itertools.chain(_iter_items_from_pages(), _iter_from_random_ids())
    cnt = 0
    pg = tqdm(total=max_cnt)
    for item in source:
        if cnt >= max_cnt:
            break
        if time.time() - start_time >= max_time_limit:
            break

        if item['id'] in exist_ids:
            logging.info(f'Post {item["id"]} already in, skipped.')
            continue

        logging.info(f'Post {item["id"]} confirmed.')
        if item['file_url']:
            mtype, _ = mimetypes.guess_type(item['file_url'])
        else:
            mtype = None

        record = {
            **item,
            'created_at': item['created_at']['s'],
            'tags': ' ' + item['tags'].strip() + ' ',
            'type': mtype,
        }
        records.append(record)
        item_tags = sorted(set(filter(bool, re.split(r'\s+', item['tags'].strip()))))
        for tag in item_tags:
            if tag not in tags:
                tags[tag] = {'name': tag, 'count': 0}
            tags[tag]['count'] += 1
        exist_ids.add(item['id'])
        cnt += 1
        pg.update()

    with TemporaryDirectory() as td:
        df_records = pd.DataFrame(records)
        df_records = df_records.sort_values(['id'], ascending=False)
        df_records.to_csv(os.path.join(td, '3dbooru.csv'), index=False)
        df_records_shown = df_records[:50]
        df_records_shown = df_records_shown[['id', 'width', 'height', 'type', 'tags', 'file_url']]

        df_tags = pd.DataFrame(list(tags.values()))
        df_tags = df_tags.sort_values(['count', 'name'], ascending=[False, True])
        df_tags.to_csv(os.path.join(td, 'tags.csv'), index=False)
        df_tags_shown = df_tags[:500][['name', 'count']]

        with open(os.path.join(td, 'exist_ids.json'), 'w') as f:
            json.dump(sorted(exist_ids), f)

        md_file = os.path.join(td, 'README.md')
        with open(md_file, 'w') as f:
            print('---', file=f)
            print('license: mit', file=f)
            print('---', file=f)
            print('', file=f)
            print('## Records', file=f)
            print(f'', file=f)
            print(f'ATTENTION: Remember to add `Referer: http://behoimi.org/` into request headers '
                  f'when download these images.', file=f)
            print(f'', file=f)
            print(f'{plural_word(len(df_records), "record")} in total. '
                  f'Only {plural_word(len(df_records_shown), "record")} shown.', file=f)
            print(f'', file=f)
            print(df_records_shown.to_markdown(index=False), file=f)
            print(f'', file=f)
            print(f'## Tags', file=f)
            print(f'', file=f)
            print(f'{plural_word(len(df_tags), "tag")} in total. '
                  f'Only {plural_word(len(df_tags_shown), "tag")} shown.', file=f)
            print(f'', file=f)
            print(df_tags_shown.to_markdown(index=False), file=f)
            print('', file=f)

        upload_directory_as_directory(
            repo_id=repository,
            repo_type='dataset',
            local_directory=td,
            path_in_repo='.',
            hf_token=hf_token,
        )


if __name__ == '__main__':
    logging.try_init_root(logging.INFO)
    crawl_bh_index(
        repository=os.environ['REMOTE_REPOSITORY_BH'],
        quit_page_when_exist=True,
        max_cnt=100000,
        max_time_limit=45 * 60,
    )
