import datetime
import json
import logging
import os
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from threading import Lock

import pandas as pd
from hbutils.system import TemporaryDirectory
from huggingface_hub import hf_hub_download, HfApi
from orator import DatabaseManager
from tqdm.auto import tqdm
from waifuc.utils import get_requests_session

logging.basicConfig(level=logging.DEBUG)


def create_sqlite(db_file):
    db_ = DatabaseManager({
        'sqlite': {
            'driver': 'sqlite',
            'database': db_file,
        }
    })
    db_.connection().enable_query_log()
    return db_


@lru_cache()
def _get_danbooru_db() -> DatabaseManager:
    return create_sqlite(hf_hub_download(
        'deepghs/site_tags',
        'danbooru.donmai.us/tags.sqlite',
        repo_type='dataset'
    ))


_danbooru_db = _get_danbooru_db()


def _tag_post_cnt(tag) -> int:
    lst = list(_danbooru_db.table('tags').select('*').where('name', tag).get())
    print(lst)
    if lst:
        return lst[0]['post_count']
    else:
        return 0


session = get_requests_session(headers={
    "User-Agent": f"cyberharem_artists/v1.0",
    'Content-Type': 'application/json; charset=utf-8',
})


def _get_page_data(page: int):
    resp = session.get(
        'https://danbooru.donmai.us/artists.json',
        params={
            "format": "json",
            "limit": "1000",
            "page": str(page),
            'search[order]': 'post_count',
        }
    )
    resp.raise_for_status()
    return resp.json()


def _get_total_pages():
    l, r = 1, 2
    while _get_page_data(r):
        logging.info(f'Find r: {r}')
        l, r = r, r * 2

    while l < r:
        m = (l + r + 1) // 2
        logging.info(f'Finding at {m} (l: {l}, r: {r}) ...')
        if _get_page_data(m):
            l = m
        else:
            r = m - 1

    return r


def _get_all_data(pages: int = None):
    result = []
    keys = set()
    lock = Lock()

    total_pages = _get_total_pages() if pages is None else pages
    pg = tqdm(desc='ALL', total=total_pages)

    def _append_data(data):
        with lock:
            if data['id'] not in keys:
                keys.add(data['id'])
                result.append(data)

    def _fn(p):
        items = list(_get_page_data(p))
        tags = [item['name'] for item in items]
        mapping = {
            item['name']: item['post_count'] for item in
            _danbooru_db.table('tags').select('*').where_in('name', tags).get()
        }
        for item in items:
            _append_data({
                **item,
                'created_at': datetime.datetime.fromisoformat(item['created_at']).timestamp(),
                'updated_at': datetime.datetime.fromisoformat(item['updated_at']).timestamp(),
                # 'created_at': dateparser.parse(item['created_at']).timestamp(),
                # 'updated_at': dateparser.parse(item['updated_at']).timestamp(),
                'post_count': mapping.get(item['name'], 0)
            })
        pg.update()

    tp = ThreadPoolExecutor(max_workers=max(6, os.cpu_count()))
    for i in range(1, total_pages + 1):
        tp.submit(_fn, i)

    tp.shutdown()
    result = sorted(result, key=lambda x: (-x['post_count'], x['id']))
    return result


def _save_all_data_to_sql(sql_file, pages=None):
    sql = sqlite3.connect(sql_file)

    all_data = _get_all_data(pages)
    df = pd.DataFrame(all_data)

    df['created_at'] = pd.to_datetime(df['created_at'] * 1e9)
    df['updated_at'] = pd.to_datetime(df['updated_at'] * 1e9)
    df['other_names'] = df['other_names'].apply(json.dumps).astype(str)

    artists_indices = [
        'id', 'created_at', 'name', 'updated_at', 'is_deleted',
        'group_name', 'is_banned', 'other_names', 'post_count'
    ]
    df.to_sql('artists', sql, index=False)
    for column in artists_indices:
        sql.execute(f"CREATE INDEX artists_index_{column} ON artists ({column});").fetchall()

    alias_records = []
    for item in all_data:
        for alias in item['other_names']:
            alias_records.append({
                'alias_name': alias,
                'name': item['name'],
                'tag_id': item['id'],
            })
    df_alias = pd.DataFrame(alias_records)

    artists_aliases_indices = ['alias_name', 'name']
    df_alias.to_sql('artists_aliases', sql, index_label='id')
    for column in artists_aliases_indices:
        sql.execute(f"CREATE INDEX artists_aliases_index_{column} ON artists_aliases ({column});").fetchall()

    sql.close()


def push_artists_sqlite():
    hf_client = HfApi(token=os.environ['HF_TOKEN_X'])
    repository = os.environ['REMOTE_REPOSITORY_X']

    with TemporaryDirectory() as td:
        sql_file = os.path.join(td, 'artists.sqlite')
        _save_all_data_to_sql(sql_file)

        hf_client.upload_file(
            repo_id=repository,
            repo_type='dataset',
            path_or_fileobj=sql_file,
            path_in_repo='artists.sqlite'
        )
