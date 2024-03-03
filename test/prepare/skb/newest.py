import itertools
import json
import os.path
import shutil
import time
import zipfile
from typing import Iterator, Tuple

import pandas as pd
import requests
from ditk import logging
from hbutils.scale import size_to_bytes_str
from hbutils.string import plural_word
from hbutils.system import TemporaryDirectory, urlsplit
from hfutils.operate import download_file_to_file, upload_directory_as_directory
from huggingface_hub import hf_hub_url
from tqdm import tqdm

from pyskeb.utils import download_file
from ..base import hf_fs, hf_client, hf_token
from ..listing import list_newest_posts, client, split_username_and_id_from_path
from ..url import extract_urls


def _iter_artwork_ids_from_page(max_count_limit: int = 1000) -> Iterator[Tuple[str, int]]:
    yield from list_newest_posts(limit=max_count_limit)


class Inc:
    def __init__(self):
        self.v = 0

    def inc(self):
        self.v += 1


def _iter_artwork_ids_in_queue(queue, i: Inc) -> Iterator[Tuple[str, int]]:
    i.v = 0
    while i.v < len(queue):
        yield queue[i.v]['username'], queue[i.v]['post_id']
        i.inc()


def skb_newest_crawl(repository: str, maxcnt: int = 500, max_time_limit: int = 50 * 60, use_random: bool = True):
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

    if hf_fs.exists(f'datasets/{repository}/queue.json'):
        queue = json.loads(hf_fs.read_text(f'datasets/{repository}/queue.json'))
    else:
        queue = []
    queue_suit_ids = set([f'{qitem["username"]}_{qitem["post_id"]}' for qitem in queue])

    from ..repack import _timestamp
    pack_name = f'skb_newest_pack_{_timestamp()}.zip'

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
            all_tags = pd.read_csv(tags_csv).to_dict('records')
            all_tags_map = {tag_item['name'].lower(): tag_item for tag_item in all_tags}
            os.remove(tags_csv)
        else:
            all_tags = []
            all_tags_map = {}

        img_dir = os.path.join(td, 'images')
        os.makedirs(img_dir, exist_ok=True)

        inc = Inc()
        if use_random:
            id_source = itertools.chain(
                _iter_artwork_ids_from_page(max_count_limit=1000),
                _iter_artwork_ids_in_queue(queue, inc),
            )
        else:
            id_source = _iter_artwork_ids_from_page(max_count_limit=1000),

        current_count = 0
        for username, post_id in id_source:
            suit_id = f'{username}_{post_id}'
            logging.info(f'Resource {suit_id!r} confirmed.')
            if suit_id in exist_sids:
                logging.info(f'Resource {suit_id!r} already crawled, skipped.')
                continue

            info = client.get_post(username, post_id)
            item_id = info['id']

            if 'client' not in info:
                logging.info(f'No client for this work {suit_id!r}, skipped')

            else:
                client_info = info['client']
                client_id = client_info['id']
                client_name = client_info['screen_name']

                creator_info = info['creator']
                creator_id = creator_info['id']
                creator_name = creator_info['screen_name']

                item_name = f'{creator_id}_{creator_name}__{client_id}_{client_name}__{item_id}'
                item_url = info.get('article_image_url')
                if not item_url:
                    logging.warning(f'No article image for this work {suit_id!r}.')

                else:
                    body = info['body']
                    cleaned_body = body
                    for url in extract_urls(body):
                        cleaned_body = cleaned_body.replace(url, '')
                    tags = [tag for tag in info['tag_list'] if tag.lower() in cleaned_body.lower()]
                    final_tags = []
                    for tag in tags:
                        if tag.lower() in all_tags_map:
                            tag = all_tags_map[tag.lower()]['name']
                        else:
                            tag_data = {
                                'name': tag,
                            }
                            all_tags.append(tag_data)
                            all_tags_map[tag.lower()] = tag_data
                        final_tags.append(tag)

                    _, ext = os.path.splitext(urlsplit(item_url).filename)
                    if not ext:
                        fmt = urlsplit(item_url).query_dict.get('fm')
                        if fmt:
                            ext = f'.{fmt}'
                    dst_file = os.path.join(img_dir, f'{item_name}{ext}')
                    logging.info(f'Downloading {item_url!r} to {dst_file!r} ...')
                    try:
                        download_file(item_url, filename=dst_file, session=client._session)
                    except (AssertionError, requests.exceptions.RequestException) as err:
                        logging.error(f'Download of {item_url!r} skipped due to error: {err!r}')

                    for sitem in info['similar_works']:
                        susername, spid = split_username_and_id_from_path(sitem['path'])
                        s_suit_id = f'{susername}_{spid}'
                        if s_suit_id not in exist_sids and s_suit_id not in queue_suit_ids:
                            queue_suit_ids.add(s_suit_id)
                            queue.append({
                                'username': susername,
                                'post_id': spid,
                            })

                    all_artworks.append({
                        'id': item_id,
                        'post_id': post_id,
                        'creator_id': creator_id,
                        'creator_name': creator_name,
                        'client_id': client_id,
                        'client_name': client_name,
                        'filename': os.path.basename(dst_file),
                        'packname': pack_name,
                        'body': body,
                        'article_image_url': info['article_image_url'],
                        'preview_url': info['preview_url'],
                        'og_image_url': info['og_image_url'],
                        'tags': final_tags,
                    })

            exist_sids.add(suit_id)
            pg.update()
            current_count += 1
            if current_count >= maxcnt:
                break
            if time.time() - start_time >= max_time_limit:
                break

        queue = queue[inc.v:]
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
            for tag in artwork_item['tags']:
                tags_analysis[tag] = tags_analysis.get(tag, 0) + 1
        all_cnt = len(all_artworks)
        for tag_item in all_tags:
            tag_item['count'] = tags_analysis.get(tag_item['name'], 0)
        all_tags = sorted(all_tags, key=lambda x: (-x['count'], x['name']))
        df_tags = pd.DataFrame(all_tags)
        df_tags.to_csv(os.path.join(export_dir, 'tags.csv'), index=False)
        with open(os.path.join(export_dir, 'exist_sids.json'), 'w') as f:
            json.dump(sorted(exist_sids), f)
        all_artworks = sorted(all_artworks, key=lambda x: x['id'])
        with open(os.path.join(export_dir, 'artworks.json'), 'w') as f:
            json.dump(all_artworks, f, ensure_ascii=False, sort_keys=True, indent=4)
        with open(os.path.join(export_dir, 'queue.json'), 'w') as f:
            json.dump(queue, f)

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
            print('## Tags', file=f)
            print('', file=f)
            print('Only some selected tags are shown.', file=f)
            print('', file=f)
            t_df_tags = df_tags
            t_df_tags = t_df_tags[t_df_tags['count'] >= 10]
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
    skb_newest_crawl(
        repository=os.environ['REMOTE_REPOSITORY_SKB_NEWEST'],
        maxcnt=20,
        max_time_limit=45 * 60,
    )
