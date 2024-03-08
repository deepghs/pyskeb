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

from pyskeb.utils import get_random_mobile_ua, download_file, get_requests_session
from ..base import hf_fs, hf_client, hf_token


def bact_crawl(repository: str, maxcnt: int = 100):
    session = get_requests_session()
    session.headers.update({
        'User-Agent': get_random_mobile_ua(),
        'Referer': 'https://www.bilibili.com/',
    })

    def _name_safe(name_text):
        return re.sub(r'[\W_]+', '_', name_text).strip('_')

    logging.info('Getting SPI ...')
    resp = session.get('https://api.bilibili.com/x/frontend/finger/spi')
    resp.raise_for_status()
    b3 = resp.json()['data']['b_3']

    logging.info('Access act card list ...')
    resp = session.get('https://www.bilibili.com/h5/mall/v2/cardSubject/42')
    resp.raise_for_status()

    if not hf_client.repo_exists(repo_id=repository, repo_type='dataset'):
        hf_client.create_repo(repo_id=repository, repo_type='dataset', private=True)

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

        img_dir = os.path.join(td, 'videos')
        os.makedirs(img_dir, exist_ok=True)

        resp = session.get(
            'https://api.bilibili.com/x/garb/card/subject/list',
            params={
                'buvid': b3,
                'subject_id': '42'
            }
        )
        resp.raise_for_status()

        current_count = 0
        lst = resp.json()['data']['subject_card_list']
        for item in lst:
            act_id = item['act_id']
            act_name = item['act_name']
            lottery_id = item['lottery_id']
            suit_id = f'act_{act_id}_lottery_{lottery_id}'
            logging.info(f'Suit item {suit_id!r} (name: {act_name!r}) detected.')
            if suit_id in exist_sids:
                logging.info(f'Suit item {suit_id!r} already crawled, skipped.')
                continue
            if not item.get('act_link'):
                logging.info(f'No act link found for {suit_id!r}, skipped.')
                continue

            resp = session.get(
                'https://api.bilibili.com/x/vas/dlc_act/lottery_home_detail',
                params={
                    'act_id': str(act_id),
                    'lottery_id': str(lottery_id),
                }
            )
            resp.raise_for_status()
            lottery_name = resp.json()['data']['name']

            for li_id, li_item in enumerate(resp.json()['data']['item_list']):
                vlist = li_item['card_info']['video_list'] or []
                vname = f'act_{act_id}__{_name_safe(act_name)}__lottery_{lottery_id}__{_name_safe(lottery_name)}__{li_id}'
                if vlist:
                    vurl = vlist[0]
                    logging.info(f'Video URL: {vurl!r}')

                    _, ext = os.path.splitext(urlsplit(vurl).filename)
                    dst_file = os.path.join(img_dir, f'{vname}{ext}')
                    logging.info(f'Downloading {vurl!r} to {dst_file!r} ...')
                    download_file(vurl, filename=dst_file, session=session)

            exist_sids.add(suit_id)
            pg.update()
            current_count += 1
            if current_count >= maxcnt:
                break

        if not os.listdir(img_dir):
            logging.warning('No videos found, quit.')
            return

        item_cnt = len(os.listdir(img_dir))
        from ..repack import _timestamp
        img_pack_file = os.path.join(td, f'act_pack_{_timestamp()}.zip')
        with zipfile.ZipFile(img_pack_file, 'w') as zf:
            for file in os.listdir(img_dir):
                zf.write(os.path.join(img_dir, file), file)
                os.remove(os.path.join(img_dir, file))

        filename = os.path.basename(img_pack_file)
        records.append({
            'Filename': filename,
            'Videos': item_cnt,
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
    bact_crawl(
        repository=os.environ['REMOTE_REPOSITORY_BACT_V'],
        maxcnt=10000,
    )
