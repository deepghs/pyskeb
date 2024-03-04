import glob
import json
import os.path
import random
import shutil

import dateparser
from ditk import logging
from hbutils.string import plural_word
from hbutils.system import TemporaryDirectory
from hfutils.operate import download_archive_as_directory, upload_directory_as_archive
from hfutils.operate import get_hf_client, get_hf_fs
from hfutils.operate.base import _get_hf_token
from tqdm import tqdm
from waifuc.action import AlignMaxAreaAction, FileExtAction, ModeConvertAction, FilterAction, ProcessAction
from waifuc.model import ImageItem
from waifuc.source import LocalSource

logging.try_init_root(logging.INFO)

hf_client = get_hf_client()
hf_fs = get_hf_fs()
hf_fs_2 = get_hf_fs(os.environ['HF_TOKEN_X'])

remote_repo = 'DeepBase/artists_packs_new'

if hf_fs_2.exists(f'datasets/{remote_repo}/exist_names.json'):
    exist_names = json.loads(hf_fs_2.read_text(f'datasets/{remote_repo}/exist_names.json'))
else:
    exist_names = []
exist_names = set(exist_names)
logging.info(f'{plural_word(len(exist_names), "existing name")} found.')

all_repos = list(hf_client.list_datasets(author='StyleMuseum'))
random.shuffle(all_repos)

interval = 5
# total = 2500
total = 5
pg = tqdm(desc='Total', total=total)
pg.update(len(exist_names))


class TimeFilterAction(FilterAction):
    def __init__(self, time_threshold: float):
        self.time_threshold = time_threshold

    def check(self, item: ImageItem) -> bool:
        return dateparser.parse(item.meta['danbooru']['created_at']).timestamp() >= self.time_threshold


class FileRenameAction(ProcessAction):
    def process(self, item: ImageItem) -> ImageItem:
        danbooru_id = item.meta['danbooru']["id"]
        timestamp = int(dateparser.parse(item.meta['danbooru']['created_at']).timestamp())
        meta_info = {
            **item.meta,
            'filename': f'danbooru_{danbooru_id}_{timestamp}.jpg',
        }
        return ImageItem(item.image, meta_info)


if __name__ == '__main__':
    cnt = len(exist_names)
    with TemporaryDirectory() as otd:
        save_dir = os.path.join(otd, 'save')

        for ritem in all_repos:
            repository = ritem.id
            if not hf_fs.exists(f'datasets/{repository}/dataset-raw.zip'):
                logging.info(f'No data pack in {repository!r}, skipped.')
                continue

            name = repository.split('/')[-1]
            if name in exist_names:
                logging.info(f'Name {name!r} already crawled, skipped.')
                continue

            with TemporaryDirectory() as ttd:
                download_archive_as_directory(
                    repo_id=repository,
                    repo_type='dataset',
                    file_in_repo='dataset-raw.zip',
                    local_directory=ttd,
                    hf_token=_get_hf_token(),
                )

                imgs_cnt = len(glob.glob(os.path.join(ttd, '.*.json')))
                logging.info(f'{plural_word(imgs_cnt, "image")} found in {repository!r}.')

                if imgs_cnt < 150:
                    logging.info(f'Not enough images in repository {repository!r}, skipped.')
                    exist_names.add(name)
                    continue

                all_created_ats = []
                for json_file in glob.glob(os.path.join(ttd, '.*.json')):
                    with open(json_file, 'r') as f:
                        data = json.load(f)
                        created_at = dateparser.parse(data['danbooru']['created_at']).timestamp()
                        all_created_ats.append(created_at)
                min_time, max_time = min(all_created_ats), max(all_created_ats)
                draw_duration = max_time - min_time
                if draw_duration < 60 * 60 * 24 * 365 * 3:
                    logging.info(f'Duration of artist {repository!r} too short, skipped.')
                    exist_names.add(name)
                    continue
                all_created_ats = sorted(all_created_ats, key=lambda x: -x)
                time_threshold = all_created_ats[100]

                os.makedirs(save_dir, exist_ok=True)
                LocalSource(ttd, shuffle=True).attach(
                    TimeFilterAction(time_threshold),
                    ModeConvertAction('RGB', 'white'),
                    AlignMaxAreaAction(1024),
                    FileRenameAction(),
                    FileExtAction(quality=90, ext='.jpg'),
                )[:100].export(os.path.join(save_dir, name))

                cnt += 1
                exist_names.add(name)
                pg.update()
                if cnt % interval == 0:
                    archive_name = f'pack_{int(cnt // interval)}.zip'
                    upload_directory_as_archive(
                        local_directory=save_dir,
                        archive_in_repo=archive_name,
                        repo_id=remote_repo,
                        hf_token=os.environ['HF_TOKEN_X']
                    )
                    hf_fs_2.write_text(
                        f'datasets/{remote_repo}/exist_names.json',
                        json.dumps(sorted(exist_names)),
                    )
                    shutil.rmtree(save_dir)

                if cnt >= total:
                    break
