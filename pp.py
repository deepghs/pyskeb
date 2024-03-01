import glob
import os.path
import random
import shutil

from ditk import logging
from hbutils.string import plural_word
from hbutils.system import TemporaryDirectory
from hfutils.operate import get_hf_client, get_hf_fs, download_archive_as_directory, upload_directory_as_archive
from hfutils.operate.base import _get_hf_token
from tqdm import tqdm
from waifuc.action import AlignMaxAreaAction, FileExtAction, ModeConvertAction
from waifuc.source import LocalSource

logging.try_init_root(logging.INFO)

hf_client = get_hf_client()
hf_fs = get_hf_fs()

all_repos = list(hf_client.list_datasets(author='StyleMuseum'))
random.shuffle(all_repos)

interval = 2
total = 4000
pg = tqdm(desc='Total', total=total)

if __name__ == '__main__':
    cnt = 0
    with TemporaryDirectory() as otd:
        save_dir = os.path.join(otd, 'save')

        for ritem in all_repos:
            repository = ritem.id
            if not hf_fs.exists(f'datasets/{repository}/dataset-raw.zip'):
                logging.info(f'No data pack in {repository!r}, skipped.')
                continue

            name = repository.split('/')[-1]
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

                if imgs_cnt < 100:
                    logging.info(f'Not enough images in repository {repository!r}, skipped.')
                    continue

                os.makedirs(save_dir, exist_ok=True)
                LocalSource(ttd, shuffle=True).attach(
                    ModeConvertAction('RGB', 'white'),
                    AlignMaxAreaAction(1024),
                    FileExtAction(quality=90, ext='.jpg'),
                )[:100].export(os.path.join(save_dir, name))

                cnt += 1
                pg.update()
                if cnt % interval == 0:
                    archive_name = f'pack_{int(cnt // interval)}.zip'
                    upload_directory_as_archive(
                        local_directory=save_dir,
                        archive_in_repo=archive_name,
                        repo_id='DeepBase/artists_packs',
                        hf_token=os.environ['HF_TOKEN_X']
                    )
                    shutil.rmtree(save_dir)

                if cnt >= total:
                    break
