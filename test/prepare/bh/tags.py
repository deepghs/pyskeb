import os

import pandas as pd
from hbutils.system import TemporaryDirectory
from hfutils.operate import upload_directory_as_directory
from pyquery import PyQuery as pq
from tqdm import tqdm
from waifuc.source import ThreeDBooruSource
from waifuc.utils import srequest

from test.prepare.base import hf_client, hf_fs, hf_token


def _get_tags_data():
    from .index import _safe_json

    s = ThreeDBooruSource(['solo'])
    resp = srequest(s.session, 'GET', 'http://behoimi.org/tag/index.json?limit=0')
    data = _safe_json(resp.text)

    df = pd.DataFrame(data)
    df = df.sort_values(['id'], ascending=[True])
    return df


def _get_tag_aliases_data():
    s = ThreeDBooruSource(['solo'])
    session = s.session

    page = 1
    data = []
    pg_page = tqdm(desc='Aliases Page')
    pg = tqdm(desc='Aliases')
    exist_names = set()
    while True:
        resp = session.get('http://behoimi.org/tag_alias/index', params={
            'page': str(page)
        })
        resp.raise_for_status()

        table = pq(resp.text)('#aliases table')
        changed = False
        for row in table('tr').items():
            if not row('td'):
                continue

            from_text = row('td:nth-child(2) a').text().strip()
            to_text = row('td:nth-child(3) a').text().strip()
            reason_text = row('td:nth-child(4)').text().strip()
            if (from_text, to_text) not in exist_names:
                exist_names.add((from_text, to_text))
                data.append({
                    'alias': from_text,
                    'tag': to_text,
                    'reason': reason_text,
                })

                pg.update()
                changed = True

        pg_page.update()
        page += 1
        if not changed:
            break

    df = pd.DataFrame(data)
    return df


def crawl_index(repository: str):
    if not hf_client.repo_exists(repo_id=repository, repo_type='dataset'):
        hf_client.create_repo(repo_id=repository, repo_type='dataset', private=True)
        attr_lines = hf_fs.read_text(f'datasets/{repository}/.gitattributes').splitlines(keepends=False)
        attr_lines.append('*.json filter=lfs diff=lfs merge=lfs -text')
        attr_lines.append('*.csv filter=lfs diff=lfs merge=lfs -text')
        hf_fs.write_text(
            f'datasets/{repository}/.gitattributes',
            os.linesep.join(attr_lines),
        )

    df_tags = _get_tags_data()
    df_tag_aliases = _get_tag_aliases_data()

    with TemporaryDirectory() as td:
        df_tags.to_csv(os.path.join(td, 'index_tags.csv'), index=False)
        df_tag_aliases.to_csv(os.path.join(td, 'index_tag_aliases.csv'), index=False)

        upload_directory_as_directory(
            repo_id=repository,
            repo_type='dataset',
            local_directory=td,
            path_in_repo='.',
            hf_token=hf_token,
        )


if __name__ == '__main__':
    crawl_index(
        repository=os.environ['REMOTE_REPOSITORY_BH'],
    )
