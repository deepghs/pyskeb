import mimetypes
import os.path
import re
from functools import lru_cache
from urllib.parse import urljoin

import requests
from hbutils.system import urlsplit
from pyquery import PyQuery as pq

from pyskeb.utils.download import download_file


@lru_cache()
def _get_client_id():
    resp = requests.get('https://imgur.com/')
    resp.raise_for_status()

    main_js = None
    for item in pq(resp.text)('script[src]').items():
        if 'main' in item.attr('src'):
            main_js = urljoin(resp.url, item.attr('src'))

    assert main_js

    resp = requests.get(main_js)
    resp.raise_for_status()

    return re.findall(r'apiClientId:\s*\"([a-z\d]+)\"', resp.text)[0]


def _get_medias(id_):
    resp = requests.get(
        f'https://api.imgur.com/post/v1/albums/{id_}',
        params={
            'client_id': _get_client_id(),
            'include': 'media,adconfig,account',
        },
        headers={
            'Referer': 'https://imgur.com/',
            'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/117.0.0.0 Safari/537.36',
        }
    )
    resp.raise_for_status()
    return list(resp.json().get('media') or [])


def is_imgur(url):
    splitted = urlsplit(url)
    return splitted.host == 'imgur.com' and splitted.path_segments[1] == 'a'


def get_imgur_resource(url):
    splitted = urlsplit(url)
    assert splitted.path_segments[1] == 'a'
    return f'imgur_{splitted.path_segments[2]}'


def download_imgur_to_directory(url, output_directory):
    splitted = urlsplit(url)
    assert splitted.path_segments[1] == 'a'
    id_ = splitted.path_segments[2]
    for item in _get_medias(id_):
        if 'url' in item and 'name' in item:
            filename = item['name']
            if not os.path.splitext(filename)[1]:
                filename = filename + (mimetypes.guess_extension(item.get('mime_type')) or '')
            download_file(item['url'], os.path.join(output_directory, filename))
