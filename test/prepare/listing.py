import re
from itertools import islice
from typing import Tuple

from pyskeb.client.client import SkebClient
from .url import extract_urls

client = SkebClient()


def split_username_and_id_from_path(path: str) -> Tuple[str, int]:
    matching = re.fullmatch(r'^/?@(?P<username>[\s\S]+?)/works/(?P<work_id>\d+?)/?$', path)
    username, work_id = matching.group('username'), int(matching.group('work_id'))
    return username, work_id


def list_newest_posts(limit: int = 200):
    for item in islice(client.iter_art_pages(), limit):
        username, work_id = split_username_and_id_from_path(item['path'])
        yield username, work_id


def get_urls_from_post(username, work_id):
    post_data = client.get_post(username, work_id)
    text = f"{post_data.get('source_body', '')}\n{post_data.get('body', '')}"
    return extract_urls(text)
