import re
from itertools import islice

from pyskeb.client.client import SkebClient
from .url import extract_urls

client = SkebClient()


def list_newest_posts(limit: int = 200):
    for item in islice(client.iter_art_pages(), limit):
        matching = re.fullmatch(r'^/?@(?P<username>[\s\S]+?)/works/(?P<work_id>\d+?)/?$', item['path'])
        username, work_id = matching.group('username'), int(matching.group('work_id'))

        yield username, work_id


def get_urls_from_post(username, work_id):
    post_data = client.get_post(username, work_id)
    text = f"{post_data.get('source_body', '')}\n{post_data.get('body', '')}"
    return extract_urls(text)
