import logging
import time

from hbutils.string import plural_word

from .listing import get_urls_from_post, list_newest_posts
from .process import try_process_url


def batch_process_via_iterator(f_iter, timespan: float = 2.5):
    for username, work_id in f_iter:
        _last_time = time.time()
        urls = get_urls_from_post(username, work_id)
        logging.info(f'{plural_word(len(urls), "url")} found in @{username}/works/{work_id}')

        for url in urls:
            try_process_url(url, prefix=f'{username}_{work_id}_')

        _duration = _last_time + timespan - time.time()
        if _duration > 0.0:
            time.sleep(_duration)


def batch_process_newest(limit: int = 100, timespan: float = 2.5):
    batch_process_via_iterator(
        list_newest_posts(limit),
        timespan=timespan,
    )
