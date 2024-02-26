from urllib.parse import urljoin

import requests

from ..utils import get_random_ua

SKEB_WEBISTE = 'https://skeb.jp'


class SkebClient:

    def __init__(self):
        self._session = requests.session()
        self._session.headers.update({
            'Referer': 'https://skeb.jp',
            'User-Agent': get_random_ua(),
            "Authorization": "Bearer null",
            "Accept": "application/json, text/plain, */*",
        })

    def _get(self, url, params=None):
        while True:
            resp = self._session.get(urljoin(SKEB_WEBISTE, url), params=params or {})
            if not resp.ok and resp.status_code == 429 and 'request_key' in resp.cookies:
                continue

            resp.raise_for_status()
            return resp.json()

    def get_page(self, offset: int = 0, limit: int = 90):
        return self._get(
            '/api/works',
            {
                'sort': 'date',
                'genre': 'art',
                'offset': offset,
                'limit': limit,
            }
        )

    def iter_art_pages(self, limit: int = 90):
        offset = 0
        while True:
            items = self.get_page(offset, limit)
            yield from items

            if not items:
                break
            offset += len(items)

    def get_post(self, username, post_id):
        return self._get(f'/api/users/{username}/works/{post_id}')
