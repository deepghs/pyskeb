from urllib.parse import urljoin, quote_plus

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

    def get_user_page(self, offset: int = 0, limit: int = 90, sort: str = 'popularity'):
        return self._get(
            '/api/users',
            {
                'sort': sort,
                'offset': offset,
                'limit': limit,
            }
        )

    def iter_user_pages(self, limit: int = 90, sort: str = 'popularity'):
        # sort : popularity / date / request_masters / first_requesters
        offset = 0
        while True:
            items = self.get_user_page(offset, limit, sort)
            yield from items

            if not items:
                break
            offset += len(items)

    def get_user_info(self, screen_name: str):
        return self._get(f'/api/users/{quote_plus(screen_name)}')

    def get_work_page(self, screen_name: str, role: str = 'client', sort='date', offset: int = 0):
        return self._get(
            f'/api/users/{quote_plus(screen_name)}/works',
            {
                'role': role,
                'sort': sort,
                'offset': offset,
            }
        )

    def iter_work_pages(self, screen_name: str, role: str = 'client', sort='date'):
        # role : client/creator
        offset = 0
        while True:
            items = self.get_work_page(screen_name, role, sort, offset)
            yield from items

            if not items:
                break
            offset += len(items)

    def get_post(self, username, post_id):
        return self._get(f'/api/users/{username}/works/{post_id}')
