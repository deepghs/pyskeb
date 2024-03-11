import os

from ditk import logging

from pyskeb.utils import get_requests_session


def refresh_pp(bd_token, zone):
    session = get_requests_session()
    session.headers.update({
        'Authorization': f'Bearer {bd_token}',
    })

    r = session.get(
        'https://api.brightdata.com/zone/route_ips',
        params={'zone': zone}
    )
    ips = list(filter(bool, r.text.splitlines(keepends=False)))
    logging.info(f'Current ips: {ips!r}.')

    data = {'zone': zone, 'ips': ips}
    r = session.post('https://api.brightdata.com/zone/ips/refresh', data=data)
    r.raise_for_status()
    logging.info('Refresh success.')


if __name__ == '__main__':
    logging.try_init_root(logging.INFO)
    refresh_pp(
        bd_token=os.environ['BD_TOKEN'],
        zone=os.environ['BD_MHS_ZONE']
    )
