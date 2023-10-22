import os.path
import zipfile

from hbutils.system import urlsplit
from urlobject import URLObject

from pyskeb.utils.download import download_file


def is_dropbox(url):
    splitted = urlsplit(url)
    return splitted.host in {'dropbox.com', 'www.dropbox.com'} and splitted.query_dict.get('dl') == '0'


def get_dropbox_resource(url):
    splitted = urlsplit(url)
    assert splitted.host in {'dropbox.com', 'www.dropbox.com'}
    return '_'.join(['dropbox', *(item for item in splitted.path_segments if item)])


def download_dropbox_to_directory(url, output_directory):
    splitted = urlsplit(url)
    assert splitted.host in {'dropbox.com', 'www.dropbox.com'}

    download_url = URLObject(url).set_query_param('dl', '1')
    target_file = download_file(download_url, output_directory=output_directory)
    if os.path.splitext(target_file)[1] == '.zip':
        with zipfile.ZipFile(target_file, 'r') as zf:
            zf.extractall(output_directory)
        os.remove(target_file)
