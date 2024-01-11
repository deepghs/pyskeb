import json
import mimetypes
import os
import os.path as osp
import random
import re
import textwrap
import time
import warnings

import six
from gdown import download
from gdown.download import get_url_from_gdrive_confirmation, _get_session
from gdown.download_folder import _download_and_parse_google_drive_link
from gdown.parse_url import parse_url
from hbutils.system import urlsplit
from urlobject import URLObject

from .base import GenericException


class FileURLRetrievalError(GenericException):
    pass


def is_google_drive(url):
    return urlsplit(url).host == 'drive.google.com'


_last_time = time.time()
_wait_time = 5.0
_ratio = 0.1


def _get_wait_time():
    return (_ratio * 2 * random.random() + (1 - _ratio)) * _wait_time


def _wait():
    global _last_time
    _duration = _last_time + _get_wait_time() - time.time()
    if _duration > 0:
        time.sleep(_duration)
    _last_time = time.time()


def _get_filename_from_id(resource_id, use_cookies: bool = False):
    url = "https://drive.google.com/uc?id={id}".format(id=resource_id)
    url_origin = url

    sess, cookies_file = _get_session(use_cookies=use_cookies, return_cookies_file=True)
    gdrive_file_id, is_gdrive_download_link = parse_url(url, warning=True)

    while True:
        res = sess.get(url, stream=True, verify=True)

        if url == url_origin and res.status_code == 500:
            # The file could be Google Docs or Spreadsheets.
            url = "https://drive.google.com/open?id={id}".format(
                id=gdrive_file_id
            )
            continue

        if res.headers["Content-Type"].startswith("text/html"):
            m = re.search("<title>(.+)</title>", res.text)
            if m and m.groups()[0].endswith(" - Google Docs"):
                url = (
                    "https://docs.google.com/document/d/{id}/export"
                    "?format={format}".format(
                        id=gdrive_file_id,
                        format="docx" if format is None else format,
                    )
                )
                continue
            elif m and m.groups()[0].endswith(" - Google Sheets"):
                url = (
                    "https://docs.google.com/spreadsheets/d/{id}/export"
                    "?format={format}".format(
                        id=gdrive_file_id,
                        format="xlsx" if format is None else format,
                    )
                )
                continue
            elif m and m.groups()[0].endswith(" - Google Slides"):
                url = (
                    "https://docs.google.com/presentation/d/{id}/export"
                    "?format={format}".format(
                        id=gdrive_file_id,
                        format="pptx" if format is None else format,
                    )
                )
                continue
        elif (
                "Content-Disposition" in res.headers
                and res.headers["Content-Disposition"].endswith("pptx")
                and format not in {None, "pptx"}
        ):
            url = (
                "https://docs.google.com/presentation/d/{id}/export"
                "?format={format}".format(
                    id=gdrive_file_id,
                    format="pptx" if format is None else format,
                )
            )
            continue

        if use_cookies:
            if not osp.exists(osp.dirname(cookies_file)):
                os.makedirs(osp.dirname(cookies_file))
            # Save cookies
            with open(cookies_file, "w") as f:
                cookies = [
                    (k, v)
                    for k, v in sess.cookies.items()
                    if not k.startswith("download_warning_")
                ]
                json.dump(cookies, f, indent=2)

        if "Content-Disposition" in res.headers:
            # This is the file
            break
        if not (gdrive_file_id and is_gdrive_download_link):
            break

        # Need to redirect with confirmation
        try:
            url = get_url_from_gdrive_confirmation(res.text)
        except FileURLRetrievalError as e:
            message = (
                "Failed to retrieve file url:\n\n{}\n\n"
                "You may still be able to access the file from the browser:"
                "\n\n\t{}\n\n"
                "but Gdown can't. Please check connections and permissions."
            ).format(
                textwrap.indent("\n".join(textwrap.wrap(str(e))), prefix="\t"),
                url_origin,
            )
            raise FileURLRetrievalError(message)

    if gdrive_file_id and is_gdrive_download_link:
        content_disposition = six.moves.urllib_parse.unquote(
            res.headers["Content-Disposition"]
        )
        m = re.search(r"filename\*=UTF-8''(.*)", content_disposition)
        filename_from_url = m.groups()[0]
        filename_from_url = filename_from_url.replace(osp.sep, "_")
    else:
        filename_from_url = osp.basename(url)

    return filename_from_url


def get_google_resource_id(drive_url):
    drive_url = str(URLObject(drive_url).without_query())
    file_id, is_downloadable_link = parse_url(drive_url)
    if file_id is not None:
        return f'googledrive_{file_id}'
    else:
        sess = _get_session(use_cookies=False)
        return_code, gdrive_file = _download_and_parse_google_drive_link(sess, drive_url, remaining_ok=True)
        if gdrive_file is not None:
            fid = re.sub(r'\?[\s\S]+?$', '', gdrive_file.id)
            return f'googledrive_{fid}'
        else:
            return None


def get_google_drive_ids(drive_url):
    file_id, is_downloadable_link = parse_url(drive_url)
    if file_id is not None:
        return [(file_id, [_get_filename_from_id(file_id, use_cookies=False)])]
    else:
        sess = _get_session(use_cookies=False)
        return_code, gdrive_file = _download_and_parse_google_drive_link(sess, drive_url, remaining_ok=True)

        def _recursive(gf, paths):
            if 'folder' in gf.type:
                for item in gf.children:
                    yield from _recursive(item, [*paths, gf.name])
            else:
                ga_ext = [item.lower() for item in mimetypes.guess_all_extensions(gf.type)]
                f_ext = os.path.splitext(gf.name)[1]
                if not f_ext or (ga_ext and f_ext.lower() not in ga_ext):
                    name = gf.name + (mimetypes.guess_extension(gf.type) or '')
                else:
                    name = gf.name
                yield gf.id, [*paths, name]

        return list(_recursive(gdrive_file, []))


def download_google_to_directory(drive_url, output_directory):
    for id_, segments in get_google_drive_ids(drive_url):
        filename = os.path.join(output_directory, *segments)
        if os.path.dirname(filename):
            os.makedirs(os.path.dirname(filename), exist_ok=True)

        _wait()
        try:
            download(id=id_, output=filename, use_cookies=False)
        except Exception as err:
            warnings.warn(f'Error occurred, skipped: {err!r}')
