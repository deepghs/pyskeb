import mimetypes
import os
import os.path as osp
import random
import re
import textwrap
import time
import warnings
from http.cookiejar import MozillaCookieJar

from gdown import download
from gdown._indent import indent
from gdown.download import get_url_from_gdrive_confirmation, _get_session, _get_filename_from_response
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


def _get_filename_from_id(resource_id, use_cookies: bool = False, proxy=None, fuzzy=False, verify=True):
    url = "https://drive.google.com/uc?id={id}".format(id=resource_id)
    user_agent = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.95 Safari/537.36"  # NOQA: E501

    url_origin = url

    sess, cookies_file = _get_session(
        proxy=proxy,
        use_cookies=use_cookies,
        user_agent=user_agent,
        return_cookies_file=True,
    )

    gdrive_file_id, is_gdrive_download_link = parse_url(url, warning=not fuzzy)

    if fuzzy and gdrive_file_id:
        # overwrite the url with fuzzy match of a file id
        url = "https://drive.google.com/uc?id={id}".format(id=gdrive_file_id)
        url_origin = url
        is_gdrive_download_link = True

    while True:
        res = sess.get(url, stream=True, verify=verify)

        if not (gdrive_file_id and is_gdrive_download_link):
            break

        if url == url_origin and res.status_code == 500:
            # The file could be Google Docs or Spreadsheets.
            url = "https://drive.google.com/open?id={id}".format(id=gdrive_file_id)
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
            cookie_jar = MozillaCookieJar(cookies_file)
            for cookie in sess.cookies:
                cookie_jar.set_cookie(cookie)
            cookie_jar.save()

        if "Content-Disposition" in res.headers:
            # This is the file
            break

        # Need to redirect with confirmation
        try:
            url = get_url_from_gdrive_confirmation(res.text)
        except FileURLRetrievalError as e:
            print(e)
            message = (
                "Failed to retrieve file url:\n\n{}\n\n"
                "You may still be able to access the file from the browser:"
                "\n\n\t{}\n\n"
                "but Gdown can't. Please check connections and permissions."
            ).format(
                indent("\n".join(textwrap.wrap(str(e))), prefix="\t"),
                url_origin,
            )
            raise FileURLRetrievalError(message)

    filename_from_url = None
    if gdrive_file_id and is_gdrive_download_link:
        filename_from_url = _get_filename_from_response(response=res)
    if filename_from_url is None:
        filename_from_url = osp.basename(url)
    if filename_from_url is not None:
        filename_from_url = filename_from_url.encode('ISO-8859-1').decode()
    return filename_from_url


def get_google_resource_id(drive_url):
    drive_url = str(URLObject(drive_url).without_query())
    file_id, is_downloadable_link = parse_url(drive_url)
    if file_id is not None:
        return f'googledrive_{file_id}'
    else:
        sess = _get_session(use_cookies=False, proxy=None, user_agent=None)
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
        sess = _get_session(use_cookies=False, proxy=None, user_agent=None)
        return_code, gdrive_file = _download_and_parse_google_drive_link(sess, drive_url, remaining_ok=True)
        if not gdrive_file:
            return []

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
    try:
        for id_, segments in get_google_drive_ids(drive_url):
            filename = os.path.join(output_directory, *segments)
            if os.path.dirname(filename):
                os.makedirs(os.path.dirname(filename), exist_ok=True)

            _wait()
            if not download(id=id_, output=filename, use_cookies=False):
                warnings.warn(f'Error occurred, skip this directory: {drive_url!r}!')
                break
    except Exception as err:
        warnings.warn(f'Skipped for {drive_url!r}, err: {err!r}')
