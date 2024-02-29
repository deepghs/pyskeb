import os.path

from ditk import logging

from .project2 import mhs_project_crawl

if __name__ == '__main__':
    logging.try_init_root(logging.INFO)
    mhs_project_crawl(
        repository=os.environ['REMOTE_REPOSITORY_MHS_PROJECT'],
        maxcnt=300,
        zone=3,
    )
