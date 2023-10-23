import click
from ditk import logging

from .artists_idx import push_artists_sqlite
from .lololo import batch_process_newest
from .repack import repack_all

GLOBAL_CONTEXT_SETTINGS = dict(
    help_option_names=['-h', '--help']
)


@click.group(context_settings={**GLOBAL_CONTEXT_SETTINGS})
def cli():
    pass  # pragma: no cover


@cli.command('newest', context_settings={**GLOBAL_CONTEXT_SETTINGS})
@click.option('-n', '--number', type=int, default=200)
def newest(number):
    logging.try_init_root(logging.DEBUG)
    batch_process_newest(number)


@cli.command('pack', context_settings={**GLOBAL_CONTEXT_SETTINGS})
def pack():
    logging.try_init_root(logging.INFO)
    repack_all()


@cli.command('artists', context_settings={**GLOBAL_CONTEXT_SETTINGS})
def artists():
    logging.try_init_root(logging.INFO)
    push_artists_sqlite()


if __name__ == '__main__':
    cli()
