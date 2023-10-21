import click
from ditk import logging

from .lololo import batch_process_newest

GLOBAL_CONTEXT_SETTINGS = dict(
    help_option_names=['-h', '--help']
)


@click.group(context_settings={**GLOBAL_CONTEXT_SETTINGS})
def cli():
    pass  # pragma: no cover


@cli.command('newest', context_settings={**GLOBAL_CONTEXT_SETTINGS})
@click.option('-n', '--number', type=int, default=200)
def newest(number):
    logging.try_init_root(logging.INFO)
    batch_process_newest(number)


if __name__ == '__main__':
    cli()
