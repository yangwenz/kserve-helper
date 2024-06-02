import click
from kservehelper.docker import build as build_image
from kservehelper.docker import push as push_image


def create_cli() -> click.Group:
    @click.group()
    def cli():
        pass

    @cli.command()
    @click.argument("folder", type=click.STRING)
    @click.option(
        "--quiet",
        type=click.BOOL,
        default=False,
        is_flag=True
    )
    def build(folder, quiet):
        build_image(folder, quiet)

    @cli.command()
    @click.argument("folder", type=click.STRING)
    def push(folder):
        push_image(folder)

    return cli


def main():
    _cli = create_cli()
    _cli()
