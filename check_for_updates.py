# Copyright (c) QIU, Tian. All rights reserved.

import requests
from termcolor import cprint

from qtcls import __version__, __git_url__


def check_for_updates(git_url):
    print(f'\nFetching the latest release version from {git_url} ...\n')
    user, repo = git_url.split('/')[-2:]
    api_url = f'https://api.github.com/repos/{user}/{repo}/releases/latest'
    response = requests.get(api_url)
    if response.status_code == 200:
        release_info = response.json()
        __latest_version__ = release_info['tag_name']
        cprint(f'current version: {__version__}', 'light_green', attrs=['bold'], end='  ')
        cprint(f'latest version: {__latest_version__}', 'light_magenta', attrs=['bold'])
    else:
        print('Failed to fetch the latest release version.')
    print('\n')


if __name__ == '__main__':
    check_for_updates(__git_url__)
