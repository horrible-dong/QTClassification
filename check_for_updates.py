# Copyright (c) QIU, Tian. All rights reserved.

import requests
from termcolor import cprint

from qtcls import __version__, __git_url__


def check_for_updates(current_version: str, git_url: str):
    print(f'\nFetching the latest release version from {git_url} ...\n')
    user, repo = git_url.split('/')[-2:]
    api_url = f'https://api.github.com/repos/{user}/{repo}/releases/latest'
    response = requests.get(api_url)
    if response.status_code == 200:
        release_info = response.json()
        latest_version = release_info['tag_name']
        cprint(f'current version: {current_version}', 'light_green', attrs=['bold'], end='  ')
        cprint(f'latest version: {latest_version}', 'light_magenta', attrs=['bold'])
    else:
        print('Failed to fetch the latest release version.')
    print('\n')


if __name__ == '__main__':
    check_for_updates(__version__, __git_url__)
