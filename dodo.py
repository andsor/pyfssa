# coding: utf8

import os

DOIT_CONFIG = {'default_tasks': []}

CITEULIKE_GROUP = 19073
BIBFILE = 'docs/pyfssa.bib'


def task_download_bib():
    """Download bibliography from CiteULike group"""

    return {
        'actions': [' '.join([
            'wget', '-O', BIBFILE,
            '"http://www.citeulike.org/bibtex/group/{}?incl_amazon=0&key_type=4"'.format(CITEULIKE_GROUP),
            ])],
        # 'file_dep': [CITEULIKE_COOKIES],
        'targets': [BIBFILE],
    }


def task_upload_doc():
    """Upload built html documentation to GitHub pages"""

    return {
        'actions': [[
            'ghp-import',
            '-n',  # Include a .nojekyll file in the branch.
            '-p',  # Push the branch to origin/{branch} after committing.
            os.path.join('docs', '_build', 'html')
        ]],
    }
