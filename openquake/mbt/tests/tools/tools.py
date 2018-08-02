"""
"""

import os
import shutil

BASE_DATA_PATH = os.path.dirname(__file__)


def delete_and_create_project_dir(project_dir):
    """
    Creates a clean project directory
    """
    if os.path.exists(project_dir):
        shutil.rmtree(project_dir)
    os.makedirs(project_dir)
