"""
Simple project paths handler
"""
import os


def get_project_root() -> str:
    """returns the path to the project folder"""
    return os.path.abspath(os.curdir)


def get_datasets_dir() -> str:
    """returns the path to the datasets"""
    return os.path.join(get_project_root(), "datasets")
