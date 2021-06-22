from pathlib import Path
import os
import sys
import importlib


def import_all_modules(root, base_module):
    for file in os.listdir(root):
        if file.endswith(('.py', '.pyc')) and not file.startswith('_'):
            module = file[: file.find('.py')]
            if module not in sys.modules:
                module_name = '.'.join([base_module, module])
                importlib.import_module(module_name)

DATASETS_CLS_NAMES = set()
DATASETS_REGISTRY = {}

def create_dataset(name, *args, **kwargs):
    if name not in DATASETS_REGISTRY:
        raise ValueError('Unknown dataset name {}'.format(name))

    return DATASETS_REGISTRY[name].create_dataset(*args, **kwargs)


def register_dataset(name):
    def register_preprocessor_cls(preprocessor_cls):
        if name in DATASETS_REGISTRY:
            raise ValueError('Cannot register duplicate datasets {}'.format(name))

        DATASETS_REGISTRY[name] = preprocessor_cls
        DATASETS_CLS_NAMES.add(preprocessor_cls.__name__)
        return preprocessor_cls

    return register_preprocessor_cls

FILE_ROOT = Path(__file__).parent
import_all_modules(FILE_ROOT, 'data')

def registered_preprocessors():
    return sorted(DATASETS_REGISTRY.keys())