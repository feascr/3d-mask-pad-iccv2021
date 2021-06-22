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

PREPROCESSORS_CLS_NAMES = set()
PREPROCESSORS_REGISTRY = {}

def create_preprocessor(name, *args, **kwargs):
    if name not in PREPROCESSORS_REGISTRY:
        raise ValueError('Unknown preprocessor name {}'.format(name))

    return PREPROCESSORS_REGISTRY[name](*args, **kwargs)


def register_preprocessor(name):
    def register_preprocessor_cls(preprocessor_cls):
        if name in PREPROCESSORS_REGISTRY:
            raise ValueError('Cannot register duplicate preprocessor {}'.format(name))

        PREPROCESSORS_REGISTRY[name] = preprocessor_cls
        PREPROCESSORS_CLS_NAMES.add(preprocessor_cls.__name__)
        return preprocessor_cls

    return register_preprocessor_cls

FILE_ROOT = Path(__file__).parent
import_all_modules(FILE_ROOT, 'preprocessors')

def registered_preprocessors():
    return sorted(PREPROCESSORS_REGISTRY.keys())