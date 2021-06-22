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

MODELS_CLS_NAMES = set()
MODELS_REGISTRY = {}

def create_model(name, *args):
    assert name in MODELS_REGISTRY, 'uknown model name'

    return MODELS_REGISTRY[name](*args)


def register_model(name):
    def register_model_cls(model_cls):
        if name in MODELS_REGISTRY:
            raise ValueError('Cannot register duplicate model {}'.format(name))

        MODELS_REGISTRY[name] = model_cls
        MODELS_CLS_NAMES.add(model_cls.__name__)
        return model_cls

    return register_model_cls


GETTERS_CLS_NAMES = set()
GETTERS_REGISTRY = {}

def create_getter(name, *args, **kwargs):
    assert name in GETTERS_REGISTRY, 'uknown backbone getter name {}'.format(name)

    return GETTERS_REGISTRY[name](*args, **kwargs)


def register_getter(name):
    def register_getter_cls(gtr_cls):
        if name in GETTERS_REGISTRY:
            raise ValueError('Cannot register duplicate getter {}'.format(name))

        GETTERS_REGISTRY[name] = gtr_cls
        GETTERS_CLS_NAMES.add(gtr_cls.__name__)
        return gtr_cls

    return register_getter_cls

FILE_ROOT = Path(__file__).parent
import_all_modules(FILE_ROOT, 'models')

def registered_models():
    return sorted(MODELS_REGISTRY.keys())

def registered_getters():
    return sorted(GETTERS_REGISTRY.keys())