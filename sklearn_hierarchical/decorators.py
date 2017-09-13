from logging import getLogger


def logger(obj):
    """
    logging decorator, assigning an object the `logger` property.
    Can be used on a Python class, e.g:
        @logger
        class MyClass(object):
            ...
    """
    obj.logger = getLogger(obj.__name__)
    return obj
