"""Dummy objects."""


class DummyProgress(object):

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def update(self, value):
        pass

    def close(self):
        pass
