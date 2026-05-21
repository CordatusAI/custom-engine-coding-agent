from abc import ABC, abstractmethod


class BaseProcessor(ABC):
    def __init__(self, **kwargs):
        self.config = kwargs

    @abstractmethod
    def process(self, frame, metadata=None):
        if metadata is None:
            metadata = {}
        raise NotImplementedError

    @property
    def name(self):
        return self.__class__.__name__
