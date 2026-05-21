from queue import Queue
from logging import Logger
from .base_processor import BaseProcessor


class CustomEngine:
    def __init__(self, logger, camera_id, message_buffer) -> None:
        self.camera_id = camera_id
        self.logger: Logger = logger
        self.is_active: bool = True
        self.message_buffer: Queue = message_buffer
        self.processors: list[BaseProcessor] = []

    def add_processor(self, processor):
        if not isinstance(processor, BaseProcessor):
            raise TypeError(f"{processor} must be a BaseProcessor instance")
        self.processors.append(processor)
        self.logger.info(f"Processor added: {processor.name}")

    def __call__(self, iframe):
        frame = iframe.copy()
        metadata = {}
        for proc in self.processors:
            frame, metadata = proc.process(frame, metadata)
        if self.message_buffer and metadata:
            self.message_buffer.put(metadata)
        return frame

    def set_data(self, **kwargs):
        self.logger.info(f"KWARGS     ------->  {kwargs}")
        data = kwargs
