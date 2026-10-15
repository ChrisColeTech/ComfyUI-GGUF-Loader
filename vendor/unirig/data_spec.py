from abc import ABC, abstractmethod
from dataclasses import fields

class ConfigSpec(ABC):
    @classmethod
    def check_keys(cls, config):
        expect = [field.name for field in fields(cls)]
        for key in config.keys():
            if key not in expect:
                raise ValueError(f"expect names {expect} in {cls.__name__}, found {key}")
    
    @classmethod
    @abstractmethod
    def parse(cls, config):
        pass
