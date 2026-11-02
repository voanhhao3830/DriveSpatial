from typing import Any
from abc import ABCMeta, abstractmethod

from torch.utils.data import Dataset

class BaseDataLoader(Dataset):
    """Abstact base class for diffirent data sctructure"""
    
    @abstractmethod
    def _load_annotation(self, annotation_path: str):
        raise NotImplementedError
        
    @abstractmethod
    def load_metadata(self, data_path: str):
        raise NotImplementedError
    
    @abstractmethod
    def __call__(self, *args: Any, **kwds: Any):
        raise NotImplementedError