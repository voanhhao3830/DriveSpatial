import logging
from PIL import Image
from typing import List, Dict, Any, Iterator

from .once import ONCEDataset
from .waymo import WaymoDataset
from .nuScenes import nuScenesDataset
from .argoverse2 import Argoverse2Dataset
from .truckscenes import TruckscenesDataset
from sources.data_collection.models.ai.sampling import PeriodicSampler

logger = logging.getLogger(__name__)

DATASET_REGISTRY = {
    "once": ONCEDataset,
    "waymo": WaymoDataset,
    "nuScenes": nuScenesDataset,
    "argoverse2": Argoverse2Dataset,
    "truckscenes": TruckscenesDataset,
}

SAMPLER_REGISTRY = {
    "periodic": PeriodicSampler,
}


class DriveSpatialDataLoader:
    """
    Data loader that loads multiple datasets and applies sampling to video sequences.
    """
    
    def __init__(self, datasets_config: List[Dict[str, Any]], sampler_config: Dict[str, Any]):
        """
        Initialize the data loader with datasets and sampler configuration.
        
        Args:
            datasets_config: List of dataset configuration dictionaries
            sampler_config: Sampler configuration dictionary with 'type' and 'frequency' keys
        """
        self.datasets_config = datasets_config
        self.sampler_config = sampler_config
        self.frequency = sampler_config.get("frequency", 5)
        self.datasets = self._load_datasets()
        self.sampler = self._load_sampler()
        self.processed_dataset = []
        logger.info(f"Initialized DriveSpatialDataLoader with {len(self.datasets)} datasets")


    def _load_sampler(self) -> PeriodicSampler:
        """
        Load and return an instance of the sampler.
        
        Returns:
            Sampler instance configured with the specified parameters
        """
        sampler_type = self.sampler_config.get("type", "periodic")
        if sampler_type not in SAMPLER_REGISTRY:
            raise ValueError(f"Unknown sampler type: {sampler_type}")
        
        sampler_class = SAMPLER_REGISTRY[sampler_type]
        # Extract frequency from config for PeriodicSampler
        frequency = self.sampler_config.get("frequency", 5)
        sampler_instance = sampler_class(frequency=frequency)
        logger.info(f"Loaded sampler: {sampler_type} with frequency: {frequency}")
        return sampler_instance


    def _load_datasets(self) -> Dict[str, Any]:
        """
        Create instances of datasets and return a dictionary mapping dataset names to instances.
        
        Returns:
            Dictionary mapping dataset names to dataset instances
        """
        datasets_dict = {}
        for data_config in self.datasets_config:
            name = data_config.get("name")
            data_path = data_config.get("data_path")
            ann_file = data_config.get("ann_file")
            
            if not all([name, data_path, ann_file]):
                raise ValueError(f"Missing required config for dataset: {data_config}")
            
            if name not in DATASET_REGISTRY:
                raise ValueError(f"Unknown dataset type: {name}")
            
            dataset_class = DATASET_REGISTRY[name]
            dataset_instance = dataset_class(data_path, ann_file)
            datasets_dict[name] = dataset_instance
            logger.info(f"Loaded dataset: {name} from {data_path}")
        
        return datasets_dict


    def __len__(self) -> int:
        """
        Returns the total number of scenes across all datasets.
        
        Returns:
            Total number of scenes
        """
        return sum([len(v) for _, v in self.datasets.items()])

    def __iter__(self) -> Iterator[List[Dict[str, Any]]]:
        """
        Iterate through datasets and yield sampled video frames.
        
        Yields:
            List of sampled frame information dictionaries
        """
        for data_name in self.datasets:
            # TODO: include more detail caching
            if data_name in self.processed_dataset:
                continue

            data_instance = self.datasets[data_name]
            logger.debug(f"Processing dataset: {data_name}")

            for video_data in data_instance:
                sampled_frames = self.sampler(video_data)
                yield (video_data, sampled_frames)


            self.processed_dataset.append(data_name)
            logger.info(f"Completed processing dataset: {data_name}")