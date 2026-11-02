"""
BEV Map Renderer Utility

Handles rendering of Bird's Eye View maps for different datasets.
Manages dataset initialization and caching to avoid reloading.
"""

import os
from typing import Dict, Any, Optional
from PIL import Image
from pathlib import Path

from sources.data_collection.enum.data_sources import DataSource
from sources.data_collection.models.data import (
    nuScenesDataset, 
    Argoverse2Dataset, 
    ONCEDataset, 
    WaymoDataset,
    TruckscenesDataset
)


class BEVRenderer:
    """
    Singleton class to render BEV maps for different autonomous driving datasets.
    Handles dataset initialization and caching.
    """
    
    _instance = None
    _initialized_datasets = {}
    
    def __new__(cls):
        """Ensure only one instance exists (Singleton pattern)."""
        if cls._instance is None:
            cls._instance = super(BEVRenderer, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize the BEV renderer."""
        if not hasattr(self, '_initialized'):
            self._initialized = True
            self._initialized_datasets = {}
    
    def _initialize_nuscenes(self, data_path: str, version: str = "v1.0-trainval") -> None:
        """
        Initialize nuScenes dataset if not already initialized.
        
        Args:
            data_path: Path to nuScenes dataset
            version: nuScenes version
        """
        source_key = DataSource.NUSCENES.value  # "nuScenes"
        if source_key not in self._initialized_datasets:
            try:
                from nuscenes.nuscenes import NuScenes
                nuScenesDataset.NUSCE = NuScenes(
                    version=version,
                    dataroot=data_path,
                    verbose=False
                )
                self._initialized_datasets[source_key] = True
                print(f"Initialized nuScenes dataset from {data_path}")
            except Exception as e:
                print(f"Warning: Failed to initialize nuScenes: {e}")
                import traceback
                traceback.print_exc()
                self._initialized_datasets[source_key] = False
    
    def _initialize_truckscenes(self, data_path: str) -> None:
        """
        Initialize TruckScenes dataset if not already initialized.
        
        Args:
            data_path: Path to TruckScenes dataset
        """
        source_key = DataSource.TRUCKSCENES.value  # "truckscenes"
        if source_key not in self._initialized_datasets:
            try:
                from truckscenes.truckscenes import TruckScenes
                TruckscenesDataset.TRUCKSCE = TruckScenes(
                    version='v1.0-trainval',
                    dataroot=data_path,
                    verbose=False
                )
                self._initialized_datasets[source_key] = True
                print(f"Initialized TruckScenes dataset from {data_path}")
            except Exception as e:
                print(f"Warning: Failed to initialize TruckScenes: {e}")
                import traceback
                traceback.print_exc()
                self._initialized_datasets[source_key] = False
    
    def render_bev_map(
        self, 
        data: Dict[str, Any], 
        output_path: str,
        dataset_configs: Optional[Dict[str, Dict[str, str]]] = None
    ) -> bool:
        """
        Render BEV map based on data source.
        
        Args:
            data: Data record containing metadata, lidar, and annotation information
            output_path: Path to save the rendered BEV map
            dataset_configs: Optional dictionary with dataset paths for initialization
                            Format: {"nuscenes": {"data_path": "..."}, ...}
        
        Returns:
            bool: True if rendering succeeded, False otherwise
        """
        metadata = data.get("metadata", {})
        source = metadata.get("source")
        
        # Helper function to find config with case-insensitive match
        def get_dataset_config(dataset_name: str) -> Optional[Dict[str, str]]:
            if not dataset_configs:
                return None
            for key, value in dataset_configs.items():
                if key.lower() == dataset_name.lower():
                    return value
            return None
        
        try:
            if source == DataSource.NUSCENES.value:  # "nuScenes"
                # Initialize nuScenes if needed
                if source not in self._initialized_datasets:
                    nuscenes_config = get_dataset_config("nuscenes")
                    if nuscenes_config:
                        data_path = nuscenes_config.get("data_path")
                        if data_path:
                            print(f"Initializing nuScenes with data_path: {data_path}")
                            self._initialize_nuscenes(data_path)
                    else:
                        print(f"Warning: No config found for nuScenes in dataset_configs: {dataset_configs}")
                
                # Check if initialized successfully
                if self._initialized_datasets.get(source) == False:
                    print(f"Warning: nuScenes initialization failed previously")
                    self._create_placeholder(output_path)
                    return False
                
                if not hasattr(nuScenesDataset, 'NUSCE') or nuScenesDataset.NUSCE is None:
                    print(f"Warning: nuScenes not initialized properly")
                    self._create_placeholder(output_path)
                    return False
                
                sample = nuScenesDataset.NUSCE.get('sample', metadata['frame_token'])
                nuScenesDataset.NUSCE.render_sample_data(
                    sample['data']['LIDAR_TOP'], 
                    with_anns=True, 
                    out_path=output_path
                )
                return True
                
            elif source == DataSource.ARGOVERSE2.value:
                # Argoverse2Dataset.render_topdown_map_with_hd uses DATA_PATH and SPLIT; set from metadata or config
                data_path = metadata.get("data_path")
                if not data_path and dataset_configs:
                    argoverse_config = get_dataset_config("argoverse2")
                    if argoverse_config:
                        data_path = argoverse_config.get("data_path")
                split = metadata.get("split", "val")
                if data_path:
                    Argoverse2Dataset.DATA_PATH = data_path
                    Argoverse2Dataset.SPLIT = split
                if not data_path:
                    print("Warning: No data_path for argoverse2 BEV (set dataset_configs or metadata['data_path'])")
                    self._create_placeholder(output_path)
                    return False
                annotation = data.get("annotation", {})
                Argoverse2Dataset.render_topdown_map_with_hd(
                    log_id=metadata["scene_token"],
                    lidar_timestamp_ns=metadata["frame_token"],
                    out_path=output_path,
                    bboxes_3d=annotation.get("bbox_3d", None),
                    labels_3d=annotation.get("label_3d", None),
                )
                return True
                
            elif source == DataSource.ONCE.value:
                lidar_data = data.get('lidar', {})
                annotation = data.get('annotation', {})
                ONCEDataset.render_topdown_map(
                    lidar_path=lidar_data.get('path') if isinstance(lidar_data, dict) else lidar_data,
                    ego_pose=None,
                    bbox_3d=annotation.get('bbox_3d', []),
                    label_3d=annotation.get('label_3d', []),
                    out_path=output_path
                )
                return True
                
            elif source == DataSource.WAYMO.value:
                lidar_data = data.get('lidar', {})
                annotation = data.get('annotation', {})
                WaymoDataset.render_topdown_map(
                    lidar_path=lidar_data.get('path') if isinstance(lidar_data, dict) else lidar_data,
                    ego_pose=None,
                    bboxes_3d=annotation.get('bbox_3d', []),
                    labels_3d=annotation.get('label_3d', []),
                    out_path=output_path
                )
                return True
                
            elif source == DataSource.TRUCKSCENES.value:  # "truckscenes"
                # Initialize TruckScenes if needed
                if source not in self._initialized_datasets:
                    truckscenes_config = get_dataset_config("truckscenes")
                    if truckscenes_config:
                        data_path = truckscenes_config.get("data_path")
                        if data_path:
                            print(f"Initializing TruckScenes with data_path: {data_path}")
                            self._initialize_truckscenes(data_path)
                
                # Check if initialized successfully
                if self._initialized_datasets.get(source) == False:
                    print(f"Warning: TruckScenes initialization failed previously")
                    self._create_placeholder(output_path)
                    return False
                
                if not hasattr(TruckscenesDataset, 'TRUCKSCE') or TruckscenesDataset.TRUCKSCE is None:
                    print(f"Warning: TruckScenes not initialized properly")
                    self._create_placeholder(output_path)
                    return False
                
                sample = TruckscenesDataset.TRUCKSCE.get('sample', metadata['frame_token'])
                TruckscenesDataset.visualize_fused_lidar(
                    truckscenes_instance=TruckscenesDataset.TRUCKSCE,
                    sample_token=sample['token'],
                    out_path=output_path
                )
                return True
                
            else:
                # Create placeholder if source is unknown
                print(f"Warning: Unknown source '{source}', creating placeholder")
                self._create_placeholder(output_path)
                return False
                
        except Exception as e:
            print(f"Warning: Failed to render BEV map for {source}: {e}")
            self._create_placeholder(output_path)
            return False
    
    def _create_placeholder(self, output_path: str) -> None:
        """Create a placeholder image when rendering fails."""
        placeholder = Image.new('RGB', (800, 800), color=(64, 64, 64))
        placeholder.save(output_path)
    
    @classmethod
    def get_instance(cls) -> 'BEVRenderer':
        """Get the singleton instance of BEVRenderer."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance


# Global instance for easy access
_bev_renderer = None


def get_bev_renderer() -> BEVRenderer:
    """
    Get the global BEV renderer instance.
    
    Returns:
        BEVRenderer: Singleton renderer instance
    """
    global _bev_renderer
    if _bev_renderer is None:
        _bev_renderer = BEVRenderer.get_instance()
    return _bev_renderer


def render_bev_map(
    data: Dict[str, Any], 
    output_path: str,
    dataset_configs: Optional[Dict[str, Dict[str, str]]] = None
) -> bool:
    """
    Convenience function to render BEV map using the global renderer.
    
    Args:
        data: Data record containing metadata, lidar, and annotation information
        output_path: Path to save the rendered BEV map
        dataset_configs: Optional dictionary with dataset paths for initialization
        
    Returns:
        bool: True if rendering succeeded, False otherwise
    """
    renderer = get_bev_renderer()
    return renderer.render_bev_map(data, output_path, dataset_configs)
