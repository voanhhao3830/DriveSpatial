import math
import pickle
import logging
import numpy as np

from nuscenes.nuscenes import NuScenes
from typing import Dict, Any, List, Tuple
from nuscenes.eval.common.utils import Quaternion
from sources.data_collection.enum.data_sources import DataSource
from .base import BaseDataLoader

logger = logging.getLogger(__name__)

class nuScenesDataset(BaseDataLoader):
    """
    Dataset loader for nuScenes dataset.
    Returns video information including camera views, lidar, annotations, and calibration.
    """
    NAME = DataSource.NUSCENES.value
    NUSCE = None
    
    CLASS_NAME = [
        "adult", "child", "stroller", "personal_mobility", "police_officer",
        "construction_worker", "car", "motorcycle", "bicycle", "bus",
        "truck", "construction", "police", "trailer", "barrier",
        "trafficcone", "pushable_pullable", "debris"
    ]
    def __init__(self, data_path: str, ann_file: str):
        """
        Initialize nuScenes dataset loader.
        
        Args:
            data_path: Path to the dataset root directory
            ann_file: Path to the annotation pickle file
        """
        self.data_path = data_path
        self.ann_file = ann_file
        self.scene_set = set()
        self.data_infos = self._load_annotation(ann_file)
        nuScenesDataset.NUSCE = NuScenes(version='v1.0-trainval', dataroot=data_path, verbose=True)
        logger.info(f"Loaded nuScenes dataset with {len(self.data_infos)} frames from {ann_file}")

        
    def _load_annotation(self, annotation_path: str) -> List[Dict[str, Any]]:
        """
        Load annotation file and return sorted data infos.
        
        Args:
            annotation_path: Path to the annotation pickle file
            
        Returns:
            List of sorted data information dictionaries
        """
        with open(annotation_path, 'rb') as f:
            data = pickle.load(f)
        data_infos = list(sorted(data['infos'], key=lambda e: e['timestamp']))
        self.metadata = data.get('metadata', {})
        self.version = self.metadata.get('version', 'unknown')
        logger.debug(f"Loaded {len(data_infos)} frames, version: {self.version}")
        return data_infos

    
    def load_metadata(self, data_path: str) -> dict:
        """
        Load metadata from the dataset.
        
        Args:
            data_path: Path to the dataset
            
        Returns:
            Dictionary containing metadata
        """
        return self.metadata if hasattr(self, 'metadata') else {}
    
    def __len__(self):
        """
        Returns the number of unique scenes in the dataset.
        """
        # Count unique scenes by iterating through data_infos
        unique_scenes = set()
        for info in self.data_infos:
            unique_scenes.add(info['scene_token'])
        return len(unique_scenes)
    
    def _parse_info(self, index: int) -> Dict[str, Any]:
        """
        Parse frame information at the given index.
        
        Args:
            index: Index of the frame in data_infos
            
        Returns:
            Dictionary containing views, lidar, annotation, calibration, and metadata
        """
        if index >= len(self.data_infos):
            raise IndexError(f"Index {index} out of range for {len(self.data_infos)} frames")
        
        results: Dict[str, Any] = {
            "views": {},
            "lidar": "",
            "annotation": {},
            "calibration": {},
            "metadata": {}
        }
        infos = self.data_infos[index]
        # ================= IMAGE ================= 
        cams_data = infos['cams']
        for camera_name, cam_data in cams_data.items():
            results['views'][camera_name] = cam_data["data_path"].replace("./data/nuscenes/", self.data_path) # remove relative path
        
        # ================= Lidar ================= 
        results['lidar'] = {
            "path": infos['lidar_path'].replace("./data/nuscenes/", self.data_path)
        }
        results['CAM_FRONT'] = {
            "path": infos['cams']['CAM_FRONT']['data_path'].replace("./data/nuscenes/", self.data_path)
        }
        
        # ================= Calibration ================= 
        e2g_rotation = Quaternion(infos['ego2global_rotation']).rotation_matrix
        e2g_translation = infos['ego2global_translation']
        l2e_rotation = Quaternion(infos['lidar2ego_rotation']).rotation_matrix
        l2e_translation = infos['lidar2ego_translation']
        e2g_matrix = convert_egopose_to_matrix_numpy(e2g_rotation, e2g_translation)
        l2e_matrix = convert_egopose_to_matrix_numpy(l2e_rotation, l2e_translation)
        ego_pose =  e2g_matrix @ l2e_matrix # lidar2global

        ego_pose_inv = invert_matrix_egopose_numpy(ego_pose)
        # yaw = np.arctan2(ego_pose[1, 0], ego_pose[0, 0])
        # rotation_matrix = np.array([
        #     [np.cos(yaw), -np.sin(yaw), 0],
        #     [np.sin(yaw),  np.cos(yaw), 0],
        #     [0, 0, 1]
        # ])
        gt_boxes = np.array(infos['gt_boxes'])
        # if len(gt_boxes) > 0:
        #     # Transform bbox positions
        #     # positions = gt_boxes[:, :3]  # x, y, z
        #     # positions_transformed = (rotation_matrix @ positions.T).T
        #     # gt_boxes[:, :3] = positions_transformed
        #     gt_boxes[:, 6] = gt_boxes[:, 6] + yaw
        
        # ================= Annotation ================= 
        # 3D Annotation
        results['annotation']['bbox_3d'] = gt_boxes
        results['annotation']['label_3d'] = infos['gt_names']
        
        # 2D Annotation
        results['annotation']['bbox_2d'] = infos['bboxes2d']
        results['annotation']['label_2d'] = infos['labels2d']
        # 3D Track ID
        results['annotation']['track_3d'] = infos['gt_instance_tokens'] 
        results['annotation']['track_2d'] = infos['gt_2dinstance_tokens'] 
        
        # GT velocity
        results['annotation']['velocity'] = infos['gt_velocity'] 
        results['annotation']['visibility'] = infos['visibilities']
    
        lidar2img_rts = []
        intrinsics = []
        extrinsics = []
        img_timestamp = []
        
        for cam_type, cam_info in infos['cams'].items():
            img_timestamp.append(cam_info['timestamp'] / 1e6)
            
            # obtain lidar to image transformation matrix
            cam2lidar_r = cam_info['sensor2lidar_rotation']
            cam2lidar_t = cam_info['sensor2lidar_translation']
            cam2lidar_rt = convert_egopose_to_matrix_numpy(cam2lidar_r, cam2lidar_t)
            lidar2cam_rt = invert_matrix_egopose_numpy(cam2lidar_rt)

            intrinsic = cam_info['cam_intrinsic']
            viewpad = np.eye(4)
            viewpad[:intrinsic.shape[0], :intrinsic.shape[1]] = intrinsic
            lidar2img_rt = (viewpad @ lidar2cam_rt)
            intrinsics.append(viewpad)
            extrinsics.append(lidar2cam_rt)
            lidar2img_rts.append(lidar2img_rt)
        
        results['calibration'] = dict(
            img_timestamp=img_timestamp,
            lidar2img=lidar2img_rts,
            intrinsics=intrinsics,
            extrinsics=extrinsics,
            
            ego_pose=ego_pose,
            ego_pose_inv = ego_pose_inv,
        )
        # ================= Metadata ================= 
        results['metadata']['frame_token'] = infos.get('token', '')
        results['metadata']['scene_token'] = infos.get('scene_token', '')
        results['metadata']['source'] = self.NAME
        results['metadata']['data_path'] = self.data_path
        self.scene_set.add(infos.get('scene_token', ''))
        # results['visibilities'] = infos['visibilities']
        return results
    
    def _get_video(self, index: int) -> list[dict]:
        """
        Get all frames from the same scene starting from the given index.
        
        Args:
            index: Starting index in data_infos
            
        Returns:
            List of parsed frame information dictionaries
        """
        if index >= len(self.data_infos):
            return []
        
        video_results = []
        scene_token = self.data_infos[index]['scene_token']
        
        for i in range(index, len(self.data_infos)):
            if self.data_infos[i]['scene_token'] != scene_token:
                break
            video_results.append(self._parse_info(i))
        
        return video_results
    
    def __getitem__(self, index: int) -> Tuple[List[Dict[str, Any]], int]:
        """
        This function returning the current camera views, its annotation, caliberation and lidar.
        Camera view is presents as dict[str,str]
        Annotation is 2d and 3d annotation, and its 2d track id and 3d track id
        Caliberation may present as world2pix only
        return list of results and next index
        """
        video_results = self._get_video(index)
        return video_results, index + len(video_results)
    
    def __iter__(self):
        """
        Iterator that yields video sequences from the dataset.
        Each video contains all frames from the same scene.
        """
        index = 0
        while index < len(self.data_infos):
            video_results = self._get_video(index)
            if video_results:
                yield video_results
                index += len(video_results)
            else:
                index += 1
    
    def __call__(self, index: int, *args: Any, **kwds: Any) -> List[Dict[str, Any]]:
        """
        Call method to get video information at a specific index.
        
        Args:
            index: Index of the video scene
            *args: Additional positional arguments (unused)
            **kwds: Additional keyword arguments (unused)
            
        Returns:
            List of frame information dictionaries for the video
        """
        return self._get_video(index)
    
def invert_matrix_egopose_numpy(egopose: np.ndarray) -> np.ndarray:
    """
    Compute the inverse transformation of a 4x4 egopose numpy matrix.
    
    Args:
        egopose: 4x4 transformation matrix
        
    Returns:
        4x4 inverse transformation matrix
    """
    try:
        inverse_matrix = np.zeros((4, 4), dtype=np.float32)
        rotation = egopose[:3, :3]
        translation = egopose[:3, 3]
        inverse_matrix[:3, :3] = rotation.T
        inverse_matrix[:3, 3] = -np.dot(rotation.T, translation)
        inverse_matrix[3, 3] = 1.0
        return inverse_matrix
    except Exception as e:
        logger.error(f"Error inverting egopose matrix: {e}", exc_info=True)
        raise

def convert_egopose_to_matrix_numpy(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    """
    Convert rotation and translation to a 4x4 transformation matrix.
    
    Args:
        rotation: 3x3 rotation matrix
        translation: 3x1 translation vector
        
    Returns:
        4x4 transformation matrix
    """
    try:
        transformation_matrix = np.zeros((4, 4), dtype=np.float32)
        transformation_matrix[:3, :3] = rotation
        transformation_matrix[:3, 3] = translation
        transformation_matrix[3, 3] = 1.0
        return transformation_matrix
    except Exception as e:
        logger.error(f"Error converting egopose to matrix: {e}", exc_info=True)
        raise
