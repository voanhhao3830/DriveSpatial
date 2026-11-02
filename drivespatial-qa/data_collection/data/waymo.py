import os
import re
import pickle
import logging
import numpy as np
import matplotlib.pyplot as plt

from .base import BaseDataLoader
from typing import Dict, Any, List, Tuple
from nuscenes.eval.common.utils import Quaternion
from sources.data_collection.enum.data_sources import DataSource

logger = logging.getLogger(__name__)

class WaymoDataset(BaseDataLoader):
    NAME = DataSource.WAYMO.value
    CLASS_NAME = [ "VEHICLE", "PEDESTRIAN", "SIGN", "CYCLIST"]
    
    def __init__(self, data_path: str, ann_file: str):
        self.data_path = data_path
        self.ann_file = ann_file
        
        self.data_infos  = self._load_annotation(ann_file)
        
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
        data_infos = list(sorted(data['infos'], key=lambda e: float(e['timestamp'])))
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
            unique_scenes.add(info['frame_token']) # *.tfrecord
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
            results['views'][camera_name] = cam_data["data_path"]
            
        # ================= Lidar ================= 
        results['lidar'] = {
            "path": infos['lidar_path']
        }
        results['CAM_FRONT'] = {
            "path": infos['cams']['FRONT']['data_path']
        }
        
        # ================= Annotation ================= 
        # ================= Coordinate Transformation =================
        # Waymo coordinate system: X=right, Y=forward, Z=up, yaw=angle from Y-axis (forward)
        # Target (nuScenes) coordinate system: X=forward, Y=left, Z=up, yaw=angle from X-axis (forward)
        # 
        # Transformation needed:
        # 1. Rotate 90 degrees counterclockwise (CCW) around Z-axis: (x,y) -> (-y, x)
        #    This makes: X_new=forward (was Y), Y_new=left (was -X), Z_new=up (same)
        # 2. Adjust yaw: yaw_new = yaw_waymo + π/2 (since we rotated the frame)
        
        angle = np.pi / 2  # 90 degrees CCW rotation
        rotation_matrix = np.array([
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle),  np.cos(angle), 0],
            [0, 0, 1]
        ])
        
        # 3D Annotation
        gt_boxes = np.array(infos['gt_boxes'])
        if len(gt_boxes) > 0:
            # Transform bbox positions: (x, y, z) -> (-y, x, z)
            positions = gt_boxes[:, :3]  # x, y, z
            positions_transformed = (rotation_matrix @ positions.T).T
            gt_boxes[:, :3] = positions_transformed
            gt_boxes[:, 2] = gt_boxes[:, 2] + gt_boxes[:, 5] / 2
            
            # Transform bbox dimensions: swap length and width
            # Waymo: [length, width, height] where length is along Y (forward in Waymo)
            # After rotation: we need [length, width, height] where length is along new X (forward)
            # Since Y becomes X and X becomes -Y, we swap length and width
            gt_boxes[:, 3], gt_boxes[:, 4] = gt_boxes[:, 4], gt_boxes[:, 3]
            
            # Transform yaw angle  
            # Analysis from observations:
            # - yaw + π: forward correct, left reversed
            # - -yaw + π: left correct, forward reversed
            # Solution: Combine rotation adjustment with sign flip
            # yaw_target = -yaw_waymo + π/2
            # This accounts for: opposite rotation direction + coordinate frame rotation
            gt_boxes[:, 6] = -gt_boxes[:, 6] # angle = π/2
            
            # Normalize yaw to [-π, π]
            gt_boxes[:, 6] = np.arctan2(np.sin(gt_boxes[:, 6]), np.cos(gt_boxes[:, 6]))
        
        results['annotation']['bbox_3d'] = gt_boxes
        results['annotation']['label_3d'] = [WaymoDataset.CLASS_NAME[int(i) - 1] for i in infos['gt_names']]
        
        # 2D Annotation
        results['annotation']['bbox_2d'] = infos['bboxes2d']
        results['annotation']['label_2d'] = [
                                    [WaymoDataset.CLASS_NAME[int(i)] for i in list(sublist)]
                                    for sublist in infos['labels2d']
                                ]

        # 3D Track ID
        results['annotation']['track_3d'] = infos['gt_instance_tokens'] 
        results['annotation']['track_2d'] =  [
                                            [re.sub(r'_(FRONT|FRONT_LEFT|FRONT_RIGHT|SIDE_LEFT|SIDE_RIGHT)$', '', token) 
                                            for token in sublist]
                                            for sublist in infos['instance_token_2d']
                                        ] 
        # GT velocity - MUST transform to match bbox transformation
        gt_velocity = infos['gt_velocity']
        results['annotation']['velocity'] = None
        # if gt_velocity is not None and len(gt_velocity) > 0:
        #     gt_velocity = np.array(gt_velocity)
        #     # swap x and y
        #     # velocity_rotated = (rotation_matrix[:2, :2] @ gt_velocity.T).T
        #     results['annotation']['velocity'] = -gt_velocity
        # else:
        #     results['annotation']['velocity'] = gt_velocity
        
        # ================= Calibration ================= 
        e2g_rotation = Quaternion(infos['ego2global_rotation']).rotation_matrix
        e2g_translation = infos['ego2global_translation']
        l2e_rotation = Quaternion(infos['lidar2ego_rotation']).rotation_matrix
        l2e_translation = infos['lidar2ego_translation']
        e2g_matrix = convert_egopose_to_matrix_numpy(e2g_rotation, e2g_translation)
        l2e_matrix = convert_egopose_to_matrix_numpy(l2e_rotation, l2e_translation)
        ego_pose =  e2g_matrix @ l2e_matrix # lidar2global (in ORIGINAL coordinate frame)
        
        # ================= CRITICAL: Ego Pose Compensation for Bbox Rotation =================
        # Since we rotated the bounding boxes by 'angle' above (lines 101-117),
        # we need to apply the INVERSE rotation to ego_pose so that:
        # 1. Bboxes are in ROTATED coordinate frame
        # 2. ego_pose transforms from ROTATED frame -> global (instead of original -> global)
        # 
        # Math: If p_rotated = R @ p_original, then to transform p_rotated to global:
        #       p_global = ego_pose_orig @ p_original = ego_pose_orig @ R^(-1) @ p_rotated
        #       So: ego_pose_new = ego_pose_orig @ R^(-1)
        #
        # For rotation matrices: R^(-1) = R^T (transpose)
        ego_rotation_compensation = np.eye(4, dtype=np.float32)
        ego_rotation_compensation[:3, :3] = rotation_matrix.T  # Inverse of bbox rotation
        ego_pose = ego_pose @ ego_rotation_compensation  # Now transforms from rotated frame -> global

        ego_pose_inv = invert_matrix_egopose_numpy(ego_pose)
    
        lidar2img_rts = []
        intrinsics = []
        extrinsics = []
        img_timestamp = []
        
        # Prepare rotation compensation for camera extrinsics
        # Since bboxes are in rotated coordinate frame, camera extrinsics must also be rotated
        extrinsic_rotation_compensation = np.eye(4, dtype=np.float32)
        extrinsic_rotation_compensation[:3, :3] = rotation_matrix  # Same rotation as bboxes
        
        for cam_type, cam_info in infos['cams'].items():
            img_timestamp.append(float(cam_info['timestamp']) )
            
            # obtain lidar to image transformation matrix
            # Convert quaternion to rotation matrix
            cam2lidar_r_quat = Quaternion(cam_info['sensor2lidar_rotation'])
            cam2lidar_r = cam2lidar_r_quat.rotation_matrix
            cam2lidar_t = cam_info['sensor2lidar_translation']
            cam2lidar_rt = convert_egopose_to_matrix_numpy(cam2lidar_r, cam2lidar_t)
            lidar2cam_rt = invert_matrix_egopose_numpy(cam2lidar_rt)
            
            # Apply rotation to camera extrinsics
            # Camera extrinsic transforms from LiDAR to camera frame
            # Since LiDAR coordinates are now rotated, we need:
            # point_cam = lidar2cam_original @ rotation_matrix^(-1) @ point_lidar_rotated
            # Which means: lidar2cam_rotated = lidar2cam_original @ rotation_matrix^(-1)
            lidar2cam_rt_rotated = lidar2cam_rt @ invert_matrix_egopose_numpy(extrinsic_rotation_compensation)
            T_front_cam_to_ref = np.array([
                        [0.0, -1.0, 0.0, 0.0],
                        [0.0, 0.0, -1.0, 0.0],
                        [1.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 1.0]
                    ])
            lidar2cam_rt_rotated = T_front_cam_to_ref @ lidar2cam_rt_rotated 
            intrinsic = cam_info['cam_intrinsic']
            viewpad = np.eye(4)
            viewpad[:intrinsic.shape[0], :intrinsic.shape[1]] = intrinsic
            lidar2img_rt = (viewpad @ lidar2cam_rt_rotated)
            intrinsics.append(viewpad)
            extrinsics.append(lidar2cam_rt_rotated)
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
        results['metadata']['frame_token'] = infos.get('frame_token', '')
        results['metadata']['scene_token'] = infos.get('frame_token', '')
        results['metadata']['source'] = self.NAME
        results['metadata']['data_path'] = self.data_path
        results['metadata'].update(
                {k.replace(":",""):v for k,v in infos['metadata'].items()})
        # Optionally render BEV map (uncomment to enable)
        # WaymoDataset.render_topdown_map(
        #     lidar_path=infos['lidar_path'],
        #     ego_pose=None,
        #     bboxes_3d=gt_boxes,  # Use transformed boxes
        #     labels_3d=results['annotation']['label_3d'],
        #     out_path="test_waymo_bev.png"
        # )
        # import ipdb; ipdb.set_trace()
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
        scene_token = self.data_infos[index]['frame_token']
        
        for i in range(index, len(self.data_infos)):
            if self.data_infos[i]['frame_token'] != scene_token:
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

    @staticmethod
    def render_topdown_map(
        lidar_path: str,
        ego_pose: np.ndarray = None,
        bboxes_3d: np.ndarray = None,
        labels_3d: List[str] = None,
        range_m: float = 50.0,
        figsize: Tuple[int, int] = (12, 12),
        out_path: str = None
    ) -> None:
        """
        Render top-down BEV map with lidar points and optional 3D bounding boxes.

        Coordinate frame after _parse_info: same R(+pi/2) applied to both lidar
        and bboxes so ego always faces pi/2 (+Y, up in the plot).
        Box format: [x, y, z, width, length, h, yaw]
          - width  (index 3): lateral extent (original Waymo W, along old X/right -> new Y)
          - length (index 4): longitudinal extent (original Waymo L, along old Y/forward -> new -X)
          - yaw: angle from +X axis (CCW); heading arrow = [cos(yaw), sin(yaw)]
          - front of box is at +X in the local box frame

        Args:
            lidar_path: Path to lidar .bin file
            ego_pose: Unused (ego always faces pi/2; kept for API compatibility)
            bboxes_3d: [N, 7] boxes already in transformed frame from _parse_info
            labels_3d: Label names for each bbox
            range_m: Display range in metres around ego
            figsize: Figure size
            out_path: Save path; if None, shows interactively
        """
        # Load lidar points
        points = np.fromfile(lidar_path, dtype=np.float32)
        if points.shape[0] % 4 == 0:
            points = points.reshape(-1, 4)
        elif points.shape[0] % 6 == 0:
            points = points.reshape(-1, 6)
        else:
            raise ValueError(f"Unexpected lidar point stride: {points.shape[0]}")
        points_xyz = points[:, :3]

        # Apply the same R(+pi/2) as _parse_info so lidar matches the bbox frame.
        # Ego always faces pi/2 (+Y) after this transform — no ego-yaw rotation needed.
        rotation_matrix = np.array([
            [0, -1, 0],
            [1,  0, 0],
            [0,  0, 1]
        ], dtype=np.float32)
        points_xyz = (rotation_matrix @ points_xyz.T).T

        fig, ax = plt.subplots(figsize=figsize)
        ax.scatter(points_xyz[:, 0], points_xyz[:, 1],
                   s=0.5, c='red', alpha=0.5, zorder=4, label='LiDAR Points')

        if bboxes_3d is not None and len(bboxes_3d) > 0:
            class_colors = {
                'VEHICLE': 'blue',
                'PEDESTRIAN': 'black',
                'CYCLIST': 'brown',
                'SIGN': 'purple'
            }

            for i, bbox in enumerate(bboxes_3d):
                # [x, y, z, width(lateral), length(longitudinal), h, yaw]
                x, y, z, width, length, h, bbox_yaw = bbox[:7]

                margin = max(length, width)
                if abs(x) > range_m + margin or abs(y) > range_m + margin:
                    continue

                # Front of box is at +X in local frame (consistent with heading arrow)
                corners = np.array([
                    [ length/2,  width/2],   # front-right
                    [ length/2, -width/2],   # front-left
                    [-length/2, -width/2],   # rear-left
                    [-length/2,  width/2],   # rear-right
                    [ length/2,  width/2],   # close
                ])

                rot_mat = np.array([
                    [np.cos(bbox_yaw), -np.sin(bbox_yaw)],
                    [np.sin(bbox_yaw),  np.cos(bbox_yaw)]
                ])
                corners_world = (rot_mat @ corners.T).T + np.array([x, y])

                label = labels_3d[i] if labels_3d and i < len(labels_3d) else 'Unknown'
                color = class_colors.get(label, 'gray')

                ax.plot(corners_world[:, 0], corners_world[:, 1],
                        color=color, linewidth=2, zorder=6, alpha=0.8)

                # Heading arrow points in [cos(yaw), sin(yaw)] direction
                ax.arrow(x, y,
                         length / 2 * np.cos(bbox_yaw),
                         length / 2 * np.sin(bbox_yaw),
                         head_width=0.5, head_length=0.3,
                         fc=color, ec=color, zorder=7, alpha=0.6, linewidth=1.5)

        # Ego always faces pi/2 (+Y = up in the plot)
        arrow_length = 3.0
        ax.arrow(0, 0, 0, arrow_length,
                 head_width=1.5, head_length=1.0,
                 fc='green', ec='green', linewidth=2, zorder=5, label='Ego Vehicle')

        ax.set_xlim(-range_m, range_m)
        ax.set_ylim(-range_m, range_m)
        ax.legend(fontsize=12, markerscale=2, framealpha=0.9)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        plt.tight_layout()

        if out_path is not None:
            plt.savefig(out_path, dpi=150, bbox_inches='tight')
            logger.info(f"BEV map saved to {out_path}")
            plt.close()
        else:
            plt.show()
                

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