import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

from uuid import uuid4
from .base import BaseDataLoader
from typing import Dict, Any, List, Tuple
from sources.data_collection.enum.data_sources import DataSource
from scipy.spatial.transform import Rotation

class ONCEDataset(BaseDataLoader):
    NAME = DataSource.ONCE.value
    CLASS_NAME = ['Bus', 'Cyclist', 'Pedestrian', 'Car', 'Truck']
    
    def __init__(self, data_path: str, ann_file: str):
        self.data_path = data_path
        self.ann_file = ann_file
        
        self.data_infos = self._load_annotation(ann_file)
        
    def _load_annotation(self, annotation_path: str):
        with open(annotation_path, 'rb') as f:
            data = pickle.load(f)
        data_infos = list(sorted(data, key=lambda e: e['timestamp']))
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
        unique_scenes = set()
        for info in self.data_infos:
            unique_scenes.add(info['sequence_id'])
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
        if 'annos' not in infos:
            return None
        # ================= IMAGE ================= 
        # cams_data = infos['cams']
        cams_keys = ['cam01', 'cam03', 'cam05', 'cam06', 'cam07', 'cam08', 'cam09']
        for camera_name in cams_keys:
            cam_data = infos[camera_name]
            results['views'][camera_name] = cam_data # remove relative path
        
        # ================= Lidar ================= 
        results['lidar'] = {
            "path": infos['lidar']
        }
        results['CAM_FRONT'] = {
            "path": infos['cam01']
        }
        # ================= Annotation ================= 
        # Create mapping from 3D box index to track ID
        num_3d_boxes = len(infos['annos']['boxes_3d'])
        track_3d_ids = [str(uuid4()) for _ in range(num_3d_boxes)]

        # Process 2D boxes
        bbox_2d_list = []
        label_2d_list = []
        track_2d_list = []

        for cam_name in cams_keys:
            # Filter out invalid boxes for this camera view
            valid_boxes = []
            valid_labels = []
            valid_tracks = []
            boxes_array = infos['annos']['boxes_2d'][cam_name]
            for box_idx, box in enumerate(boxes_array):
                # Skip invalid boxes (all -1 values)
                if not (box == -1).all():
                    valid_boxes.append(box)
                    
                    # Map to the same label as the corresponding 3D box
                    if box_idx < len(infos['annos']['name']):
                        valid_labels.append(infos['annos']['name'][box_idx])
                        valid_tracks.append(track_3d_ids[box_idx])
                    else:
                        valid_labels.append('unknown')
                        valid_tracks.append(str(uuid4()))
            
            # Append arrays for this camera view (even if empty)
            bbox_2d_list.append(np.array(valid_boxes) if valid_boxes else np.array([]).reshape(0, 4))
            label_2d_list.append(np.array(valid_labels) if valid_labels else np.array([]))
            track_2d_list.append(np.array(valid_tracks).tolist() if valid_tracks else np.array([]).tolist())

        # ================= Coordinate Transformation ================= 
        # Rotate 180 degrees around z-axis to convert: x+ left,y+ rear -> x+ right,y+ front
        flip_transform = np.array([
            [-1, 0, 0, 0],
            [0, -1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        
        # 3D Annotation
        boxes_3d = np.array(infos['annos']['boxes_3d'])
        if len(boxes_3d) > 0:
            boxes_3d[:, 0] = -boxes_3d[:, 0]
            boxes_3d[:, 1] = -boxes_3d[:, 1]
            boxes_3d[:, 6] = boxes_3d[:, 6] + np.pi
        
        results['annotation']['bbox_3d'] = boxes_3d
        results['annotation']['label_3d'] = infos['annos']['name']

        # 2D Annotation
        results['annotation']['bbox_2d'] = bbox_2d_list if bbox_2d_list else []
        results['annotation']['label_2d'] = label_2d_list if label_2d_list else []
        # Track IDs
        results['annotation']['track_3d'] = track_3d_ids
        results['annotation']['track_2d'] = track_2d_list if track_2d_list else []

        # GT velocity
        results['annotation']['velocity'] = None 
        
        # ================= Calibration ================= 
        pose_array = infos['pose']
        quat = pose_array[:4]
        trans = pose_array[4:]
        
        rotation_matrix = Rotation.from_quat(quat).as_matrix()
        ego_pose = np.eye(4)
        ego_pose[:3, :3] = rotation_matrix
        ego_pose[:3, 3] = trans
        
        ego_pose = flip_transform @ ego_pose
        ego_pose_inv = np.linalg.inv(ego_pose)
    
        lidar2img_rts = []
        intrinsics = []
        extrinsics = []
        
        for cam_name in cams_keys:
            cam_calib = infos['calib'][cam_name]
            
            cam_to_velo = cam_calib['cam_to_velo']
            cam_intrinsic = cam_calib['cam_intrinsic']
            
            lidar2cam_rt = np.linalg.inv(cam_to_velo)
            lidar2cam_rt = lidar2cam_rt @ np.linalg.inv(flip_transform)
            
            viewpad = np.eye(4)
            viewpad[:cam_intrinsic.shape[0], :cam_intrinsic.shape[1]] = cam_intrinsic
            lidar2img_rt = viewpad @ lidar2cam_rt
            
            intrinsics.append(viewpad)
            extrinsics.append(lidar2cam_rt)
            lidar2img_rts.append(lidar2img_rt)
        
        ego_rotation_transform = np.eye(4)
        angle = -90 * np.pi / 180  # 90 degrees to radians
        rotation_matrix = np.array([
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle),  np.cos(angle), 0],
            [0, 0, 1]
        ])
        ego_rotation_transform[:3, :3] = rotation_matrix
        ego_pose = ego_rotation_transform  @ ego_pose 
        
        results['calibration'] = dict(
            lidar2img=lidar2img_rts,
            intrinsics=intrinsics,
            extrinsics=extrinsics,
            ego_pose=ego_pose,
            ego_pose_inv=ego_pose_inv,
        )
        # ================= Metadata ================= 
        results['metadata']['frame_token'] = infos.get('frame_id', '')
        results['metadata']['scene_token'] = infos.get('sequence_id', '')
        results['metadata']['source'] = self.NAME
        results['metadata']['data_path'] = self.data_path
        results['metadata'].update(infos['meta_info'])
        # ONCEDataset.render_topdown_map(
        #     lidar_path=infos['lidar'],
        #     ego_pose=None,
        #     bbox_3d=results['annotation']['bbox_3d'],
        #     label_3d=results['annotation']['label_3d'],
        #     out_path="test.png"
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
        scene_token = self.data_infos[index]['sequence_id']
        
        for i in range(index, len(self.data_infos)):
            if self.data_infos[i]['sequence_id'] != scene_token:
                break
            result = self._parse_info(i)
            if result is not None:
                video_results.append(result)
        
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
        bbox_3d: np.ndarray = None,
        label_3d: List[str] = None,
        range_m: float = 50.0,
        figsize: Tuple[int, int] = (12, 12),
        out_path: str = None
    ) -> None:
        """
        Render top-down BEV map with lidar points and 3D bounding boxes.

        Coordinate frame after _parse_info: R(pi) (negate x,y) applied to both
        lidar and bboxes so ego always faces pi/2 (+Y, up in the plot).
        Box format: [x, y, z, length, width, h, yaw]
          - length (index 3): longitudinal extent (front-rear)
          - width  (index 4): lateral extent (side-to-side)
          - yaw: angle from +X axis (CCW); heading arrow = [cos(yaw), sin(yaw)]
          - front of box is at +X in the local box frame

        Args:
            lidar_path: Path to lidar .bin file
            ego_pose: Unused (ego always faces pi/2; kept for API compatibility)
            bbox_3d: [N, 7] boxes already in transformed frame from _parse_info
            label_3d: Label names for each bbox
            range_m: Display range in metres around ego
            figsize: Figure size
            out_path: Save path; if None, shows interactively
        """
        points = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 4)
        points_xyz = points[:, :3]

        # Apply the same R(pi) as _parse_info so lidar matches the bbox frame.
        # Ego always faces pi/2 (+Y) after this transform — no ego-yaw rotation needed.
        rotation_matrix = np.array([
            [-1,  0, 0],
            [ 0, -1, 0],
            [ 0,  0, 1]
        ], dtype=np.float32)
        points_xyz = (rotation_matrix @ points_xyz.T).T

        fig, ax = plt.subplots(figsize=figsize)
        ax.scatter(points_xyz[:, 0], points_xyz[:, 1],
                   s=0.5, c='red', alpha=0.5, zorder=4, label='LiDAR Points')

        if bbox_3d is not None and len(bbox_3d) > 0:
            class_colors = {
                'Car': 'blue',
                'Truck': 'cyan',
                'Bus': 'purple',
                'Pedestrian': 'yellow',
                'Cyclist': 'orange'
            }

            for idx, box in enumerate(bbox_3d):
                # [x, y, z, length(longitudinal), width(lateral), h, yaw]
                x, y, z, length, width, height, box_yaw = box

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

                rot_matrix = np.array([
                    [np.cos(box_yaw), -np.sin(box_yaw)],
                    [np.sin(box_yaw),  np.cos(box_yaw)]
                ])
                corners_world = (rot_matrix @ corners.T).T + np.array([x, y])

                label = label_3d[idx] if label_3d is not None and idx < len(label_3d) else 'Unknown'
                color = class_colors.get(label, 'gray')

                ax.plot(corners_world[:, 0], corners_world[:, 1],
                        color=color, linewidth=2, zorder=6, alpha=0.8)

                # Heading arrow points in [cos(yaw), sin(yaw)] direction
                ax.arrow(x, y,
                         length / 2 * np.cos(box_yaw),
                         length / 2 * np.sin(box_yaw),
                         head_width=0.5, head_length=0.3,
                         fc=color, ec=color, zorder=7, alpha=0.6, linewidth=1.5)

                ax.text(x, y + width / 2 + 1, label,
                        fontsize=8, color=color, weight='bold',
                        ha='center', va='bottom', zorder=8,
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                                  alpha=0.7, edgecolor=color, linewidth=1.5))

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
            plt.close()

if __name__ == "__main__":
    dataset = ONCEDataset()