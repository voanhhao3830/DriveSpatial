import os
import pickle
import numpy as np
from pathlib import Path
from typing import Tuple
import matplotlib.pyplot as plt

from .base import BaseDataLoader
from sources.data_collection.enum.data_sources import DataSource
from av2.datasets.sensor.av2_sensor_dataloader import AV2SensorDataLoader
from av2.map.map_api import ArgoverseStaticMap
from av2.structures.sweep import Sweep
from av2.utils.typing import NDArrayFloat

class Argoverse2Dataset(BaseDataLoader):
    NAME = DataSource.ARGOVERSE2.value
    DATA_PATH = None
    SPLIT = None
    CLASS_NAME = ['ARTICULATED_BUS', 'BICYCLE', 'BICYCLIST', 'BOLLARD', 'BOX_TRUCK', 'BUS',
               'CONSTRUCTION_BARREL', 'CONSTRUCTION_CONE', 'DOG', 'LARGE_VEHICLE',
               'MESSAGE_BOARD_TRAILER', 'MOBILE_PEDESTRIAN_CROSSING_SIGN', 'MOTORCYCLE',
               'MOTORCYCLIST', 'PEDESTRIAN', 'REGULAR_VEHICLE', 'SCHOOL_BUS', 'SIGN',
               'STOP_SIGN', 'STROLLER', 'TRUCK', 'TRUCK_CAB', 'VEHICULAR_TRAILER',
               'WHEELCHAIR', 'WHEELED_DEVICE','WHEELED_RIDER']
    
    def __init__(self, data_path: str, ann_file: str):
        self.data_path = data_path
        Argoverse2Dataset.DATA_PATH = data_path
        self.ann_file = ann_file
        
        self.data_infos, self.split = self._load_annotation(ann_file)
        Argoverse2Dataset.SPLIT = self.split
        
    def _load_annotation(self, annotation_path: str):
        with open(annotation_path, 'rb') as f:
            data = pickle.load(f)
        return data["infos"], data["split"]
    
    def load_metadata(self, data_path: str):
        pass
    
    def __len__(self):
        return len(set([item['scene_id'] for item in self.data_infos])) if self.data_infos is not None else 0
    
    def _get_video(self, index: int):
        if index >= len(self.data_infos):
            return []
        
        video_results = []
        prev_scene_token = self.data_infos[index]['scene_id']
        for i in range(index, len(self.data_infos)):
            if self.data_infos[i]['scene_id'] != prev_scene_token:
                break
            video_results.append(self._parse_info(i))
            
        return video_results
    
    def _parse_info(self, index: int):
        results = {
            "views": {},
            "lidar": "",
            "annotation": {},
            "calibration": {},
            "metadata": {}
        } # include 3 of expected result
        infos = self.data_infos[index]
        
        # ================= IMAGE ================= 
        cams_data = infos['cam_infos']
        for camera_name, cam_data in cams_data.items():
            results['views'][camera_name] = os.path.join(self.data_path, cam_data["fpath"]) # remove relative path
        
        # ================= Lidar ================= 
        lidar_time = infos['lidar_timestamp_ns']
        results['lidar'] = {
            "path":  os.path.join(self.data_path, self.split, infos['scene_id'], "sensors/lidar", f"{int(lidar_time)}.feather")
        }
        results['CAM_FRONT'] = {
            "path": os.path.join(self.data_path, cams_data['ring_front_center']['fpath'])
        }
        
        # ================= Annotation ================= 
        # 3D Annotation
        angle = -(-90 * np.pi / 180)  # -90 degrees to radians
        rotation_matrix = np.array([
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle),  np.cos(angle), 0],
            [0, 0, 1]
        ])
        gt_boxes = np.array(infos['gt3d_infos']['gt_boxes'])
        if len(gt_boxes) > 0:
            # Transform bbox positions
            positions = gt_boxes[:, :3]  # x, y, z
            positions_transformed = (rotation_matrix @ positions.T).T
            gt_boxes[:, :3] = positions_transformed
            # Adjust yaw angle: rotate by -135 degrees (-3π/4)
            gt_boxes[:, 6] = gt_boxes[:, 6] + angle
            # Normalize yaw to [-π, π]
            gt_boxes[:, 6] = np.arctan2(np.sin(gt_boxes[:, 6]), np.cos(gt_boxes[:, 6]))
            
        results['annotation']['bbox_3d'] = gt_boxes
        results['annotation']['label_3d'] = infos['gt3d_infos']['gt_names']
        
        # 2D Annotation
        results['annotation']['bbox_2d'] = infos['gt2d_infos']['gt_2dbboxes']
        results['annotation']['label_2d'] = infos['gt2d_infos']['gt_2dlabels']

        # 3D Track ID
        results['annotation']['track_3d'] = infos['gt3d_infos']['gt_instance_tokens'] 
        results['annotation']['track_2d'] = infos['gt2d_infos']['instance_token_2d'] 
        
        # GT velocity
        results['annotation']['velocity'] = None
        
        # ================= Calibration ================= 
        lidar2img_rts = []
        intrinsics = []
        extrinsics = []
        img_timestamp = []
        city_SE3_ego_lidar_t = infos['city_SE3_ego_lidar_t']
        transform_matrix = np.eye(4)
        transform_matrix[:3, :3] = city_SE3_ego_lidar_t.rotation
        transform_matrix[:3, 3] = city_SE3_ego_lidar_t.translation
        ego_pose = transform_matrix
        
        # ================= CRITICAL: Ego Pose Compensation for Bbox Rotation =================
        ego_rotation_compensation = np.eye(4, dtype=np.float32)
        ego_rotation_compensation[:3, :3] = rotation_matrix.T  # Inverse of bbox rotation
        ego_pose = ego_pose @ ego_rotation_compensation  # Now transforms from rotated frame -> global
        
        ego_pose_inv = invert_matrix_egopose_numpy(ego_pose)
        
        # Prepare rotation compensation for camera extrinsics
        extrinsic_rotation_compensation = np.eye(4, dtype=np.float32)
        extrinsic_rotation_compensation[:3, :3] = rotation_matrix  # Same rotation as bboxes
        
        for cam_type, cam_info in infos['cam_infos'].items():
            if cam_info is None:
                return None
            img_timestamp.append(cam_info['cam_timestamp_ns']/ 1e9)
            # obtain lidar to image transformation matrix
            city_SE3_ego_cam_t = cam_info['city_SE3_ego_cam_t']
            ego_SE3_cam = cam_info['ego_SE3_cam']
            ego_cam_t_SE3_ego_lidar_t = city_SE3_ego_cam_t.inverse().compose(city_SE3_ego_lidar_t) #ego2glo_lidar -> glo2ego_cam
            cam_SE3_ego_cam_t = ego_SE3_cam.inverse().compose(ego_cam_t_SE3_ego_lidar_t) #ego -> cam
            transform_matrix = np.eye(4)
            transform_matrix[:3, :3] = cam_SE3_ego_cam_t.rotation
            transform_matrix[:3, 3] = cam_SE3_ego_cam_t.translation
            
            # Apply rotation to camera extrinsics
            # Since LiDAR coordinates are now rotated, we need:
            # lidar2cam_rotated = lidar2cam_original @ rotation_matrix^(-1)
            transform_matrix_rotated = transform_matrix @ invert_matrix_egopose_numpy(extrinsic_rotation_compensation)

            intrinsic = cam_info['intrinsics']
            viewpad = np.eye(4)
            viewpad[:intrinsic.shape[0], :intrinsic.shape[1]] = intrinsic
            lidar2img_rt = (viewpad @ transform_matrix_rotated)
            intrinsics.append(viewpad)
            extrinsics.append(transform_matrix_rotated)
            lidar2img_rts.append(lidar2img_rt)
        
        results['calibration'] = dict(
            img_timestamp=img_timestamp,
            lidar2img=lidar2img_rts,
            intrinsics=intrinsics,
            extrinsics=extrinsics,
            ego_pose=ego_pose,
            ego_pose_inv=ego_pose_inv,
        )
        # ================= Metadata ================= 
        # Argoverse2Dataset.render_topdown_map_with_hd(
        #     log_id=infos['scene_id'],
        #     lidar_timestamp_ns=infos['lidar_timestamp_ns'],
        #     out_path="test.png",
        #     bboxes_3d=results['annotation']['bbox_3d'],
        #     labels_3d=results['annotation']['label_3d']
        # )
        # import ipdb; ipdb.set_trace()
        results['metadata']['scene_token'] = infos['scene_id']
        results['metadata']['frame_token'] = infos['lidar_timestamp_ns']
        results['metadata']['source'] = self.NAME
        results['metadata']['data_path'] = self.data_path
        results['metadata']['split'] = self.split
        return results
    
    def __getitem__(self, index):
        """
        This function returning the current camera views, its annotation, caliberation and lidar.
        Camera view is presents as dict[str,str]
        Annotation is 2d and 3d annotation, and its 2d track id and 3d track id
        Caliberation may present as world2pix only
        """
        video_results = self._get_video(index)
        return video_results, index + len(video_results) 
    def __iter__(self):
        """
        Iterator that yields video sequences from the dataset.
        Each video contains all frames from the same log.
        """
        index = 0
        while index < len(self.data_infos):
            video_results = self._get_video(index)
            if video_results:
                yield video_results
                index += len(video_results)
            else:
                index += 1
    @staticmethod
    def render_topdown_map_with_hd(
        log_id: str,
        lidar_timestamp_ns: int,
        range_m: float = 50.0,
        figsize: Tuple[int, int] = (12, 12),
        out_path: str = None,
        bboxes_3d: np.ndarray = None,
        labels_3d: list = None,
    ) -> None:
        """
        Render top-down map with HD map, lidar points, and optional 3D bounding boxes.

        The scene is rotated so ego always faces pi/2 (+Y = up in the plot).
        Bboxes from _parse_info are already in this ego-centric R(+pi/2) frame and
        are drawn directly without additional transformation.

        Box format expected: [x, y, z, length, width, h, yaw]
          - length (index 3): longitudinal extent (front-rear)
          - width  (index 4): lateral extent (side-to-side)
          - yaw: angle from +X axis (CCW); front of box is at +X in local frame

        Args:
            log_id: Scene token / log ID
            lidar_timestamp_ns: LiDAR timestamp in nanoseconds
            range_m: Display range in metres around ego
            figsize: Figure size for matplotlib
            out_path: Save path; if None, shows interactively
            bboxes_3d: Optional [N, 7] boxes already in transformed frame from _parse_info
            labels_3d: Optional label names for each bbox
        """
        if Argoverse2Dataset.DATA_PATH is None or Argoverse2Dataset.SPLIT is None:
            raise ValueError(
                "Argoverse2Dataset.DATA_PATH and SPLIT must be set before render_topdown_map_with_hd "
                "(e.g. by BEVRenderer from dataset_configs or metadata)"
            )
        data_dir = Path(Argoverse2Dataset.DATA_PATH) / Argoverse2Dataset.SPLIT
        loader = AV2SensorDataLoader(data_dir=data_dir, labels_dir=data_dir)
        
        log_map_dirpath = loader.get_log_map_dirpath(log_id)
        avm = ArgoverseStaticMap.from_map_dir(log_map_dirpath, build_raster=False)
        
        lidar_fpath = loader.get_lidar_fpath(log_id, lidar_timestamp_ns)
        sweep = Sweep.from_feather(lidar_fpath)
        
        city_SE3_ego = loader.get_city_SE3_ego(log_id, lidar_timestamp_ns)

        points_ego = sweep.xyz
        points_city = city_SE3_ego.transform_point_cloud(points_ego)

        ego_center_city: NDArrayFloat = city_SE3_ego.translation[:2]

        # Compute rotation that makes ego face pi/2 (+Y = up in the plot).
        # All city-frame elements are rotated around ego_center by (pi/2 - yaw).
        ego_rotation = city_SE3_ego.rotation
        yaw = np.arctan2(ego_rotation[1, 0], ego_rotation[0, 0])
        rot_angle = np.pi / 2 - yaw
        cos_r, sin_r = np.cos(rot_angle), np.sin(rot_angle)
        rot_2d = np.array([[cos_r, -sin_r], [sin_r, cos_r]])

        def _to_ego_frame(pts_2d: np.ndarray) -> np.ndarray:
            """Translate to ego-centric and rotate so ego faces +Y."""
            return (rot_2d @ (pts_2d - ego_center_city).T).T

        fig, ax = plt.subplots(figsize=figsize)

        road_legend_added = False
        for ls in avm.get_scenario_lane_segments():
            left_pts  = _to_ego_frame(ls.left_lane_boundary.xyz[:, :2])
            right_pts = _to_ego_frame(ls.right_lane_boundary.xyz[:, :2])

            polygon_pts = np.vstack([left_pts, right_pts[::-1]])
            if not road_legend_added:
                ax.fill(polygon_pts[:, 0], polygon_pts[:, 1],
                        color='grey', alpha=0.3, zorder=0, label='Road Lanes')
                road_legend_added = True
            else:
                ax.fill(polygon_pts[:, 0], polygon_pts[:, 1],
                        color='grey', alpha=0.3, zorder=0)

            ax.plot(left_pts[:, 0],  left_pts[:, 1],  'black', linewidth=0.5, alpha=0.6, zorder=2)
            ax.plot(right_pts[:, 0], right_pts[:, 1], 'black', linewidth=0.5, alpha=0.6, zorder=2)

        ped_crossing_legend_added = False
        for pc in avm.get_scenario_ped_crossings():
            poly = _to_ego_frame(pc.polygon[:, :2])
            if not ped_crossing_legend_added:
                ax.fill(poly[:, 0], poly[:, 1],
                        color='cyan', alpha=0.4, zorder=1, label='Pedestrian Crossing')
                ped_crossing_legend_added = True
            else:
                ax.fill(poly[:, 0], poly[:, 1], color='cyan', alpha=0.4, zorder=1)
            ax.plot(poly[:, 0], poly[:, 1], 'blue', linewidth=1.0, alpha=0.7, zorder=3)

        pts_ego_frame = _to_ego_frame(points_city[:, :2])
        ax.scatter(pts_ego_frame[:, 0], pts_ego_frame[:, 1],
                   s=0.5, c='red', alpha=0.5, zorder=4, label='LiDAR Points')

        # Draw 3D bounding boxes if provided.
        # Bboxes are already in the ego-centric R(+pi/2) frame from _parse_info,
        # which matches the frame used here — no additional transform needed.
        if bboxes_3d is not None and len(bboxes_3d) > 0:
            class_colors = {
                'ARTICULATED_BUS': 'purple', 'BICYCLE': 'lime', 'BICYCLIST': 'lime',
                'BOX_TRUCK': 'cyan',  'BUS': 'purple', 'LARGE_VEHICLE': 'cyan',
                'MOTORCYCLE': 'orange', 'MOTORCYCLIST': 'orange',
                'PEDESTRIAN': 'yellow', 'REGULAR_VEHICLE': 'blue',
                'SCHOOL_BUS': 'purple', 'TRUCK': 'cyan', 'TRUCK_CAB': 'cyan',
            }

            for i, bbox in enumerate(bboxes_3d):
                # [x, y, z, length(longitudinal), width(lateral), h, yaw]
                x, y, z, length, width, h, bbox_yaw = bbox[:7]

                margin = max(length, width)
                if abs(x) > range_m + margin or abs(y) > range_m + margin:
                    continue

                # Front of box at +X in local frame (consistent with heading arrow)
                corners = np.array([
                    [ length / 2,  width / 2],   # front-right
                    [ length / 2, -width / 2],   # front-left
                    [-length / 2, -width / 2],   # rear-left
                    [-length / 2,  width / 2],   # rear-right
                    [ length / 2,  width / 2],   # close
                ])
                rot_mat = np.array([
                    [np.cos(bbox_yaw), -np.sin(bbox_yaw)],
                    [np.sin(bbox_yaw),  np.cos(bbox_yaw)],
                ])
                corners_world = (rot_mat @ corners.T).T + np.array([x, y])

                label = labels_3d[i] if labels_3d and i < len(labels_3d) else 'Unknown'
                color = class_colors.get(label, 'gray')

                ax.plot(corners_world[:, 0], corners_world[:, 1],
                        color=color, linewidth=2, zorder=6, alpha=0.9)

                # Heading arrow points in [cos(yaw), sin(yaw)] direction
                ax.arrow(x, y,
                         length / 2 * np.cos(bbox_yaw),
                         length / 2 * np.sin(bbox_yaw),
                         head_width=0.5, head_length=0.3,
                         fc=color, ec=color, zorder=7, alpha=0.7, linewidth=1.5)

        # Ego is at origin, always faces pi/2 (+Y = up in the plot)
        arrow_length = 5.0
        ax.scatter(0, 0, s=80, c='green', marker='o', zorder=5)
        ax.arrow(0, 0, 0, arrow_length,
                 head_width=2.0, head_length=1.5, fc='green', ec='darkgreen',
                 linewidth=2, zorder=6, label='Ego Vehicle')

        ax.set_xlim(-range_m, range_m)
        ax.set_ylim(-range_m, range_m)
        ax.legend(fontsize=12, markerscale=2, framealpha=0.9)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        plt.tight_layout()

        if out_path is not None:
            plt.savefig(out_path)

def invert_matrix_egopose_numpy(egopose):
    """ Compute the inverse transformation of a 4x4 egopose numpy matrix."""
    inverse_matrix = np.zeros((4, 4), dtype=np.float32)
    rotation = egopose[:3, :3]
    translation = egopose[:3, 3]
    inverse_matrix[:3, :3] = rotation.T
    inverse_matrix[:3, 3] = -np.dot(rotation.T, translation)
    inverse_matrix[3, 3] = 1.0
    return inverse_matrix

