import cv2
import logging
import os
import math
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import Dict, Any, List, Optional, Tuple
from sources.data_collection.utils.constant import STATIC_LABELS
from sources.data_collection.enum import (
    Relation, 
    IntrinsicInteraction, 
    ExtrinsicInteraction
)
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

logger = logging.getLogger(__name__)

class RelationalExtractor:
    @staticmethod
    def extract_relations(video_data: List[Dict[str, Any]], alpha: float = 1, beta: float = 1) -> List[Dict[str, Any]]:
        """
        x+: front
        x-: back    
        y+: right
        y-: left
        z+: up
        z-: down
        
        bbox_3d format: [x, y, z, length, width, height, yaw]
        
        BEV:
        ^
        | y+ (right)
        |
        |   /
        |  /
        | /
        |/ yaw
        +--------------> x+ (front)
        
        """
        results = []
        
        for frame in video_data:
            bbox_3d = frame['annotation']['bbox_3d']
            label_3d = frame['annotation']['label_3d']
            track_3d = frame['annotation']['track_3d']
            
            # Calculate ego yaw from ego_pose rotation matrix
            ego_pose = frame['calibration']['ego_pose']
            ego_yaw = float(np.arctan2(ego_pose[1, 0], ego_pose[0, 0]))
            # Add ego vehicle with ID 00000 at origin
            bbox_3d = [[0.0, 0.0, 0.0, 4.5, 2.0, 1.8, math.pi / 2]] + list(bbox_3d)
            label_3d = ['ego'] + list(label_3d)
            track_3d = ['00000'] + list(track_3d)
            
            frame_relations = []
            
            for i in range(len(bbox_3d)):
                for j in range(len(bbox_3d)):
                    if i == j:
                        continue
                    
                    center_i = np.asarray(bbox_3d[i][:3], dtype=float)
                    center_j = np.asarray(bbox_3d[j][:3], dtype=float)
                    size_i   = np.asarray(bbox_3d[i][3:6], dtype=float)
                    size_j   = np.asarray(bbox_3d[j][3:6], dtype=float)
                    yaw_i    = float(bbox_3d[i][6])

                    # displacement from source to target in LiDAR coords
                    d = center_j - center_i
                    dx, dy, dz = float(d[0]), float(d[1]), float(d[2])

                    # --- Build the "line" direction from source point using BEV yaw ---
                    # Forward (heading) unit vector in XY
                    f = np.array([np.cos(yaw_i), np.sin(yaw_i), 0.0], dtype=float)
                    # Right unit vector in XY (perpendicular to forward)
                    r = np.array([np.sin(yaw_i), -np.cos(yaw_i), 0.0], dtype=float)

                    # Project displacement onto forward/right axes (signed projection value - how far along the forward/right axis the target is from the source)
                    s = float(np.dot(d, f))  # forward(+)/behind(-)
                    u = float(np.dot(d, r))  # right(+)/left(-)

                    # Optional: closest point on the infinite line (for debugging/visualization)
                    # P_line = center_i + s * f

                    # Thresholds (same spirit as your original)
                    avg_size = (np.mean(size_i) + np.mean(size_j)) / 2.0
                    threshold_xy = alpha * avg_size
                    threshold_z  = beta * avg_size

                    relations = []

                    # --- Front/back relative to source forward vector ---
                    # If target is mostly along forward axis
                    if s > threshold_xy and abs(u) < threshold_xy:
                        relations.append(Relation.AHEAD_OF)
                    elif s < -threshold_xy and abs(u) < threshold_xy:
                        relations.append(Relation.BEHIND)

                    # --- Left/right relative to source forward vector ---
                    if u > threshold_xy and abs(s) < threshold_xy:
                        relations.append(Relation.RIGHT_OF)
                    elif u < -threshold_xy and abs(s) < threshold_xy:
                        relations.append(Relation.LEFT_OF)

                    # --- Diagonals (combine) ---
                    if s > threshold_xy and u > threshold_xy:
                        relations.append(Relation.AHEAD_RIGHT_OF)
                    elif s > threshold_xy and u < -threshold_xy:
                        relations.append(Relation.AHEAD_LEFT_OF)
                    elif s < -threshold_xy and u > threshold_xy:
                        relations.append(Relation.REAR_RIGHT_OF)
                    elif s < -threshold_xy and u < -threshold_xy:
                        relations.append(Relation.REAR_LEFT_OF)

                    # --- Vertical ---
                    if dz > threshold_z:
                        relations.append(Relation.ABOVE)
                    elif dz < -threshold_z:
                        relations.append(Relation.BELOW)

                    if relations:
                        frame_relations.append({
                            'source_object': {'track_id': track_3d[i], 'label': str(label_3d[i])},
                            'target_object': {'track_id': track_3d[j], 'label': str(label_3d[j])},
                            'relations': relations,
                            # helpful debug numbers (optional)
                            'rel_coords_in_source_frame': {'forward': s, 'right': u, 'up': dz},
                            'source_forward_vec_xy': {'fx': float(f[0]), 'fy': float(f[1])}
                        })
            # RelationalExtractor.visualize_relations(
            #     frame,
            #     frame_relations, 
            #     view_idx="CAMERA_LEFT_FRONT",
            #     target_track="00000",
            #     save_path='debug_relations.png', 
            # )
            # if "motorcycle" in label_3d.tolist() or "bicycle" in label_3d.tolist():
            #     for view_name in frame['views'].keys():
            #         RelationalExtractor.visualize_relations(frame,frame_relations, view_idx=view_name, save_path=f'debug_relations_{view_name}.png',)
            #     import ipdb; ipdb.set_trace()
            results.append({
                **frame,
                'spatial_relations': frame_relations
            })
        
        return results
    
    @staticmethod
    def visualize_relations(
        frame: Dict[str, Any], 
        spatial_relations: List[Dict[str, Any]], 
        save_path: str = None,
        view_idx: str = None,
        target_track: str = None,
    ):
        view_names = list(frame['views'].keys())
        if view_idx is None:
            view_idx = view_names[-1]
        if view_idx not in view_names:
            logger.warning(f"View {view_idx} not found in frame. Available: {view_names}")
            return
        idx = view_names.index(view_idx)
        
        bbox_2d = frame['annotation']['bbox_2d'][idx]
        track_2d = frame['annotation']['track_2d'][idx]
        image = plt.imread(frame['views'][view_idx])
        
        # Add ego vehicle at image center
        img_h, img_w = image.shape[:2]
        ego_size = 60  # pixels
        ego_bbox = [img_w/2 - ego_size/2, img_h - ego_size - 10, img_w/2 + ego_size/2, img_h - 10]
        bbox_2d = [ego_bbox] + list(bbox_2d)
        track_2d = ['00000'] + list(track_2d)
        
        track_to_bbox = {track_2d[i]: bbox_2d[i] for i in range(len(track_2d))}
        
        fig, ax = plt.subplots(1, figsize=(16, 12))
        ax.imshow(image)
        
        bbox_centers = {}
        for i, (track_id, bbox) in enumerate(track_to_bbox.items()):
            x1, y1, x2, y2 = bbox
            if track_id == '00000':
                color = 'green'  # ego vehicle
            elif target_track and track_id == target_track:
                color = 'red'
            else:
                color = 'blue'
            rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor=color, facecolor='none')
            ax.add_patch(rect)
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            bbox_centers[track_id] = (center_x, center_y)
            label_text = 'EGO' if track_id == '00000' else f'{track_id[:8]}'
            ax.text(x1, y1 - 5, label_text, color=color, fontsize=8, weight='bold')
        
        for relation in spatial_relations:
            s_track = relation['source_object']['track_id']
            t_track = relation['target_object']['track_id']
            
            if target_track and s_track != target_track:
                continue
            
            relations_text = ', '.join(relation['relations'])
            if s_track in bbox_centers and t_track in bbox_centers:
                x1, y1 = bbox_centers[s_track]
                x2, y2 = bbox_centers[t_track]
                ax.plot([x1, x2], [y1, y2], 'g-', linewidth=1, alpha=0.5)
                mid_x = (x1 + x2) / 2
                mid_y = (y1 + y2) / 2
                ax.text(mid_x, mid_y, relations_text, color='yellow', fontsize=8, weight='bold', 
                       bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
        
        ax.axis('off')
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=150)
        plt.show()
