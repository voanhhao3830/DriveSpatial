"""
Occlusion Extractor

Calculates occlusion levels for each object based on 2D IOU overlapping
across all camera views. Projects 3D bounding boxes to 2D for each camera.
"""

import cv2
import logging
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from typing import Dict, Any, List, Tuple, Optional

logger = logging.getLogger(__name__)

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
    
# Visibility level definitions
VISIBILITY_LEVELS = [
    {
        "token": "f95237cfe4e0480c899fab6d5e5a235c",
        "level": 1,
        "min_percent": 0,
        "max_percent": 40,
        "description": "The object is 0% to 40% visible in panoramic view of all cameras."
    },
    {
        "token": "34fecb88f60d4842931ab1ce2187e6a0",
        "level": 2,
        "min_percent": 41,
        "max_percent": 60,
        "description": "The object is 41% to 60% visible in panoramic view of all cameras."
    },
    {
        "token": "d8bb855a234c4831a6671a99671bdefa",
        "level": 3,
        "min_percent": 61,
        "max_percent": 80,
        "description": "The object is 61% to 80% visible in panoramic view of all cameras"
    },
    {
        "token": "6adc1826065b452195ae6a76317b8ab5",
        "level": 4,
        "min_percent": 81,
        "max_percent": 100,
        "description": "The object is 81% to 100% visible in panoramic view of all cameras."
    }
]


class OcclusionExtractor:
    @staticmethod
    def project_3d_bbox_to_2d(
        bbox_3d: List[float],
        lidar2img: np.ndarray,
        img_shape: Tuple[int, int] = (1600, 900),
        source: str = "",
        cam_idx: int = 0
    ) -> Optional[List[float]]:
        """
        Project 3D bounding box to 2D image plane.
        
        Args:
            bbox_3d: [x, y, z, length, width, height, yaw]
            lidar2img: 4x4 transformation matrix from lidar to image coordinates
            img_shape: (width, height) of image
            source: Dataset source (e.g., 'waymo', 'nuscenes')
            cam_idx: Camera index (0=front, others need special handling for Waymo)
            
        Returns:
            2D bounding box [x1, y1, x2, y2] or None if not visible
        """
        x, y, z, length, width, height, yaw = bbox_3d
        
        # Get 8 corners of 3D bounding box in lidar coordinates
        # Center is (x, y, z), dimensions are (length, width, height)
        l, w, h = length / 2, width / 2, height / 2
        
        # Define 8 corners relative to center
        corners_3d = np.array([
            [l, w, -h],   [l, w, h],    [l, -w, -h],  [l, -w, h],
            [-l, w, -h],  [-l, w, h],   [-l, -w, -h], [-l, -w, h]
        ])
        
        # Rotate corners by yaw
        cos_yaw, sin_yaw = np.cos(yaw), np.sin(yaw)
        rot_matrix = np.array([
            [cos_yaw, -sin_yaw, 0],
            [sin_yaw, cos_yaw, 0],
            [0, 0, 1]
        ])
        
        corners_3d = corners_3d @ rot_matrix.T
        
        # Translate to actual position
        corners_3d += np.array([x, y, z])
        
        # Convert to homogeneous coordinates
        corners_3d_hom = np.hstack([corners_3d, np.ones((8, 1))])
        
        # Apply Waymo camera transformation for non-front cameras
        # This aligns the camera coordinate system
        # if source.lower() == 'waymo' and cam_idx != 0:
        #     T_front_cam_to_ref = np.array([
        #         [0.0, -1.0, 0.0, 0.0],
        #         [0.0, 0.0, -1.0, 0.0],
        #         [1.0, 0.0, 0.0, 0.0],
        #         [0.0, 0.0, 0.0, 1.0]
        #     ])
        #     # Modify lidar2img to include the camera frame transformation
        #     lidar2img = lidar2img @ T_front_cam_to_ref
        
        # Project to image
        corners_2d_hom = (lidar2img @ corners_3d_hom.T).T
        
        # Normalize by depth (z coordinate)
        depths = corners_2d_hom[:, 2]

        # Filter points behind camera (depth <= 0)
        if np.all(depths <= 0):
            print(f"All points are behind the camera: {corners_2d_hom}")
            return None
        
        # Normalize to get 2D coordinates
        corners_2d = corners_2d_hom[:, :2] / corners_2d_hom[:, 2:3]
        
        # Get bounding box from projected corners
        x1, y1 = corners_2d[:, 0].min(), corners_2d[:, 1].min()
        x2, y2 = corners_2d[:, 0].max(), corners_2d[:, 1].max()
        
        # Clip to image bounds
        x1 = max(0, min(x1, img_shape[0]))
        y1 = max(0, min(y1, img_shape[1]))
        x2 = max(0, min(x2, img_shape[0]))
        y2 = max(0, min(y2, img_shape[1]))
        
        # Check if box is valid
        if x2 <= x1 or y2 <= y1:
            print(f"Box is not valid: {x1}, {y1}, {x2}, {y2}")
            return None
        
        return [float(x1), float(y1), float(x2), float(y2)]

    @staticmethod
    def calculate_iou(box1: List[float], box2: List[float]) -> float:
        """
        Calculate 2D IOU between two bounding boxes.
        
        Args:
            box1: [x1, y1, x2, y2]
            box2: [x1, y1, x2, y2]
            
        Returns:
            IOU value between 0 and 1
        """
        try:
            x1_min, y1_min, x1_max, y1_max = box1
            x2_min, y2_min, x2_max, y2_max = box2
            
            # Calculate intersection
            inter_x_min = max(x1_min, x2_min)
            inter_y_min = max(y1_min, y2_min)
            inter_x_max = min(x1_max, x2_max)
            inter_y_max = min(y1_max, y2_max)
            
            if inter_x_max <= inter_x_min or inter_y_max <= inter_y_min:
                return 0.0
            
            inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
            
            # Calculate union
            box1_area = (x1_max - x1_min) * (y1_max - y1_min)
            box2_area = (x2_max - x2_min) * (y2_max - y2_min)
            union_area = box1_area + box2_area - inter_area
            
            if union_area <= 0:
                return 0.0
            
            iou = inter_area / union_area
            return float(iou)
            
        except Exception as e:
            logger.error(f"Error calculating IOU: {e}")
            return 0.0
    
    @staticmethod
    def get_visibility_level(visibility_percent: float) -> Dict[str, Any]:
        """
        Map visibility percentage to visibility level.
        
        Args:
            visibility_percent: Percentage of object visible (0-100)
            
        Returns:
            Visibility level dictionary
        """
        for level_info in VISIBILITY_LEVELS:
            if level_info["min_percent"] <= visibility_percent <= level_info["max_percent"]:
                return {
                    "level": level_info["level"],
                    "token": level_info["token"],
                    "visibility_percent": visibility_percent,
                    "description": level_info["description"]
                }
        
        # Default to level 4 if out of range
        return {
            "level": 4,
            "token": VISIBILITY_LEVELS[3]["token"],
            "visibility_percent": visibility_percent,
            "description": VISIBILITY_LEVELS[3]["description"]
        }
    
    @staticmethod
    def _intersection_area(box1: List[float], box2: List[float]) -> float:
        """
        Compute intersection area of two 2D boxes [x1, y1, x2, y2].
        """
        inter_x_min = max(box1[0], box2[0])
        inter_y_min = max(box1[1], box2[1])
        inter_x_max = min(box1[2], box2[2])
        inter_y_max = min(box1[3], box2[3])
        if inter_x_max <= inter_x_min or inter_y_max <= inter_y_min:
            return 0.0
        return (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)

    @staticmethod
    def extract_occlusions(video_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Extract occlusion information for objects in the scene.
        
        Strategy:
        - For waymo, argoverse2, once: Calculate ego-to-object visibility (ego as source)
        - For truckscenes, nuscenes: Use existing visibility attribute from dataset
        
        For computed visibility:
        1. Project 3D bboxes to 2D per camera
        2. Accumulate target's total projected area and area occluded by other objects (when others are closer)
        3. Visibility of target = (target_area - occluded_area) / target_area
        4. Output: list of { source_object: ego, target_object, visibility } for all objects.

        Args:
            video_data: List of frame data dictionaries

        Returns:
            video_data: Each frame updated with occlusions list.
        """
        results: List[Dict[str, Any]] = []

        for frame in video_data:
            metadata = frame.get("metadata", {})
            source = metadata.get("source", "").lower()
            
            annotation = frame.get("annotation", {})
            calibration = frame.get("calibration", {})

            bbox_3d = annotation.get("bbox_3d", [])
            track_3d = annotation.get("track_3d", [])
            label_3d = annotation.get("label_3d", [])
            visibility = annotation.get("visibility", [])  # Existing visibility from dataset

            lidar2img_list = calibration.get("lidar2img", [])

            # Check if data is present (handle both lists and numpy arrays)
            if len(bbox_3d) == 0 or len(track_3d) == 0:
                logger.warning(
                    "Missing annotation data, skipping occlusion extraction"
                )
                frame["occlusions"] = []
                results.append(frame)
                continue
            
            num_objects = len(track_3d)
            labels = label_3d if len(label_3d) == num_objects else [""] * num_objects
            
            # Determine which strategy to use based on dataset
            if source in ['truckscenes', 'nuscenes']:
                # Use existing visibility attribute from dataset
                occlusion_list = OcclusionExtractor._use_existing_visibility(
                    track_3d, labels, visibility, frame=frame
                )
                logger.info(
                    f"Using existing visibility for {source}: {len(occlusion_list)} objects"
                )
            else:
                # Compute ego-to-object visibility for waymo, argoverse2, once
                if len(lidar2img_list) == 0:
                    logger.warning(
                        "Missing calibration data for computed visibility, skipping occlusion extraction"
                    )
                    frame["occlusions"] = []
                    results.append(frame)
                    continue
                
                occlusion_list = OcclusionExtractor._compute_ego_visibility(
                    bbox_3d, track_3d, labels, lidar2img_list, source=source,frame=frame
                )
                logger.info(
                    f"Computed ego visibility for {source}: {len(occlusion_list)} objects"
                )
            
            frame["occlusions"] = occlusion_list
            
            # Uncomment to enable debug visualization
            # OcclusionExtractor.visualize_occlusions(
            #     frame,
            #     occlusion_list,
            #     view_idx=0,  # or camera name like "CAMERA_LEFT_FRONT"
            #     save_path='debug_occlusions.png',
            # )
            
            results.append(frame)
        
        return results
    
    @staticmethod
    def _use_existing_visibility(
        track_3d: List[str],
        labels: List[str],
        visibility: List[Any],
        frame: Dict[str, Any] = None
    ) -> List[Dict[str, Any]]:
        """
        Use existing visibility attribute from dataset (truckscenes, nuscenes).
        
        For nuScenes/TruckScenes, visibility is per-camera (2D objects):
        visibility = [['1', '4', '3'], [], ['1', '1'], ...]  # Per camera, per 2D object
        
        We need to:
        1. Map 2D objects to 3D objects using track_2d and track_3d
        2. Get visibility for each 3D object across all cameras
        3. Use the best (highest) visibility level
        
        Args:
            track_3d: List of 3D track IDs
            labels: List of object labels (aligned with track_3d)
            visibility: Visibility data (can be per-object or per-camera-per-object)
            frame: Frame data with 2D annotations (needed for per-camera visibility)
            
        Returns:
            List of occlusion dictionaries with ego as source
        """
        occlusion_list = []
        
        # Check if visibility is 2D (per-camera) or 1D (per-object)
        is_2d_visibility = isinstance(visibility, list) and len(visibility) > 0 and isinstance(visibility[0], (list, tuple))
        
        if is_2d_visibility and frame is not None:
            # Handle 2D visibility (per-camera, per-object)
            annotation = frame.get('annotation', {})
            track_2d_per_cam = annotation.get('track_2d', [])
            
            # Build map of track_id to visibility levels across cameras
            track_visibility_map = {}
            
            for cam_idx, (vis_cam, tracks_cam) in enumerate(zip(visibility, track_2d_per_cam)):
                for obj_idx_2d, track_id in enumerate(tracks_cam):
                    if track_id == '00000':  # Skip ego
                        continue
                    
                    # Get visibility for this 2D object
                    if obj_idx_2d < len(vis_cam):
                        vis_str = vis_cam[obj_idx_2d]
                        
                        # Convert string to int level
                        try:
                            vis_level = int(vis_str)
                        except (ValueError, TypeError):
                            vis_level = 4  # Default to fully visible
                        
                        # Store visibility for this track (keep highest across cameras)
                        if track_id not in track_visibility_map:
                            track_visibility_map[track_id] = vis_level
                        else:
                            # Keep the highest visibility (best view)
                            track_visibility_map[track_id] = max(track_visibility_map[track_id], vis_level)
            
            # Create occlusion entries for all 3D objects
            for idx, track_id in enumerate(track_3d):
                if track_id == '00000':  # Skip ego
                    continue
                
                # Get visibility level (default to 4 if not found in any camera)
                vis_level = track_visibility_map.get(track_id, 4)
                
                level_info = OcclusionExtractor.get_visibility_level(
                    # Convert level to percentage for get_visibility_level
                    (vis_level - 1) * 30 + 20 if vis_level <= 4 else 100
                )
                
                occlusion_list.append({
                    "source_object": {
                        "track_id": "00000",  # Ego vehicle
                        "label": "ego",
                    },
                    "target_object": {
                        "track_id": track_id,
                        "label": labels[idx] if idx < len(labels) else 'unknown',
                    },
                    "visibility": level_info,
                })
        
        else:
            # Handle 1D visibility (per-object) - legacy format
            # Ensure visibility list has same length as objects
            if len(visibility) != len(track_3d):
                logger.warning(
                    f"Visibility list length ({len(visibility)}) doesn't match objects ({len(track_3d)}), "
                    "using default visibility level 4"
                )
                visibility = [4] * len(track_3d)
            
            # Create ego-to-object entries
            for idx in range(len(track_3d)):
                if track_3d[idx] == '00000':  # Skip ego
                    continue
                
                vis_value = visibility[idx] if idx < len(visibility) else 4
                
                # Convert visibility to level (1-4) if needed
                if isinstance(vis_value, (int, float)):
                    if vis_value <= 4:
                        # Already a level (1-4)
                        level = int(vis_value)
                    else:
                        # Might be a percentage, convert to level
                        level_info = OcclusionExtractor.get_visibility_level(float(vis_value))
                        level = level_info["level"]
                else:
                    # Try to convert string to int
                    try:
                        level = int(vis_value)
                    except (ValueError, TypeError):
                        level = 4
                
                level_info = OcclusionExtractor.get_visibility_level(
                    (level - 1) * 30 + 20 if level <= 4 else 100
                )
                
                occlusion_list.append({
                    "source_object": {
                        "track_id": "00000",  # Ego vehicle
                        "label": "ego",
                    },
                    "target_object": {
                        "track_id": track_3d[idx],
                        "label": labels[idx] if idx < len(labels) else 'unknown',
                    },
                    "visibility": level_info,
                })
        
        return occlusion_list
    
    @staticmethod
    def _compute_ego_visibility(
        bbox_3d: List[List[float]],
        track_3d: List[str],
        labels: List[str],
        lidar2img_list: List[Any],
        source: str = "",
        frame: Dict[str, Any] = None
    ) -> List[Dict[str, Any]]:
        """
        Compute visibility of objects from ego vehicle perspective.
        
        Uses existing 2D bounding boxes from annotations instead of projecting 3D boxes.
        For each object:
        1. Find its 2D bbox in each camera using track_id matching
        2. Check occlusion by other objects (those closer to camera)
        3. Calculate visibility percentage across all cameras
        4. Use the highest visibility across cameras (best view)
        
        Also includes object-to-object occlusions (not just ego-to-object).
        
        Args:
            bbox_3d: List of 3D bounding boxes (for depth information)
            track_3d: List of track IDs
            labels: List of object labels
            lidar2img_list: List of lidar-to-image transformation matrices (unused, kept for API compatibility)
            source: Dataset source name
            frame: Frame data with 2D annotations
            
        Returns:
            List of occlusion dictionaries (both ego-to-object and object-to-object)
        """
        if frame is None:
            logger.error("Frame data is required for visibility computation")
            return []
        
        annotation = frame.get('annotation', {})
        bbox_2d_per_cam = annotation.get('bbox_2d', [])
        track_2d_per_cam = annotation.get('track_2d', [])
        
        num_objects = len(track_3d)
        num_cameras = len(bbox_2d_per_cam)
        
        # Per-object visibility info per camera
        # visibility_per_camera[obj_idx][cam_idx] = visibility_percent
        visibility_per_camera: Dict[int, Dict[int, float]] = {i: {} for i in range(num_objects)}
        
        # Track object-to-object occlusions
        object_occlusions: List[Dict[str, Any]] = []
        
        for cam_idx in range(num_cameras):
            bbox_2d_cam = bbox_2d_per_cam[cam_idx]
            track_2d_cam = track_2d_per_cam[cam_idx]
            
            # Build mapping from track_id to 2D bbox and object index
            track_to_info: Dict[str, Tuple[List[float], int, float]] = {}
            
            for obj_idx in range(num_objects):
                track_id = track_3d[obj_idx]
                
                # Find this track in the camera's 2D annotations
                if track_id in track_2d_cam:
                    bbox_2d_idx = track_2d_cam.index(track_id)
                    if bbox_2d_idx < len(bbox_2d_cam):
                        bbox_2d = bbox_2d_cam[bbox_2d_idx]
                        # bbox_2d format: [x1, y1, x2, y2]
                        
                        # Calculate depth as Euclidean distance from camera origin (0,0,0) to 3D bbox center
                        x, y, z = bbox_3d[obj_idx][0], bbox_3d[obj_idx][1], bbox_3d[obj_idx][2]
                        depth = np.sqrt(x**2 + y**2 + z**2)
                        
                        track_to_info[track_id] = (bbox_2d, obj_idx, depth)
            
            # Per-object total area and occluded area in this camera
            total_area_cam: Dict[int, float] = {}
            occluded_area_cam: Dict[int, float] = {}
            
            # Calculate area and occlusions for objects visible in this camera
            for target_track_id, (target_box, target_idx, target_depth) in track_to_info.items():
                target_area = (target_box[2] - target_box[0]) * (target_box[3] - target_box[1])
                
                # Skip invalid boxes
                if target_area <= 0:
                    continue
                
                total_area_cam[target_idx] = target_area
                occluded_area_cam[target_idx] = 0.0
                
                # Check occlusion by other objects in this camera
                for occluder_track_id, (occluder_box, occluder_idx, occluder_depth) in track_to_info.items():
                    if target_idx == occluder_idx:
                        continue
                    
                    # Only objects closer to camera can occlude (smaller depth = closer)
                    if occluder_depth >= target_depth:
                        continue
                    
                    # Calculate intersection area
                    overlap = OcclusionExtractor._intersection_area(occluder_box, target_box)
                    if overlap > 0:
                        occluded_area_cam[target_idx] += overlap
                        
                        # Record object-to-object occlusion
                        # Only add if not already recorded for this camera
                        occlusion_key = (occluder_idx, target_idx, cam_idx)
                        if not any(
                            occ.get('source_object', {}).get('track_id') == track_3d[occluder_idx] and
                            occ.get('target_object', {}).get('track_id') == track_3d[target_idx] and
                            occ.get('camera_idx') == cam_idx
                            for occ in object_occlusions
                        ):
                            object_occlusions.append({
                                'source_object': {
                                    'track_id': track_3d[occluder_idx],
                                    'label': labels[occluder_idx],
                                },
                                'target_object': {
                                    'track_id': track_3d[target_idx],
                                    'label': labels[target_idx],
                                },
                                'camera_idx': cam_idx,
                                'overlap_area': overlap,
                                'occluder_depth': occluder_depth,
                                'target_depth': target_depth,
                            })
            
            # Calculate visibility percentage for each object in this camera
            for obj_idx, obj_total in total_area_cam.items():
                obj_occluded = occluded_area_cam.get(obj_idx, 0.0)
                visibility_percent = ((obj_total - obj_occluded) / obj_total) * 100.0
                visibility_percent = min(100.0, max(0.0, visibility_percent))
                visibility_per_camera[obj_idx][cam_idx] = visibility_percent
        
        # Build ego-to-object visibility list
        # For each object, use the HIGHEST visibility across all cameras (best view)
        occlusion_list: List[Dict[str, Any]] = []
        for obj_idx in range(num_objects):
            camera_visibilities = visibility_per_camera.get(obj_idx, {})
            
            if len(camera_visibilities) == 0:
                # Object not visible in any camera
                visibility_percent = 100.0
            else:
                # Use the highest visibility (best camera view)
                visibility_percent = max(camera_visibilities.values())
            
            level_info = OcclusionExtractor.get_visibility_level(float(visibility_percent))
            occlusion_list.append({
                "source_object": {
                    "track_id": "00000",  # Ego vehicle
                    "label": "ego",
                },
                "target_object": {
                    "track_id": track_3d[obj_idx],
                    "label": labels[obj_idx],
                },
                "visibility": level_info,
                "best_visibility_percent": visibility_percent,
                "camera_visibilities": camera_visibilities,
            })
        
        # Add object-to-object occlusions to the list
        # For each unique occluder-target pair, calculate overall visibility
        obj_to_obj_map: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
        for occ in object_occlusions:
            key = (occ['source_object']['track_id'], occ['target_object']['track_id'])
            if key not in obj_to_obj_map:
                obj_to_obj_map[key] = []
            obj_to_obj_map[key].append(occ)
        
        for (source_track, target_track), occ_list in obj_to_obj_map.items():
            # Find the target object's index
            target_idx = track_3d.index(target_track) if target_track in track_3d else None
            if target_idx is None:
                continue
            
            # Get visibility for this target from the occluder's perspective
            # Use the average occlusion across cameras where it occurs
            avg_occluded_percent = sum(
                (occ['overlap_area'] / total_area_cam.get(target_idx, 1.0)) * 100.0
                for occ in occ_list
            ) / len(occ_list)
            
            visibility_percent = max(0.0, 100.0 - avg_occluded_percent)
            level_info = OcclusionExtractor.get_visibility_level(float(visibility_percent))
            
            occlusion_list.append({
                "source_object": {
                    "track_id": occ_list[0]['source_object']['track_id'],
                    "label": occ_list[0]['source_object']['label'],
                },
                "target_object": {
                    "track_id": occ_list[0]['target_object']['track_id'],
                    "label": occ_list[0]['target_object']['label'],
                },
                "visibility": level_info,
                "occlusion_type": "object_to_object",
                "cameras": [occ['camera_idx'] for occ in occ_list],
            })
        
        return occlusion_list
    
    @staticmethod
    def visualize_occlusions(
        frame: Dict[str, Any],
        occlusions: List[Dict[str, Any]],
        save_path: str = None,
        view_idx: int = 0,
        min_occlusion_level: int = 4,
    ):
        """
        Visualize occlusion relationships on a specific camera view.
        Now only shows ego-to-object visibility.
        
        Args:
            frame: Frame data with views, annotation, and calibration
            occlusions: List of occlusion dictionaries from extract_occlusions
            save_path: Path to save the visualization image
            view_idx: Camera index or camera name string
            min_occlusion_level: Only show occlusions with visibility level <= this value (1-4)
        """
        try:
            # Get camera information
            view_names = list(frame['views'].keys())
            if isinstance(view_idx, str):
                if view_idx not in view_names:
                    logger.warning(f"View {view_idx} not found. Available: {view_names}")
                    return
                cam_name = view_idx
                cam_idx = view_names.index(view_idx)
            else:
                if view_idx >= len(view_names):
                    logger.warning(f"View index {view_idx} out of range. Max: {len(view_names)-1}")
                    return
                cam_idx = view_idx
                cam_name = view_names[cam_idx]
            
            # Load image
            image_path = frame['views'][cam_name]
            if isinstance(image_path, dict):
                image_path = image_path.get('path', image_path)
            image = plt.imread(image_path)
            img_h, img_w = image.shape[:2]
            
            # Get annotations
            metadata = frame.get("metadata", {})
            source_dataset = metadata.get("source", "").lower()
            bbox_3d = frame['annotation']['bbox_3d']
            track_3d = frame['annotation']['track_3d']
            label_3d = frame['annotation']['label_3d']
            
            # Project all 3D boxes to 2D for this camera (only if we have calibration)
            projected_boxes = {}
            if source_dataset not in ['truckscenes', 'nuscenes']:
                # For datasets with computed visibility, project boxes
                lidar2img = np.array(frame['calibration']['lidar2img'][cam_idx])
                
                for obj_idx, track_id in enumerate(track_3d):
                    box_2d = OcclusionExtractor.project_3d_bbox_to_2d(
                        bbox_3d[obj_idx],
                        lidar2img,
                        img_shape=(img_w, img_h),
                        source=source_dataset,
                        cam_idx=cam_idx
                    )
                    if box_2d is not None:
                        projected_boxes[track_id] = {
                            'bbox': box_2d,
                            'label': label_3d[obj_idx],
                            'depth': bbox_3d[obj_idx][2]  # z coordinate
                        }
            else:
                # For datasets with existing visibility, use 2D annotations if available
                bbox_2d = frame['annotation'].get('bbox_2d', [])
                track_2d = frame['annotation'].get('track_2d', [])
                
                if cam_idx < len(bbox_2d) and cam_idx < len(track_2d):
                    for box, track_id in zip(bbox_2d[cam_idx], track_2d[cam_idx]):
                        obj_idx = track_3d.index(track_id) if track_id in track_3d else None
                        if obj_idx is not None:
                            projected_boxes[track_id] = {
                                'bbox': box,
                                'label': label_3d[obj_idx],
                                'depth': bbox_3d[obj_idx][2] if obj_idx < len(bbox_3d) else 0
                            }
            
            # Create visualization
            fig, ax = plt.subplots(1, figsize=(16, 12))
            ax.imshow(image)
            
            # Calculate bbox centers for drawing
            bbox_centers = {}
            for track_id, obj_info in projected_boxes.items():
                x1, y1, x2, y2 = obj_info['bbox']
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                bbox_centers[track_id] = (center_x, center_y)
            
            # Filter occlusions to visualize (only ego-to-object now)
            visible_occlusions = []
            for occ in occlusions:
                s_track = occ['source_object']['track_id']
                t_track = occ['target_object']['track_id']
                visibility = occ['visibility']
                
                # Only show if visibility level meets filter
                if visibility > min_occlusion_level:
                    continue
                
                # Only show if target object is visible in this camera
                if t_track in projected_boxes:
                    visible_occlusions.append(occ)
            
            # Draw bounding boxes
            for track_id, obj_info in projected_boxes.items():
                x1, y1, x2, y2 = obj_info['bbox']
                depth = obj_info['depth']
                label = obj_info['label']
                
                # Find visibility for this object
                vis_level = None
                for occ in visible_occlusions:
                    if occ['target_object']['track_id'] == track_id:
                        vis_level = occ['visibility']
                        break
                
                # Color based on visibility level
                if vis_level is not None:
                    if vis_level == 1:
                        color = (1.0, 0.0, 0.0)  # Red - Heavy occlusion
                    elif vis_level == 2:
                        color = (1.0, 0.5, 0.0)  # Orange - Moderate occlusion
                    elif vis_level == 3:
                        color = (1.0, 1.0, 0.0)  # Yellow - Light occlusion
                    else:
                        color = (0.0, 1.0, 0.0)  # Green - Minimal/no occlusion
                    linewidth = 3
                else:
                    # Not in filtered occlusions
                    color = (0.5, 0.5, 0.5)  # Gray
                    linewidth = 1
                
                rect = patches.Rectangle(
                    (x1, y1), x2 - x1, y2 - y1,
                    linewidth=linewidth,
                    edgecolor=color,
                    facecolor='none'
                )
                ax.add_patch(rect)
                
                # Add label
                label_text = f'{track_id[:8]}\n{label}\nd={depth:.1f}m'
                if vis_level is not None:
                    label_text += f'\nLvl {vis_level}'
                
                ax.text(
                    x1, y1 - 5,
                    label_text,
                    color=color,
                    fontsize=7,
                    weight='bold',
                    bbox=dict(boxstyle='round', facecolor='black', alpha=0.6)
                )
            
            # Add ego vehicle indicator at bottom center
            ego_x = img_w / 2
            ego_y = img_h - 30
            ax.plot(ego_x, ego_y, marker='o', markersize=15, color='cyan', markeredgecolor='white', markeredgewidth=2)
            ax.text(ego_x, ego_y - 20, 'EGO', color='cyan', fontsize=10, weight='bold',
                   ha='center', bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
            
            # Add legend
            legend_text = (
                f"Camera: {cam_name}\n"
                f"Objects visible: {len(projected_boxes)}\n"
                f"Objects with visibility ≤ Lvl {min_occlusion_level}: {len(visible_occlusions)}\n\n"
                f"Visibility Levels (Ego → Object):\n"
                f"  Lvl 1 (red): 0-40% visible\n"
                f"  Lvl 2 (orange): 41-60% visible\n"
                f"  Lvl 3 (yellow): 61-80% visible\n"
                f"  Lvl 4 (green): 81-100% visible"
            )
            ax.text(
                0.02, 0.98,
                legend_text,
                transform=ax.transAxes,
                fontsize=9,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='black', alpha=0.7),
                color='white'
            )
            
            ax.set_title(f'Ego-to-Object Visibility - {cam_name}', fontsize=14, weight='bold')
            ax.axis('off')
            
            if save_path:
                plt.savefig(save_path, bbox_inches='tight', dpi=150)
                logger.info(f"Saved occlusion visualization to {save_path}")
            
            plt.show()
            plt.close()
            
        except Exception as e:
            logger.error(f"Error visualizing occlusions: {e}", exc_info=True)
        except Exception as e:
            logger.error(f"Error visualizing occlusions: {e}", exc_info=True)


if __name__ == "__main__":
    # Test the extractor: pairwise (source_object, target_object, visibility)
    import numpy as np

    test_lidar2img = np.array([
        [1000, 0, 800, 0],
        [0, 1000, 450, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ])

    test_data = [
        {
            "annotation": {
                "bbox_3d": [
                    [5.0, 0.0, 0.5, 4.0, 2.0, 1.5, 0.0],
                    [10.0, 0.0, 0.5, 4.0, 2.0, 1.5, 0.0],
                    [8.0, 3.0, 0.5, 4.0, 2.0, 1.5, 0.0],
                ],
                "track_3d": ["track_1", "track_2", "track_3"],
                "label_3d": ["car", "truck", "bus"],
            },
            "calibration": {
                "lidar2img": [test_lidar2img.tolist(), test_lidar2img.tolist()],
            },
        }
    ]

    result = OcclusionExtractor.extract_occlusions(test_data)

    print("\nPairwise occlusion (source_object, target_object, visibility):")
    for frame in result:
        for occ in frame["occlusions"]:
            src = occ["source_object"]
            tgt = occ["target_object"]
            vis = occ["visibility"]
            print(
                f"  {src['track_id']} ({src['label']}) -> {tgt['track_id']} ({tgt['label']}): "
                f"{vis['visibility_percent']:.1f}% visible (level={vis['level']}, token={vis['token']})"
            )
