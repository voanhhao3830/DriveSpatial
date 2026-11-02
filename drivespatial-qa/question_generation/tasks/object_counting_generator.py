import os
import json
import math
import random
import logging
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
from uuid import uuid4
from collections import Counter

from sources.question_generation.enum import RelationQuestionType
from sources.question_generation.utils import visualize_multiview
from sources.db.sql import SQLiteController
from sources.data_collection.enum.data_sources import DataSource

logger = logging.getLogger(__name__)


class ObjectCountingGenerator:
    
    QUESTION_TYPE = RelationQuestionType.SCENE_COUNTING
    
    QUESTION_TEMPLATE = """Given the current driving images, how many {object_type} appear within {distance}m around the ego vehicle?
"""
    
    # Map camera keys to dataset-specific names
    CAMERA_ORDER = {
        'nuscenes': ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
                     'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT'],
        'once': ['cam01', 'cam03', 'cam05', 'cam06', 'cam07', 'cam08', 'cam09'],
        'waymo': ['FRONT_LEFT', 'FRONT', 'FRONT_RIGHT', 'SIDE_LEFT', 'SIDE_RIGHT'],
        'truckscenes': ['CAMERA_LEFT_FRONT', 'CAMERA_RIGHT_FRONT',
                        'CAMERA_LEFT_BACK', 'CAMERA_RIGHT_BACK'],
        'argoverse2': ['ring_front_left', 'ring_front_center', 'ring_front_right',
                       'ring_side_left', 'ring_side_right',
                       'ring_rear_left', 'ring_rear_right']
    }
    
    # Object types to consider for counting questions
    COUNTABLE_OBJECT_TYPES = [
        'REGULAR_VEHICLE',
        'PEDESTRIAN', 
        'BICYCLIST',
        'MOTORCYCLIST',
        'BUS',
        'TRUCK',
        'CONSTRUCTION_BARREL',
        'SIGN',
        'BOLLARD',
    ]
    
    # Map raw labels to standardized names for question text
    LABEL_TO_DISPLAY_NAME = {
        'REGULAR_VEHICLE': 'regular vehicles',
        'PEDESTRIAN': 'pedestrians',
        'BICYCLIST': 'bicyclists',
        'MOTORCYCLIST': 'motorcyclists',
        'BUS': 'buses',
        'TRUCK': 'trucks',
        'CONSTRUCTION_BARREL': 'construction barrels',
        'SIGN': 'signs',
        'BOLLARD': 'bollards',
    }
    
    # Distance ranges to use for questions (in meters)
    DISTANCE_RANGES = [10, 15, 20, 25, 30]
    
    # Count range options
    COUNT_OPTIONS = ['0', '1-2', '3-4', '5+']
    
    @staticmethod
    def generate_question(
        sampled_data: Dict[str, Any],
        db_path: str,
        save_path: str,
        num_options: int = 4,
        distance_range: Optional[int] = None,
        target_object_type: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Generate an object counting question.
        
        Args:
            sampled_data: Sampled data record from database
            db_path: Path to the database
            save_path: Path to save questions and images
            num_options: Number of options to generate (default: 4)
            distance_range: Distance range in meters (default: random from DISTANCE_RANGES)
            target_object_type: Specific object type to count (default: random selection)
            
        Returns:
            question: Question dictionary or None if generation fails
        """
        try:
            # Extract and parse JSON fields if they are strings (from database)
            # Parse annotation data
            annotation = sampled_data.get("annotation", {})
            if isinstance(annotation, str):
                try:
                    annotation = json.loads(annotation)
                    sampled_data["annotation"] = annotation  # Update for visualize_multiview
                except json.JSONDecodeError:
                    logger.warning("Failed to parse annotation JSON")
                    return None
            
            views = sampled_data.get("views", {})
            if isinstance(views, str):
                try:
                    views = json.loads(views)
                    sampled_data["views"] = views  # Update for visualize_multiview
                except json.JSONDecodeError:
                    logger.warning("Failed to parse views JSON")
                    return None
            
            # Also parse metadata if needed
            metadata_raw = sampled_data.get("metadata", {})
            if isinstance(metadata_raw, str):
                try:
                    metadata = json.loads(metadata_raw)
                    sampled_data["metadata"] = metadata
                except json.JSONDecodeError:
                    metadata = {}
            else:
                metadata = metadata_raw
            
            # Extract metadata fields
            source = metadata.get("source")
            scene_id = sampled_data.get("scene_id")
            frame_id = sampled_data.get("frame_id")
            
            # Select distance range if not specified
            if distance_range is None:
                distance_range = random.choice(ObjectCountingGenerator.DISTANCE_RANGES)
            
            # Step 1: Count objects by type within distance range
            object_counts = ObjectCountingGenerator._count_objects_within_range(
                annotation, distance_range
            )
            
            if not object_counts:
                logger.warning("No objects found within distance range for counting")
                return None
            
            # Step 2: Select target object type
            if target_object_type is None:
                # Prioritize types that have some objects for more interesting questions
                available_types = [t for t in object_counts.keys() 
                                   if t in ObjectCountingGenerator.COUNTABLE_OBJECT_TYPES]
                if not available_types:
                    logger.warning("No countable object types found")
                    return None
                target_object_type = random.choice(available_types)
            
            # Get actual count for the selected type
            actual_count = object_counts.get(target_object_type, 0)
            
            # Get objects info for visualization
            objects_in_range = ObjectCountingGenerator._get_objects_within_range(
                annotation, views, source, distance_range, target_object_type
            )
            
            # Create save directories
            question_folder = Path(save_path) / "relation" / ObjectCountingGenerator.QUESTION_TYPE.value
            images_folder = question_folder / "images"
            images_folder.mkdir(parents=True, exist_ok=True)
            
            # Generate unique ID for this question
            question_id = str(uuid4())
            
            # Step 3: Visualize the scene with highlighted objects of the target type
            multiview_path = images_folder / f"{question_id}_multiview.jpg"
            
            # Create highlight objects for visualization
            highlight_objects = []
            for idx, obj_info in enumerate(objects_in_range):
                highlight_objects.append({
                    'camera': obj_info['camera'],
                    'bbox_idx_in_camera': obj_info['bbox_idx_in_camera'],
                    'color': (0, 255, 0),  # Green in BGR
                    'label': f'({idx + 1})'
                })
            
            visualize_multiview(
                data=sampled_data,
                output_path=str(multiview_path),
                source=source,
                show_object_ids=False,
                highlight_objects=highlight_objects if highlight_objects else None,
                show_only_highlighted=False,  # Show full scene for counting context
                hide_object_ids_for_highlighted=True
            )
            
            # Step 4: Generate question text
            display_name = ObjectCountingGenerator.LABEL_TO_DISPLAY_NAME.get(
                target_object_type, target_object_type.lower().replace('_', ' ')
            )
            question_text = ObjectCountingGenerator.QUESTION_TEMPLATE.format(
                object_type=display_name,
                distance=distance_range
            )
            
            # Step 5: Generate count options
            options, correct_answer_idx = ObjectCountingGenerator._generate_count_options(
                actual_count, num_options
            )
            
            question_text += f"\nMulti-view visualization: <image>\n\nOptions:\n"
            for idx, option in enumerate(options):
                question_text += f"({chr(65+idx)}) {option}\n"
            
            # Step 6: Create question dictionary
            question_dict = {
                "question_id": question_id,
                "question_type": ObjectCountingGenerator.QUESTION_TYPE.value,
                "question": question_text,
                "images": [str(multiview_path)],
                "options": options,
                "correct_answer": correct_answer_idx,
                "metadata": {
                    "scene_id": scene_id,
                    "frame_id": frame_id,
                    "source": source,
                    "record_id": sampled_data.get("id"),
                    "reasoning_chain": {
                        "target_object_type": target_object_type,
                        "distance_range_meters": distance_range,
                        "actual_count": actual_count,
                        "all_counts_in_range": object_counts,
                        "objects_info": [
                            {
                                'track_id': obj['track_id'],
                                'label': obj['label'],
                                'distance_from_ego': obj['distance_from_ego'],
                                'position': {'x': obj['x'], 'y': obj['y'], 'z': obj['z']}
                            }
                            for obj in objects_in_range
                        ]
                    }
                }
            }
            
            return question_dict
            
        except Exception as e:
            logger.error(f"Error generating object counting question: {e}", exc_info=True)
            return None
    
    @staticmethod
    def _get_distance_from_ego(x: float, y: float, z: float) -> float:
        """
        Calculate Euclidean distance from ego (0, 0, 0) to object center.
        
        Args:
            x, y, z: Object center coordinates
            
        Returns:
            Distance in meters
        """
        return math.sqrt(x**2 + y**2 + z**2)
    
    @staticmethod
    def _count_objects_within_range(
        annotation: Dict[str, Any],
        distance_range: float
    ) -> Dict[str, int]:
        """
        Count objects by type within the specified distance range from ego.
        
        Args:
            annotation: Annotation data with 3D bboxes and labels
            distance_range: Maximum distance from ego in meters
            
        Returns:
            Dictionary mapping object type to count
        """
        bbox_3d = annotation.get('bbox_3d', [])
        label_3d = annotation.get('label_3d', [])
        track_3d = annotation.get('track_3d', [])
        
        if not bbox_3d or not label_3d:
            return {}
        
        counts: Dict[str, int] = Counter()
        counted_tracks: set = set()  # Avoid double counting same object
        
        for idx, (bbox, label) in enumerate(zip(bbox_3d, label_3d)):
            if not bbox or len(bbox) < 7:
                continue
            
            # Get track_id to avoid double counting
            track_id = track_3d[idx] if idx < len(track_3d) else None
            if track_id and track_id in counted_tracks:
                continue
            
            # bbox_3d format: [x, y, z, length, width, height, yaw]
            x, y, z = bbox[0], bbox[1], bbox[2]
            
            distance = ObjectCountingGenerator._get_distance_from_ego(x, y, z)
            
            if distance <= distance_range:
                # Normalize label
                normalized_label = label.upper().replace(' ', '_')
                counts[normalized_label] += 1
                if track_id:
                    counted_tracks.add(track_id)
        
        return dict(counts)
    
    @staticmethod
    def _get_objects_within_range(
        annotation: Dict[str, Any],
        views: Dict[str, Any],
        source: str,
        distance_range: float,
        target_object_type: str
    ) -> List[Dict[str, Any]]:
        """
        Get detailed info about objects of target type within distance range.
        
        Args:
            annotation: Annotation data
            views: Camera views dictionary
            source: Data source name
            distance_range: Maximum distance from ego
            target_object_type: Type of objects to find
            
        Returns:
            List of object info dictionaries
        """
        bbox_3d = annotation.get('bbox_3d', [])
        label_3d = annotation.get('label_3d', [])
        track_3d = annotation.get('track_3d', [])
        track_2d = annotation.get('track_2d', [])
        
        if not bbox_3d or not label_3d:
            return []
        
        original_camera_order = list(views.keys())
        objects_info = []
        found_tracks: set = set()
        
        for idx, (bbox, label) in enumerate(zip(bbox_3d, label_3d)):
            if not bbox or len(bbox) < 7:
                continue
            
            # Check if this is the target object type
            normalized_label = label.upper().replace(' ', '_')
            if normalized_label != target_object_type.upper():
                continue
            
            track_id = track_3d[idx] if idx < len(track_3d) else None
            if track_id and track_id in found_tracks:
                continue
            
            x, y, z = bbox[0], bbox[1], bbox[2]
            distance = ObjectCountingGenerator._get_distance_from_ego(x, y, z)
            
            if distance > distance_range:
                continue
            
            # Find which camera this object appears in
            camera_name = None
            bbox_idx_in_camera = None
            
            if track_id:
                for cam_idx, cam_name in enumerate(original_camera_order):
                    if cam_idx >= len(track_2d):
                        continue
                    tracks_in_camera = track_2d[cam_idx]
                    if track_id in tracks_in_camera:
                        camera_name = cam_name
                        bbox_idx_in_camera = tracks_in_camera.index(track_id)
                        break
            
            if camera_name is None:
                continue
            
            objects_info.append({
                'object_id': idx + 1,
                'track_id': track_id,
                'label': label,
                'x': x,
                'y': y,
                'z': z,
                'distance_from_ego': round(distance, 2),
                'camera': camera_name,
                'bbox_idx_in_camera': bbox_idx_in_camera
            })
            
            if track_id:
                found_tracks.add(track_id)
        
        return objects_info
    
    @staticmethod
    def _generate_count_options(
        actual_count: int,
        num_options: int = 4
    ) -> Tuple[List[str], int]:
        """
        Generate count range options based on actual count.
        
        Args:
            actual_count: The actual count of objects
            num_options: Number of options to generate
            
        Returns:
            Tuple of (options_list, correct_answer_index)
        """
        # Define count ranges
        count_ranges = [
            (0, 0, '0'),
            (1, 2, '1-2'),
            (3, 4, '3-4'),
            (5, 6, '5-6'),
            (7, 9, '7-9'),
            (10, float('inf'), '10+')
        ]
        
        # Find correct range
        correct_option = None
        for min_val, max_val, label in count_ranges:
            if min_val <= actual_count <= max_val:
                correct_option = label
                break
        
        if correct_option is None:
            correct_option = '10+'
        
        # Build options list
        all_options = [label for _, _, label in count_ranges]
        
        # Ensure we have the correct option and select distractors
        if len(all_options) <= num_options:
            options = all_options[:num_options]
        else:
            # Include correct answer and nearby ranges for realistic distractors
            correct_idx = all_options.index(correct_option)
            
            # Get neighboring options (more realistic distractors)
            selected_indices = {correct_idx}
            for offset in [-1, 1, -2, 2, -3, 3]:
                neighbor_idx = correct_idx + offset
                if 0 <= neighbor_idx < len(all_options):
                    selected_indices.add(neighbor_idx)
                if len(selected_indices) >= num_options:
                    break
            
            # If still need more, add random ones
            while len(selected_indices) < num_options:
                remaining = [i for i in range(len(all_options)) if i not in selected_indices]
                if remaining:
                    selected_indices.add(random.choice(remaining))
                else:
                    break
            
            # Sort indices to maintain order
            sorted_indices = sorted(selected_indices)[:num_options]
            options = [all_options[i] for i in sorted_indices]
        
        # Find correct answer index
        if correct_option in options:
            correct_answer_idx = options.index(correct_option)
        else:
            # This shouldn't happen, but handle gracefully
            options[0] = correct_option
            correct_answer_idx = 0
        
        return options, correct_answer_idx