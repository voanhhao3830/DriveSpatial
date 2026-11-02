import os
import json
import math
import random
import logging
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
from uuid import uuid4

from sources.question_generation.enum import RelationQuestionType
from sources.question_generation.utils import visualize_multiview
from sources.db.sql import SQLiteController
from sources.data_collection.enum.data_sources import DataSource

logger = logging.getLogger(__name__)


class ObjectDistanceGenerator:
    QUESTION_TYPE = RelationQuestionType.DISTANCE
    
    QUESTION_TEMPLATE = """Given the current driving images, what is the approximate distance between Object-1 and Object-2?
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
    
    # Distance threshold for near object preference (in meters)
    NEAR_OBJECT_THRESHOLD = 20.0
    # Probability to select from near objects (80%)
    NEAR_OBJECT_PROBABILITY = 0.8
    
    @staticmethod
    def generate_question(
        sampled_data: Dict[str, Any],
        db_path: str,
        save_path: str,
        num_options: int = 4
    ) -> Optional[Dict[str, Any]]:
        """
        Generate an object distance estimation question.
        
        Args:
            sampled_data: Sampled data record from database
            db_path: Path to the database
            save_path: Path to save questions and images
            num_options: Number of options to generate (default: 4)
            
        Returns:
            question: Question dictionary or None if generation fails
        """
        try:
            # Extract metadata
            metadata = sampled_data.get("metadata", {})
            source = metadata.get("source")
            scene_id = sampled_data.get("scene_id")
            frame_id = sampled_data.get("frame_id")
            
            # Extract annotation data
            annotation = sampled_data.get("annotation", {})
            views = sampled_data.get("views", {})
            
            # Step 1: Find object pair (different objects in different views)
            object_selection = ObjectDistanceGenerator._find_object_pair(
                annotation, views, source
            )
            
            if not object_selection:
                logger.warning("No valid object pair found for distance estimation")
                return None
            
            object1_info, object2_info, actual_distance = object_selection
            
            # Create save directories
            question_folder = Path(save_path) / "relation" / ObjectDistanceGenerator.QUESTION_TYPE.value
            images_folder = question_folder / "images"
            images_folder.mkdir(parents=True, exist_ok=True)
            
            # Generate unique ID for this question
            question_id = str(uuid4())
            
            # Step 2: Visualize only the two objects with different colors
            multiview_path = images_folder / f"{question_id}_multiview.jpg"
            visualize_multiview(
                data=sampled_data,
                output_path=str(multiview_path),
                source=source,
                show_object_ids=True,
                highlight_objects=[
                    {
                        'camera': object1_info['camera'],
                        'bbox_idx_in_camera': object1_info['bbox_idx_in_camera'],
                        'color': (255, 0, 0),  # Blue in BGR
                        'label': '(1)'
                    },
                    {
                        'camera': object2_info['camera'],
                        'bbox_idx_in_camera': object2_info['bbox_idx_in_camera'],
                        'color': (255, 0, 255),  # Magenta in BGR
                        'label': '(2)'
                    }
                ],
                show_only_highlighted=True,  # Only show these two objects
                hide_object_ids_for_highlighted=True  # Only show (1) and (2), not #ID
            )
            
            # Step 3: Generate question text
            question_text = ObjectDistanceGenerator.QUESTION_TEMPLATE
            
            # Step 4: Generate distance options
            options, correct_answer_idx = ObjectDistanceGenerator._generate_distance_options(
                actual_distance, num_options
            )
            
            question_text += f"\nMulti-view visualization: <image>\n\nOptions:\n"
            for idx, option in enumerate(options):
                question_text += f"({chr(65+idx)}) {option}\n"
            
            # Step 5: Create question dictionary
            question_dict = {
                "question_id": question_id,
                "question_type": ObjectDistanceGenerator.QUESTION_TYPE.value,
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
                        "object1_id": object1_info['object_id'],
                        "object1_label": object1_info['label'],
                        "object1_camera": object1_info['camera'],
                        "object1_track_id": object1_info['track_id'],
                        "object1_position": {
                            'x': object1_info['x'],
                            'y': object1_info['y'],
                            'z': object1_info['z']
                        },
                        "object2_id": object2_info['object_id'],
                        "object2_label": object2_info['label'],
                        "object2_camera": object2_info['camera'],
                        "object2_track_id": object2_info['track_id'],
                        "object2_position": {
                            'x': object2_info['x'],
                            'y': object2_info['y'],
                            'z': object2_info['z']
                        },
                        "actual_distance": actual_distance
                    }
                }
            }
            
            return question_dict
            
        except Exception as e:
            logger.error(f"Error generating object distance question: {e}", exc_info=True)
            return None
    
    @staticmethod
    def _get_object_distance_from_ego(x: float, y: float, z: float) -> float:
        """
        Calculate distance from ego (0,0,0) to object center.
        
        Args:
            x, y, z: Object center coordinates
            
        Returns:
            Distance in meters
        """
        return math.sqrt(x**2 + y**2 + z**2)
    
    @staticmethod
    def _weighted_choice_pair(
        pairs: List[Tuple[Dict[str, Any], Dict[str, Any], float]],
        near_threshold: float = 20.0,
        near_probability: float = 0.8
    ) -> Tuple[Dict[str, Any], Dict[str, Any], float]:
        """
        Select a pair with 80% probability from pairs where at least one object is near ego.
        
        Args:
            pairs: List of (obj1, obj2, distance) tuples
            near_threshold: Distance threshold for near objects (default: 20m)
            near_probability: Probability to select from near pairs (default: 0.8)
            
        Returns:
            Selected pair tuple
        """
        if not pairs:
            return None
        
        if len(pairs) == 1:
            return pairs[0]
        
        # Separate pairs into near and far groups
        near_pairs = []
        far_pairs = []
        for pair in pairs:
            obj1, obj2, distance = pair
            dist1 = ObjectDistanceGenerator._get_object_distance_from_ego(obj1['x'], obj1['y'], obj1['z'])
            dist2 = ObjectDistanceGenerator._get_object_distance_from_ego(obj2['x'], obj2['y'], obj2['z'])
            
            # Consider "near" if at least one object is within threshold
            if dist1 < near_threshold or dist2 < near_threshold:
                near_pairs.append(pair)
            else:
                far_pairs.append(pair)
        
        # Select with probability-based approach
        pick_near = random.random() < near_probability
        
        if pick_near and near_pairs:
            return random.choice(near_pairs)
        elif far_pairs:
            return random.choice(far_pairs)
        elif near_pairs:
            return random.choice(near_pairs)
        else:
            return random.choice(pairs)
    
    @staticmethod
    def _find_object_pair(
        annotation: Dict[str, Any],
        views: Dict[str, Any],
        source: str
    ) -> Optional[Tuple[Dict[str, Any], Dict[str, Any], float]]:
        """
        Find two different objects in different camera views for distance estimation.
        Prefers object pairs where both objects are near ego (< 20m).
        
        Args:
            annotation: Annotation data with 2D/3D bboxes and track IDs
            views: Camera views dictionary
            source: Data source name
            
        Returns:
            Tuple of (object1_info, object2_info, distance) or None
        """
        bbox_2d = annotation.get('bbox_2d', [])
        track_2d = annotation.get('track_2d', [])
        track_3d = annotation.get('track_3d', [])
        label_3d = annotation.get('label_3d', [])
        bbox_3d = annotation.get('bbox_3d', [])
        
        if not bbox_2d or not track_2d or not track_3d or not label_3d or not bbox_3d:
            logger.warning("Missing annotation data for distance estimation")
            return None
        
        source_lower = source.lower() if source else None
        camera_order = ObjectDistanceGenerator.CAMERA_ORDER.get(source_lower, [])
        original_camera_order = list(views.keys())
        
        # Build list of all objects with their camera and bbox info
        all_objects = []
        
        for cam_idx, cam_name in enumerate(original_camera_order):
            if cam_idx >= len(track_2d):
                continue
            
            tracks_in_camera = track_2d[cam_idx]
            
            for bbox_idx_in_cam, track_id in enumerate(tracks_in_camera):
                if track_id not in track_3d:
                    continue
                
                obj_idx = track_3d.index(track_id)
                
                if obj_idx >= len(bbox_3d):
                    continue
                
                bbox = bbox_3d[obj_idx]
                if not bbox or len(bbox) < 7:
                    continue
                
                # bbox_3d format: [x, y, z, length, width, height, yaw]
                x, y, z, length, width, height, yaw = bbox
                
                object_info = {
                    'object_id': obj_idx + 1,
                    'track_id': track_id,
                    'label': label_3d[obj_idx] if obj_idx < len(label_3d) else "unknown",
                    'bbox_3d': bbox,
                    'camera': cam_name,
                    'bbox_idx_in_camera': bbox_idx_in_cam,
                    'x': x,
                    'y': y,
                    'z': z
                }
                
                all_objects.append(object_info)
        
        if len(all_objects) < 2:
            logger.warning("Not enough objects for distance estimation")
            return None
        
        # Collect all valid pairs first
        valid_pairs = []
        
        for i, obj1 in enumerate(all_objects):
            for obj2 in all_objects[i+1:]:
                # Must be different objects
                if obj1['track_id'] == obj2['track_id']:
                    continue
                
                # Prefer different cameras
                if obj1['camera'] == obj2['camera']:
                    continue
                
                # Calculate 3D distance between the two objects
                distance = math.sqrt(
                    (obj1['x'] - obj2['x'])**2 +
                    (obj1['y'] - obj2['y'])**2 +
                    (obj1['z'] - obj2['z'])**2
                )
                
                # Only select pairs with reasonable distance (5m to 50m)
                # Too close or too far makes the question less meaningful
                if distance < 5.0 or distance > 50.0:
                    continue
                
                valid_pairs.append((obj1, obj2, distance))
        
        if not valid_pairs:
            logger.warning("Could not find suitable object pair for distance estimation")
            return None
        
        # 80% probability to select pairs where at least one object is near ego (< 20m)
        selected = ObjectDistanceGenerator._weighted_choice_pair(
            pairs=valid_pairs,
            near_threshold=ObjectDistanceGenerator.NEAR_OBJECT_THRESHOLD,
            near_probability=ObjectDistanceGenerator.NEAR_OBJECT_PROBABILITY
        )
        
        obj1, obj2, distance = selected
        logger.info(f"Selected objects for distance estimation: "
                   f"obj1={obj1['track_id']}, obj2={obj2['track_id']}, "
                   f"distance={distance:.2f}m")
        
        return selected
    
    @staticmethod
    def _generate_distance_options(
        actual_distance: float,
        num_options: int = 4
    ) -> Tuple[List[str], int]:
        """
        Generate distance options around the actual distance.
        
        Args:
            actual_distance: The actual distance in meters
            num_options: Number of options to generate
            
        Returns:
            Tuple of (options_list, correct_answer_index)
        """
        # Round to appropriate precision based on distance magnitude
        if actual_distance < 10:
            # For short distances, use 0.5m bins (e.g., "5-6 meters")
            bin_size = 1.0
            rounded_distance = round(actual_distance)
        elif actual_distance < 20:
            # For medium distances, use 2m bins (e.g., "10-12 meters")
            bin_size = 2.0
            rounded_distance = round(actual_distance / bin_size) * bin_size
        else:
            # For longer distances, use 5m bins (e.g., "25-30 meters")
            bin_size = 5.0
            rounded_distance = round(actual_distance / bin_size) * bin_size
        
        # Generate the correct option
        if bin_size == 1.0:
            correct_option = f"{int(rounded_distance)}-{int(rounded_distance + bin_size)} meters"
        else:
            correct_option = f"{int(rounded_distance)}-{int(rounded_distance + bin_size)} meters"
        
        # Generate distractor options
        options_set = {correct_option}
        
        # Add options above and below
        for offset in [-2, -1, 1, 2, 3]:
            if len(options_set) >= num_options:
                break
            
            distractor_distance = rounded_distance + offset * bin_size
            if distractor_distance > 0:  # Distance must be positive
                if bin_size == 1.0:
                    option = f"{int(distractor_distance)}-{int(distractor_distance + bin_size)} meters"
                else:
                    option = f"{int(distractor_distance)}-{int(distractor_distance + bin_size)} meters"
                options_set.add(option)
        
        # Convert to list and shuffle
        options = list(options_set)[:num_options]
        
        # Ensure we have enough options
        while len(options) < num_options:
            # Add more distant options if needed
            distractor_distance = rounded_distance + random.choice([-3, -4, 4, 5]) * bin_size
            if distractor_distance > 0:
                if bin_size == 1.0:
                    option = f"{int(distractor_distance)}-{int(distractor_distance + bin_size)} meters"
                else:
                    option = f"{int(distractor_distance)}-{int(distractor_distance + bin_size)} meters"
                if option not in options:
                    options.append(option)
        
        # Sort options by distance for better readability
        options.sort(key=lambda x: int(x.split('-')[0]))
        
        # Find correct answer index
        correct_answer_idx = options.index(correct_option)
        
        return options, correct_answer_idx
