import os
import json
import random
import math
import logging
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
from uuid import uuid4

from sources.question_generation.enum import RelationQuestionType
from sources.question_generation.utils import visualize_multiview
from sources.db.sql import SQLiteController
from sources.data_collection.enum.data_sources import DataSource

logger = logging.getLogger(__name__)


class SpatialCompatibilityGenerator:
    """
    Generates spatial compatibility questions.
    
    Question format:
    "Given the current driving images, can you drive through between 
    Object #{id1} and Object #{id2}?
    Options: A. Yes; B. No"
    """
    
    QUESTION_TYPE = RelationQuestionType.SPATIAL_COMPABILITY
    
    QUESTION_TEMPLATE = """Given the current driving images. Can you drive through between Object #{object1_id} ({object1_class}) and Object #{object2_id} ({object2_class})?
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
    
    # Distance thresholds (in meters)
    MAX_DISTANCE = 8.0  # Only consider object pairs closer than this
    PASSABLE_DISTANCE = 5.0  # Can pass through if distance > this
    
    # Distance threshold for near object preference (in meters)
    NEAR_OBJECT_THRESHOLD = 20.0
    # Probability to select from near objects (80%)
    NEAR_OBJECT_PROBABILITY = 0.8
    
    @staticmethod
    def generate_question(
        sampled_data: Dict[str, Any],
        db_path: str,
        save_path: str,
        num_options: int = 2
    ) -> Optional[Dict[str, Any]]:
        """
        Generate a spatial compatibility question.
        
        Args:
            sampled_data: Sampled data record from database
            db_path: Path to the database
            save_path: Path to save questions and images
            num_options: Number of options to generate (default: 2 for Yes/No)
            
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
            
            # Step 1: Find two objects with appropriate distance
            object_pair = SpatialCompatibilityGenerator._find_object_pair(annotation)
            
            if not object_pair:
                logger.warning("No valid object pair found with appropriate distance")
                return None
            
            object1, object2, distance, can_pass = object_pair
            
            # Create save directories
            question_folder = Path(save_path) / "relation" / SpatialCompatibilityGenerator.QUESTION_TYPE.value
            images_folder = question_folder / "images"
            images_folder.mkdir(parents=True, exist_ok=True)
            
            # Generate unique ID for this question
            question_id = str(uuid4())
            
            # Step 2: Visualize multi-view with highlighted objects
            multiview_path = images_folder / f"{question_id}_multiview.jpg"
            visualize_multiview(
                data=sampled_data,
                output_path=str(multiview_path),
                source=source,
                show_object_ids=True,
                highlight_objects=[
                    {
                        'object_id': object1['object_id'],
                        'color': (255, 0, 0),  # Blue in BGR
                        'label': '(obj1)'
                    },
                    {
                        'object_id': object2['object_id'],
                        'color': (255, 0, 255),  # Magenta in BGR
                        'label': '(obj2)'
                    }
                ],
                show_only_highlighted=True
            )
            
            # Step 3: Generate question text
            question_text = SpatialCompatibilityGenerator.QUESTION_TEMPLATE.format(
                object1_id=object1['object_id'],
                object1_class=object1['label'],
                object2_id=object2['object_id'],
                object2_class=object2['label']
            )
            
            # Step 4: Generate Yes/No options
            options = ["Yes", "No"]
            correct_answer_idx = 0 if can_pass else 1  # 0 for Yes, 1 for No
            
            question_text += f"\nMulti-view visualization: <image>\n\nOptions:\n"
            for idx, option in enumerate(options):
                question_text += f"({chr(65+idx)}) {option}\n"
            
            # Step 5: Create question dictionary
            question_dict = {
                "question_id": question_id,
                "question_type": SpatialCompatibilityGenerator.QUESTION_TYPE.value,
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
                        "object1_id": object1['object_id'],
                        "object1_label": object1['label'],
                        "object2_id": object2['object_id'],
                        "object2_label": object2['label'],
                        "distance": distance,
                        "can_pass": can_pass,
                        "passable_threshold": SpatialCompatibilityGenerator.PASSABLE_DISTANCE
                    }
                }
            }
            
            return question_dict
            
        except Exception as e:
            logger.error(f"Error generating spatial compatibility question: {e}", exc_info=True)
            return None
    
    @staticmethod
    def _get_object_distance_from_ego(bbox_3d: List[float]) -> float:
        """
        Calculate distance from ego (0,0,0) to object center.
        
        Args:
            bbox_3d: 3D bounding box [x, y, z, ...]
            
        Returns:
            Distance in meters, or float('inf') if invalid
        """
        if not bbox_3d or len(bbox_3d) < 3:
            return float('inf')
        
        x, y, z = bbox_3d[0], bbox_3d[1], bbox_3d[2]
        return math.sqrt(x**2 + y**2 + z**2)
    
    @staticmethod
    def _weighted_choice_pair(
        pairs: List[Tuple[Dict[str, Any], Dict[str, Any], float, bool]],
        near_threshold: float = 20.0,
        near_probability: float = 0.8
    ) -> Tuple[Dict[str, Any], Dict[str, Any], float, bool]:
        """
        Select a pair with 80% probability from pairs where at least one object is near ego.
        
        Args:
            pairs: List of (obj1, obj2, distance, can_pass) tuples
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
            obj1, obj2, distance, can_pass = pair
            dist1 = SpatialCompatibilityGenerator._get_object_distance_from_ego(obj1.get('bbox_3d', []))
            dist2 = SpatialCompatibilityGenerator._get_object_distance_from_ego(obj2.get('bbox_3d', []))
            
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
        annotation: Dict[str, Any]
    ) -> Optional[Tuple[Dict[str, Any], Dict[str, Any], float, bool]]:
        """
        Find two objects with distance < 5m and determine if you can pass through.
        Prefers object pairs where both objects are near ego (< 20m).
        
        Args:
            annotation: Annotation data with 3D bboxes
            
        Returns:
            Tuple of (object1, object2, distance, can_pass) or None
        """
        bbox_3d = annotation.get('bbox_3d', [])
        track_3d = annotation.get('track_3d', [])
        label_3d = annotation.get('label_3d', [])
        
        if not bbox_3d or len(bbox_3d) < 2:
            logger.warning("Not enough objects to create spatial compatibility question")
            return None
        
        # Find all valid pairs with distance < MAX_DISTANCE
        valid_pairs = []
        
        for i in range(len(bbox_3d)):
            for j in range(i + 1, len(bbox_3d)):
                # Get object positions (x, y, z, l, w, h, yaw)
                bbox_i = bbox_3d[i]
                bbox_j = bbox_3d[j]
                
                # Calculate 2D distance (x, y only, ignoring z)
                pos_i = (bbox_i[0], bbox_i[1])
                pos_j = (bbox_j[0], bbox_j[1])
                
                distance = math.sqrt((pos_i[0] - pos_j[0])**2 + (pos_i[1] - pos_j[1])**2)
                
                # Only consider pairs with distance < MAX_DISTANCE
                if distance < SpatialCompatibilityGenerator.MAX_DISTANCE:
                    # Determine if can pass through
                    can_pass = distance > SpatialCompatibilityGenerator.PASSABLE_DISTANCE
                    
                    obj1 = {
                        'object_id': i + 1,  # 1-indexed
                        'track_id': track_3d[i] if i < len(track_3d) else None,
                        'label': label_3d[i] if i < len(label_3d) else "unknown",
                        'bbox_3d': bbox_i
                    }
                    
                    obj2 = {
                        'object_id': j + 1,  # 1-indexed
                        'track_id': track_3d[j] if j < len(track_3d) else None,
                        'label': label_3d[j] if j < len(label_3d) else "unknown",
                        'bbox_3d': bbox_j
                    }
                    
                    valid_pairs.append((obj1, obj2, distance, can_pass))
        
        if not valid_pairs:
            logger.warning(f"No object pairs found with distance < {SpatialCompatibilityGenerator.MAX_DISTANCE}m")
            return None
        
        # 80% probability to select pairs where at least one object is near ego (< 20m)
        selected_pair = SpatialCompatibilityGenerator._weighted_choice_pair(
            pairs=valid_pairs,
            near_threshold=SpatialCompatibilityGenerator.NEAR_OBJECT_THRESHOLD,
            near_probability=SpatialCompatibilityGenerator.NEAR_OBJECT_PROBABILITY
        )
        
        logger.info(f"Selected object pair: #{selected_pair[0]['object_id']} and #{selected_pair[1]['object_id']}, "
                   f"distance: {selected_pair[2]:.2f}m, can_pass: {selected_pair[3]}")
        
        return selected_pair
