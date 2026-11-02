import os
import json
import random
import logging
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
from uuid import uuid4

from sources.question_generation.enum import RelationQuestionType
from sources.question_generation.utils import visualize_multiview
from sources.db.sql import SQLiteController
from sources.data_collection.enum.data_sources import DataSource
from sources.data_collection.enum.interaction_types import Relation

logger = logging.getLogger(__name__)


class AllocentricImaginationGenerator:
    """
    Generates allocentric imagination questions.
    
    Question format:
    "Given the current driving images:
    - Image-1: <image/>
    - Image-2: <image/>
    ...
    Imagine you are Object #{object-1}. Where is Object #{object-2} compared to you?
    Option: A. On your left; B. On your right; C. In front of you; D. Behind you"
    
    The two reference objects must be in different camera views.
    Uses spatial relations to determine the answer.
    """
    
    QUESTION_TYPE = RelationQuestionType.ALLOCENTRIC_IMAGINATION
    
    QUESTION_TEMPLATE = """Given the current driving images. Imagine you are Object #{ego_object_id} ({ego_object_class}). Where is Object #{target_object_id} ({target_object_class}) compared to you?
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
    
    # Map Relation enum to natural language for question options
    # Simplified to 4 primary directions for cleaner questions
    RELATION_TO_DIRECTION = {
        Relation.AHEAD_OF: 'in front of you',
        Relation.BEHIND: 'behind you',
        Relation.LEFT_OF: 'on your left',
        Relation.RIGHT_OF: 'on your right',
        Relation.AHEAD_LEFT_OF: 'in front of you',  # Simplified
        Relation.AHEAD_RIGHT_OF: 'in front of you',  # Simplified
        Relation.REAR_LEFT_OF: 'behind you',  # Simplified
        Relation.REAR_RIGHT_OF: 'behind you',  # Simplified
        # Vertical relations (less common for this question type)
        Relation.ABOVE: 'above you',
        Relation.BELOW: 'below you'
    }
    
    # Primary directions for options (4 main directions)
    PRIMARY_DIRECTIONS = [
        'in front of you',
        'behind you',
        'on your left',
        'on your right'
    ]
    
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
        Generate an allocentric imagination question.
        
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
            spatial_relations = sampled_data.get("spatial_relations", {})
            
            # Step 1: Find two objects in different camera views
            object_pair = AllocentricImaginationGenerator._find_objects_in_different_cameras(
                annotation, views, source
            )
            
            if not object_pair:
                logger.warning("No valid object pair found in different cameras")
                return None
            
            ego_object, target_object = object_pair
            
            # Create save directories
            question_folder = Path(save_path) / "relation" / AllocentricImaginationGenerator.QUESTION_TYPE.value
            images_folder = question_folder / "images"
            images_folder.mkdir(parents=True, exist_ok=True)
            
            # Generate unique ID for this question
            question_id = str(uuid4())
            
            # Step 2: Visualize multi-view with 2D bounding boxes and object IDs
            multiview_path = images_folder / f"{question_id}_multiview.jpg"
            visualize_multiview(
                data=sampled_data,
                output_path=str(multiview_path),
                source=source,
                show_object_ids=True,
                highlight_objects=[
                    {
                        'object_id': ego_object['object_id'],
                        'color': (255, 0, 0),  # Blue in BGR
                        'label': '(you)'
                    },
                    {
                        'object_id': target_object['object_id'],
                        'color': (0, 0, 255),  # Red in BGR
                        'label': '(target)'
                    }
                ],
                show_only_highlighted=True
            )
            
            # Step 3: Calculate where target_object is relative to ego_object
            relative_position = AllocentricImaginationGenerator._find_relative_position(
                ego_object=ego_object,
                target_object=target_object,
                annotation=annotation,
                spatial_relations=spatial_relations
            )
            
            if not relative_position:
                logger.warning("Could not determine relative position")
                return None
            
            # Step 4: Generate question text
            # Build image list
            num_cameras = len([v for v in views.values() if v])
            
            question_text = AllocentricImaginationGenerator.QUESTION_TEMPLATE.format(
                ego_object_id=ego_object['object_id'],
                ego_object_class=ego_object['label'],
                target_object_id=target_object['object_id'],
                target_object_class=target_object['label']
            )
            
            # Step 5: Generate multiple choice options based on spatial directions
            correct_answer = relative_position  # e.g., "on your left"
            options, correct_answer_idx = AllocentricImaginationGenerator._generate_direction_options(
                correct_answer=correct_answer,
                num_options=num_options
            )
            
            question_text += f"\nMulti-view visualization: <image>\n\nOptions:\n"
            for idx, direction in enumerate(options):
                question_text += f"({chr(65+idx)}) {direction.capitalize()}\n"
            
            # Step 6: Create question dictionary
            question_dict = {
                "question_id": question_id,
                "question_type": AllocentricImaginationGenerator.QUESTION_TYPE.value,
                "question": question_text,
                "images": [str(multiview_path)],
                "options": [direction.capitalize() for direction in options],
                "correct_answer": correct_answer_idx,
                "metadata": {
                    "scene_id": scene_id,
                    "frame_id": frame_id,
                    "source": source,
                    "record_id": sampled_data.get("id"),
                    "reasoning_chain": {
                        "ego_object_id": ego_object['object_id'],
                        "ego_object_label": ego_object['label'],
                        "ego_object_camera": ego_object['camera'],
                        "target_object_id": target_object['object_id'],
                        "target_object_label": target_object['label'],
                        "target_object_camera": target_object['camera'],
                        "relative_position": relative_position
                    }
                }
            }
            
            return question_dict
            
        except Exception as e:
            logger.error(f"Error generating allocentric imagination question: {e}", exc_info=True)
            return None
    
    @staticmethod
    def _get_object_distance_from_ego(
        bbox_3d: List[float]
    ) -> float:
        """
        Calculate distance from ego (0,0,0) to object center.
        
        Args:
            bbox_3d: 3D bounding box [x, y, z, ...]
            
        Returns:
            Distance in meters, or float('inf') if invalid
        """
        import math
        
        if not bbox_3d or len(bbox_3d) < 3:
            return float('inf')
        
        x, y, z = bbox_3d[0], bbox_3d[1], bbox_3d[2]
        return math.sqrt(x**2 + y**2 + z**2)
    
    @staticmethod
    def _weighted_choice_pair(
        pairs: List[Tuple[Dict[str, Any], Dict[str, Any]]],
        near_threshold: float = 20.0,
        near_probability: float = 0.8
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Select a pair with 80% probability from pairs where at least one object is near.
        
        Args:
            pairs: List of object pairs
            near_threshold: Distance threshold for near objects (default: 20m)
            near_probability: Probability to select from near pairs (default: 0.8)
            
        Returns:
            Selected pair
        """
        if not pairs:
            return None
        
        if len(pairs) == 1:
            return pairs[0]
        
        # Separate pairs into near and far groups
        near_pairs = []
        far_pairs = []
        for pair in pairs:
            obj1, obj2 = pair
            dist1 = AllocentricImaginationGenerator._get_object_distance_from_ego(obj1.get('bbox_3d', []))
            dist2 = AllocentricImaginationGenerator._get_object_distance_from_ego(obj2.get('bbox_3d', []))
            
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
    def _find_objects_in_different_cameras(
        annotation: Dict[str, Any],
        views: Dict[str, Any],
        source: str
    ) -> Optional[Tuple[Dict[str, Any], Dict[str, Any]]]:
        """
        Find two objects that appear in different camera views.
        
        Args:
            annotation: Annotation data with 2D/3D bboxes and track IDs
            views: Camera views dictionary
            source: Data source name
            
        Returns:
            Tuple of (approach_object, yaw_object) or None
        """
        bbox_2d = annotation.get('bbox_2d', [])
        track_2d = annotation.get('track_2d', [])
        track_3d = annotation.get('track_3d', [])
        label_3d = annotation.get('label_3d', [])
        bbox_3d = annotation.get('bbox_3d', [])
        
        if not bbox_2d or not track_2d or not track_3d or not label_3d:
            logger.warning("Missing annotation data for camera-based object finding")
            return None
        
        source_lower = source.lower() if source else None
        camera_order = AllocentricImaginationGenerator.CAMERA_ORDER.get(source_lower, [])
        original_camera_order = list(views.keys())
        
        # Build mapping: object_id -> list of cameras where visible
        object_to_cameras: Dict[int, List[str]] = {}
        
        for cam_idx, cam_name in enumerate(original_camera_order):
            if cam_idx >= len(track_2d):
                continue
            
            tracks_in_camera = track_2d[cam_idx]
            
            for track_id in tracks_in_camera:
                if track_id not in track_3d:
                    continue
                
                obj_idx = track_3d.index(track_id)
                object_id = obj_idx + 1  # 1-indexed
                
                if object_id not in object_to_cameras:
                    object_to_cameras[object_id] = []
                object_to_cameras[object_id].append(cam_name)
        
        # Find pairs of objects in different cameras
        valid_pairs = []
        
        object_ids = list(object_to_cameras.keys())
        for i, obj1_id in enumerate(object_ids):
            for obj2_id in object_ids[i+1:]:
                # Get cameras for each object
                cameras1 = set(object_to_cameras[obj1_id])
                cameras2 = set(object_to_cameras[obj2_id])
                
                # Check if they are in completely different cameras
                if not cameras1.intersection(cameras2):
                    # Get object details
                    obj1_idx = obj1_id - 1
                    obj2_idx = obj2_id - 1
                    
                    if obj1_idx >= len(label_3d) or obj2_idx >= len(label_3d):
                        continue
                    if obj1_idx >= len(bbox_3d) or obj2_idx >= len(bbox_3d):
                        continue
                    
                    obj1 = {
                        'object_id': obj1_id,
                        'track_id': track_3d[obj1_idx],
                        'label': label_3d[obj1_idx],
                        'bbox_3d': bbox_3d[obj1_idx],
                        'camera': list(cameras1)[0]  # Primary camera
                    }
                    
                    obj2 = {
                        'object_id': obj2_id,
                        'track_id': track_3d[obj2_idx],
                        'label': label_3d[obj2_idx],
                        'bbox_3d': bbox_3d[obj2_idx],
                        'camera': list(cameras2)[0]  # Primary camera
                    }
                    
                    valid_pairs.append((obj1, obj2))
        
        if not valid_pairs:
            logger.warning(f"No valid object pairs found in different cameras. "
                          f"Objects: {len(object_to_cameras)}, Camera mappings: {object_to_cameras}")
            return None
        
        # 80% probability to select pairs where at least one object is near (< 20m from ego)
        selected_pair = AllocentricImaginationGenerator._weighted_choice_pair(
            pairs=valid_pairs,
            near_threshold=AllocentricImaginationGenerator.NEAR_OBJECT_THRESHOLD,
            near_probability=AllocentricImaginationGenerator.NEAR_OBJECT_PROBABILITY
        )
        
        # Randomly assign which is approach vs yaw reference
        if random.random() > 0.5:
            return selected_pair
        else:
            return (selected_pair[1], selected_pair[0])
    
    @staticmethod
    def _find_relative_position(
        ego_object: Dict[str, Any],
        target_object: Dict[str, Any],
        annotation: Dict[str, Any],
        spatial_relations: List[Dict[str, Any]]
    ) -> Optional[str]:
        """
        Find where target_object is relative to ego_object using pre-computed spatial relations.
        
        Uses the existing spatial_relations data to find the relation between
        ego_object and target_object directly.
        
        Args:
            ego_object: Object to imagine being (use its position and yaw)
            target_object: Object to locate relative to ego
            annotation: Full annotation data
            spatial_relations: Spatial relations between objects (pre-computed)
            
        Returns:
            Relative position string (e.g., "on your left", "behind you") or None
        """
        ego_track_id = ego_object['track_id']
        target_track_id = target_object['track_id']
        
        # Find the relation where ego is source and target is target
        for relation_data in spatial_relations:
            if not isinstance(relation_data, dict):
                continue
            
            source_obj = relation_data.get('source_object', {})
            target_obj = relation_data.get('target_object', {})
            relations = relation_data.get('relations', [])
            
            source_track = source_obj.get('track_id')
            target_track = target_obj.get('track_id')
            
            # Check if this is the ego -> target relation
            if source_track == ego_track_id and target_track == target_track_id:
                if not relations:
                    continue
                
                # Get the primary relation (first one)
                primary_relation = relations[0] if isinstance(relations, list) else relations
                
                # Convert Relation enum to natural language
                if isinstance(primary_relation, Relation):
                    # Direct enum access
                    return AllocentricImaginationGenerator.RELATION_TO_DIRECTION.get(
                        primary_relation, 
                        None
                    )
                elif isinstance(primary_relation, str):
                    # Handle string format (enum name or value)
                    # Try to convert string to Relation enum
                    try:
                        # Handle formats like "Relation.AHEAD_OF" or "ahead_of"
                        relation_name = primary_relation.split('.')[-1].upper()
                        if hasattr(Relation, relation_name):
                            relation_enum = getattr(Relation, relation_name)
                            return AllocentricImaginationGenerator.RELATION_TO_DIRECTION.get(
                                relation_enum,
                                None
                            )
                    except Exception:
                        pass
                
                logger.warning(f"Unknown relation format: {primary_relation} (type: {type(primary_relation)})")
                return None
        
        logger.warning(f"No spatial relation found from {ego_track_id} to {target_track_id}")
        return None
    
    @staticmethod
    def _generate_direction_options(
        correct_answer: str,
        num_options: int = 4
    ) -> Tuple[List[str], int]:
        """
        Generate multiple choice options for spatial directions.
        Uses PRIMARY_DIRECTIONS (derived from Relation enum) for options.
        
        Args:
            correct_answer: Correct spatial direction (e.g., "on your left")
            num_options: Number of options (default: 4)
            
        Returns:
            Tuple of (options list, correct answer index)
        """
        all_directions = AllocentricImaginationGenerator.PRIMARY_DIRECTIONS.copy()
        
        # Ensure correct answer is in the list
        if correct_answer not in all_directions:
            logger.warning(f"Correct answer '{correct_answer}' not in primary directions, adding it")
            all_directions.append(correct_answer)
        
        # If we need fewer options, sample (but we have exactly 4, so this is just safety)
        if len(all_directions) > num_options:
            # Keep correct answer and sample others
            other_directions = [d for d in all_directions if d != correct_answer]
            selected_others = random.sample(other_directions, num_options - 1)
            options = [correct_answer] + selected_others
        else:
            options = all_directions[:num_options]
        
        # Shuffle
        shuffled_indices = list(range(len(options)))
        random.shuffle(shuffled_indices)
        shuffled_options = [options[i] for i in shuffled_indices]
        
        # Find correct answer index after shuffle
        correct_answer_idx = next(
            i for i, opt in enumerate(shuffled_options) 
            if opt == correct_answer
        )
        
        return shuffled_options, correct_answer_idx