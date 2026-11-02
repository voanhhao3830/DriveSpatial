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


class EgoRelativePositionGenerator:
    QUESTION_TYPE = RelationQuestionType.DIRECTION
    
    QUESTION_TEMPLATE = """Given the current driving images, where is Object-1 compared to us?
"""
    
    # Map Relation enum to natural language directions
    RELATION_TO_DIRECTION = {
        Relation.AHEAD_OF: "in front of you",
        Relation.BEHIND: "behind you",
        Relation.LEFT_OF: "on your left",
        Relation.RIGHT_OF: "on your right",
        Relation.AHEAD_LEFT_OF: "ahead and to your left",
        Relation.AHEAD_RIGHT_OF: "ahead and to your right",
        Relation.REAR_LEFT_OF: "behind and to your left",
        Relation.REAR_RIGHT_OF: "behind and to your right",
        Relation.ABOVE: "above you",
        Relation.BELOW: "below you",
    }
    
    # All available directions for option generation
    ALL_DIRECTIONS = [
        "in front of you",
        "behind you",
        "on your left",
        "on your right",
        "ahead and to your left",
        "ahead and to your right",
        "behind and to your left",
        "behind and to your right",
        "above you",
        "below you"
    ]
    
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
    
    @staticmethod
    def generate_question(
        sampled_data: Dict[str, Any],
        db_path: str,
        save_path: str,
        num_options: int = 4
    ) -> Optional[Dict[str, Any]]:
        """
        Generate an ego-relative position question.
        
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
            spatial_relations = sampled_data.get("spatial_relations", [])
            
            if not spatial_relations:
                logger.warning("No spatial relations data available")
                return None
            
            # Step 1: Find an object with ego relation
            object_selection = EgoRelativePositionGenerator._find_object_with_ego_relation(
                annotation, views, source, spatial_relations
            )
            
            if not object_selection:
                logger.warning("No valid object found with ego relation")
                return None
            
            object_info, relation_direction = object_selection
            
            # Create save directories
            question_folder = Path(save_path) / "relation" / EgoRelativePositionGenerator.QUESTION_TYPE.value
            images_folder = question_folder / "images"
            images_folder.mkdir(parents=True, exist_ok=True)
            
            # Generate unique ID for this question
            question_id = str(uuid4())
            
            # Step 2: Visualize only the selected object
            multiview_path = images_folder / f"{question_id}_multiview.jpg"
            visualize_multiview(
                data=sampled_data,
                output_path=str(multiview_path),
                source=source,
                show_object_ids=True,
                highlight_objects=[
                    {
                        'camera': object_info['camera'],
                        'bbox_idx_in_camera': object_info['bbox_idx_in_camera'],
                        'color': (0, 255, 0),  # Green in BGR
                        'label': '(1)'
                    }
                ],
                show_only_highlighted=True,  # Only show this object
                hide_object_ids_for_highlighted=True  # Only show (1), not #ID
            )
            
            # Step 3: Generate question text
            question_text = EgoRelativePositionGenerator.QUESTION_TEMPLATE
            
            # Step 4: Generate options with correct answer
            options, correct_answer_idx = EgoRelativePositionGenerator._generate_direction_options(
                relation_direction, num_options
            )
            
            question_text += f"\nMulti-view visualization: <image>\n\nOptions:\n"
            for idx, option in enumerate(options):
                question_text += f"({chr(65+idx)}) {option}\n"
            
            # Step 5: Create question dictionary
            question_dict = {
                "question_id": question_id,
                "question_type": EgoRelativePositionGenerator.QUESTION_TYPE.value,
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
                        "object_id": object_info['object_id'],
                        "object_label": object_info['label'],
                        "object_camera": object_info['camera'],
                        "object_track_id": object_info['track_id'],
                        "relation_to_ego": object_info.get('relation_enum', 'unknown'),
                        "direction": relation_direction
                    }
                }
            }
            
            return question_dict
            
        except Exception as e:
            logger.error(f"Error generating ego relative position question: {e}", exc_info=True)
            return None
    
    @staticmethod
    def _find_object_with_ego_relation(
        annotation: Dict[str, Any],
        views: Dict[str, Any],
        source: str,
        spatial_relations: List[Dict[str, Any]]
    ) -> Optional[Tuple[Dict[str, Any], str]]:
        """
        Find an object that has a spatial relation with the ego vehicle.
        
        Args:
            annotation: Annotation data with 2D/3D bboxes and track IDs
            views: Camera views dictionary
            source: Data source name
            spatial_relations: List of spatial relation dictionaries
            
        Returns:
            Tuple of (object_info, relation_direction) or None
        """
        bbox_2d = annotation.get('bbox_2d', [])
        track_2d = annotation.get('track_2d', [])
        track_3d = annotation.get('track_3d', [])
        label_3d = annotation.get('label_3d', [])
        bbox_3d = annotation.get('bbox_3d', [])
        
        if not bbox_2d or not track_2d or not track_3d or not label_3d:
            logger.warning("Missing annotation data for ego relation")
            return None
        
        source_lower = source.lower() if source else None
        camera_order = EgoRelativePositionGenerator.CAMERA_ORDER.get(source_lower, [])
        original_camera_order = list(views.keys())
        
        # Build mapping: track_id -> (camera_name, object_index, bbox_idx_in_camera)
        track_to_info: Dict[str, Tuple[str, int, int]] = {}
        
        for cam_idx, cam_name in enumerate(original_camera_order):
            if cam_idx >= len(track_2d):
                continue
            
            tracks_in_camera = track_2d[cam_idx]
            
            for bbox_idx_in_cam, track_id in enumerate(tracks_in_camera):
                if track_id not in track_3d:
                    continue
                
                obj_idx = track_3d.index(track_id)
                
                # Store first occurrence of this track
                if track_id not in track_to_info:
                    track_to_info[track_id] = (cam_name, obj_idx, bbox_idx_in_cam)
        
        # Find relations where source is "ego"
        ego_relations = []
        for rel in spatial_relations:
            source_obj = rel.get('source_object', {})
            target_obj = rel.get('target_object', {})
            relations_list = rel.get('relations', [])
            
            # Check if source is ego
            source_label = source_obj.get('label', '')
            source_track = source_obj.get('track_id', '')
            target_track = target_obj.get('track_id', '')
            
            # Check if source is ego (label == 'ego' or track_id == '00000')
            if (source_label == 'ego' or source_track == '00000') and target_track in track_to_info:
                # Add each relation from the list
                for relation_str in relations_list:
                    ego_relations.append({
                        'target_track_id': target_track,
                        'relation': relation_str
                    })
        
        if not ego_relations:
            logger.warning("No ego-to-object relations found")
            return None
        
        # Randomly select one ego relation
        selected_relation = random.choice(ego_relations)
        target_track_id = selected_relation['target_track_id']
        relation_str = selected_relation['relation']
        
        # Convert relation string to Relation enum
        # Relations in data are lowercase (e.g., 'rear_left_of'), enum needs uppercase
        try:
            relation_enum_str = relation_str.upper()
            relation_enum = Relation[relation_enum_str]
        except (KeyError, ValueError, AttributeError):
            logger.warning(f"Unknown relation type: {relation_str}")
            return None
        
        # Get direction from relation
        direction = EgoRelativePositionGenerator.RELATION_TO_DIRECTION.get(
            relation_enum,
            "unknown"
        )
        
        if direction == "unknown":
            logger.warning(f"Could not map relation {relation_enum} to direction")
            return None
        
        # Get object info
        cam_name, obj_idx, bbox_idx_in_cam = track_to_info[target_track_id]
        
        object_info = {
            'object_id': obj_idx + 1,
            'track_id': target_track_id,
            'label': label_3d[obj_idx] if obj_idx < len(label_3d) else "unknown",
            'bbox_3d': bbox_3d[obj_idx] if obj_idx < len(bbox_3d) else None,
            'camera': cam_name,
            'bbox_idx_in_camera': bbox_idx_in_cam,
            'relation_enum': relation_str
        }
        
        logger.info(f"Selected object {target_track_id} with ego relation: {relation_enum} -> {direction}")
        
        return object_info, direction
    
    @staticmethod
    def _generate_direction_options(
        correct_direction: str,
        num_options: int = 4
    ) -> Tuple[List[str], int]:
        """
        Generate direction options including the correct answer.
        
        Args:
            correct_direction: The correct direction string
            num_options: Number of options to generate
            
        Returns:
            Tuple of (options_list, correct_answer_index)
        """
        # Start with all available directions
        all_directions = EgoRelativePositionGenerator.ALL_DIRECTIONS.copy()
        
        # Ensure correct answer is in the list
        if correct_direction not in all_directions:
            # This shouldn't happen, but handle it gracefully
            logger.warning(f"Correct direction '{correct_direction}' not in all directions")
            all_directions.append(correct_direction)
        
        # Select options
        if len(all_directions) <= num_options:
            options = all_directions
        else:
            # Ensure correct answer is included
            other_directions = [d for d in all_directions if d != correct_direction]
            random.shuffle(other_directions)
            options = [correct_direction] + other_directions[:num_options-1]
        
        # Shuffle options
        random.shuffle(options)
        
        # Find correct answer index
        correct_answer_idx = options.index(correct_direction)
        
        return options, correct_answer_idx