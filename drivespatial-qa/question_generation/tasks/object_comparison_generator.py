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


class ObjectComparisonGenerator:
    
    QUESTION_TYPE = RelationQuestionType.MULTI_VIEW_DEPTH_AWARENESS
    
    HEIGHT_QUESTION_TEMPLATE = """Given the current driving images, is Object-1 taller than Object-2?
"""
    
    DISTANCE_QUESTION_TEMPLATE = """Given the current driving images, is Object-1 nearer to us than Object-2?
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
    
    @staticmethod
    def generate_question(
        sampled_data: Dict[str, Any],
        db_path: str,
        save_path: str,
        num_options: int = 2
    ) -> Optional[Dict[str, Any]]:
        """
        Generate an object comparison question (height or distance).
        
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
            
            # Randomly select question type: height or distance
            question_type = random.choice(['height', 'distance'])
            
            # Step 1: Find object pair (different objects in different views)
            object_selection = ObjectComparisonGenerator._find_object_pair(
                annotation, views, source, question_type
            )
            
            if not object_selection:
                logger.warning(f"No valid object pair found for {question_type} comparison")
                return None
            
            object1_info, object2_info, comparison_result = object_selection
            
            # Create save directories
            question_folder = Path(save_path) / "relation" / ObjectComparisonGenerator.QUESTION_TYPE.value
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
            
            # Step 3: Generate question text based on type
            if question_type == 'height':
                question_text = ObjectComparisonGenerator.HEIGHT_QUESTION_TEMPLATE
                comparison_value = object1_info.get('height', 0) - object2_info.get('height', 0)
            else:  # distance
                question_text = ObjectComparisonGenerator.DISTANCE_QUESTION_TEMPLATE
                comparison_value = object2_info.get('distance', 0) - object1_info.get('distance', 0)
            
            # Step 4: Generate Yes/No options
            options = ["Yes", "No"]
            correct_answer_idx = 0 if comparison_result else 1  # 0 for Yes, 1 for No
            
            question_text += f"\nMulti-view visualization: <image>\n\nOptions:\n"
            for idx, option in enumerate(options):
                question_text += f"({chr(65+idx)}) {option}\n"
            
            # Step 5: Create question dictionary
            question_dict = {
                "question_id": question_id,
                "question_type": ObjectComparisonGenerator.QUESTION_TYPE.value,
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
                        "comparison_type": question_type,
                        "object1_id": object1_info['object_id'],
                        "object1_label": object1_info['label'],
                        "object1_camera": object1_info['camera'],
                        "object1_track_id": object1_info['track_id'],
                        "object1_value": object1_info.get(question_type, 0),
                        "object2_id": object2_info['object_id'],
                        "object2_label": object2_info['label'],
                        "object2_camera": object2_info['camera'],
                        "object2_track_id": object2_info['track_id'],
                        "object2_value": object2_info.get(question_type, 0),
                        "comparison_result": comparison_result
                    }
                }
            }
            
            return question_dict
            
        except Exception as e:
            logger.error(f"Error generating object comparison question: {e}", exc_info=True)
            return None
    
    @staticmethod
    def _find_object_pair(
        annotation: Dict[str, Any],
        views: Dict[str, Any],
        source: str,
        question_type: str
    ) -> Optional[Tuple[Dict[str, Any], Dict[str, Any], bool]]:
        """
        Find two different objects in different camera views for comparison.
        
        Args:
            annotation: Annotation data with 2D/3D bboxes and track IDs
            views: Camera views dictionary
            source: Data source name
            question_type: 'height' or 'distance'
            
        Returns:
            Tuple of (object1_info, object2_info, comparison_result) or None
            comparison_result: True if object1 > object2, False otherwise
        """
        bbox_2d = annotation.get('bbox_2d', [])
        track_2d = annotation.get('track_2d', [])
        track_3d = annotation.get('track_3d', [])
        label_3d = annotation.get('label_3d', [])
        bbox_3d = annotation.get('bbox_3d', [])
        
        if not bbox_2d or not track_2d or not track_3d or not label_3d or not bbox_3d:
            logger.warning("Missing annotation data for object comparison")
            return None
        
        source_lower = source.lower() if source else None
        camera_order = ObjectComparisonGenerator.CAMERA_ORDER.get(source_lower, [])
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
                
                # Calculate distance from ego vehicle (at origin)
                distance = math.sqrt(x**2 + y**2 + z**2)
                
                object_info = {
                    'object_id': obj_idx + 1,
                    'track_id': track_id,
                    'label': label_3d[obj_idx] if obj_idx < len(label_3d) else "unknown",
                    'bbox_3d': bbox,
                    'camera': cam_name,
                    'bbox_idx_in_camera': bbox_idx_in_cam,
                    'height': height,
                    'distance': distance,
                    'x': x,
                    'y': y,
                    'z': z
                }
                
                all_objects.append(object_info)
        
        if len(all_objects) < 2:
            logger.warning("Not enough objects for comparison")
            return None
        
        # Try to find two objects in different cameras with different track_ids
        random.shuffle(all_objects)
        
        for i, obj1 in enumerate(all_objects):
            for obj2 in all_objects[i+1:]:
                # Must be different objects
                if obj1['track_id'] == obj2['track_id']:
                    continue
                
                # Prefer different cameras (but not strictly required)
                if obj1['camera'] == obj2['camera']:
                    continue
                
                # Get comparison value based on question type
                if question_type == 'height':
                    value1 = obj1['height']
                    value2 = obj2['height']
                    # Only compare if heights are significantly different (> 0.5m)
                    if abs(value1 - value2) < 0.5:
                        continue
                else:  # distance
                    value1 = obj1['distance']
                    value2 = obj2['distance']
                    # Only compare if distances are significantly different (> 2m)
                    if abs(value1 - value2) < 2.0:
                        continue
                
                # Determine comparison result
                comparison_result = value1 > value2
                
                logger.info(f"Selected objects for {question_type} comparison: "
                           f"obj1={obj1['track_id']} ({value1:.2f}), "
                           f"obj2={obj2['track_id']} ({value2:.2f}), "
                           f"result={comparison_result}")
                
                return obj1, obj2, comparison_result
        
        logger.warning("Could not find suitable object pair for comparison")
        return None