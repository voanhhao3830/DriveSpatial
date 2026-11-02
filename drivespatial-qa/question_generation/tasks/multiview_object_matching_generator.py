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

logger = logging.getLogger(__name__)


class MultiviewObjectMatchingGenerator:
    """
    Generates multiview object matching questions.
    
    Question format:
    "Given the current driving images, are Object-1 and Object-2 the same object?
    Options: A. Yes; B. No"
    """
    
    QUESTION_TYPE = RelationQuestionType.MULTI_VIEW_OBJECT_MATCHING
    
    QUESTION_TEMPLATE = """Given the current driving images. Are Object-1 and Object-2 the same object?
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
        Generate a multiview object matching question.
        
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
            
            # Step 1: Find object pair (same or different objects in different views)
            object_selection = MultiviewObjectMatchingGenerator._find_object_pair(
                annotation, views, source
            )
            
            if not object_selection:
                logger.warning("No valid object pair found for multiview matching")
                return None
            
            object1_info, object2_info, are_same = object_selection
            
            # Create save directories
            question_folder = Path(save_path) / "relation" / MultiviewObjectMatchingGenerator.QUESTION_TYPE.value
            images_folder = question_folder / "images"
            images_folder.mkdir(parents=True, exist_ok=True)
            
            # Generate unique ID for this question
            question_id = str(uuid4())
            
            # Step 2: Visualize only the two objects with different colors and labels
            # Always use different colors and IDs (1 and 2) regardless of whether they're same/different
            # This doesn't give away the answer
            # Use camera + bbox_idx to uniquely identify each appearance
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
            question_text = MultiviewObjectMatchingGenerator.QUESTION_TEMPLATE.format(
            )
            
            # Step 4: Generate Yes/No options
            options = ["Yes", "No"]
            correct_answer_idx = 0 if are_same else 1  # 0 for Yes, 1 for No
            
            question_text += f"\nMulti-view visualization: <image>\n\nOptions:\n"
            for idx, option in enumerate(options):
                question_text += f"({chr(65+idx)}) {option}\n"
            
            # Step 5: Create question dictionary
            question_dict = {
                "question_id": question_id,
                "question_type": MultiviewObjectMatchingGenerator.QUESTION_TYPE.value,
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
                        "object2_id": object2_info['object_id'],
                        "object2_label": object2_info['label'],
                        "object2_camera": object2_info['camera'],
                        "object2_track_id": object2_info['track_id'],
                        "are_same": are_same
                    }
                }
            }
            
            return question_dict
            
        except Exception as e:
            logger.error(f"Error generating multiview object matching question: {e}", exc_info=True)
            return None
    
    @staticmethod
    def _find_object_pair(
        annotation: Dict[str, Any],
        views: Dict[str, Any],
        source: str
    ) -> Optional[Tuple[Dict[str, Any], Dict[str, Any], bool]]:
        """
        Find two objects in different camera views.
        They may or may not be the same physical object.
        
        Strategy:
        - 75% chance: Select same object appearing in 2 different views (if available)
        - 25% chance: Select 2 different objects in different views
        
        Args:
            annotation: Annotation data with 2D/3D bboxes and track IDs
            views: Camera views dictionary
            source: Data source name
            
        Returns:
            Tuple of (object1_info, object2_info, are_same) or None
        """
        bbox_2d = annotation.get('bbox_2d', [])
        track_2d = annotation.get('track_2d', [])
        track_3d = annotation.get('track_3d', [])
        label_3d = annotation.get('label_3d', [])
        bbox_3d = annotation.get('bbox_3d', [])
        
        if not bbox_2d or not track_2d or not track_3d or not label_3d:
            logger.warning("Missing annotation data for multiview matching")
            return None
        
        source_lower = source.lower() if source else None
        camera_order = MultiviewObjectMatchingGenerator.CAMERA_ORDER.get(source_lower, [])
        original_camera_order = list(views.keys())
        
        # Build mapping: track_id -> list of (camera_name, object_index_in_track_3d, bbox_index_in_camera)
        track_to_cameras: Dict[str, List[Tuple[str, int, int]]] = {}
        
        for cam_idx, cam_name in enumerate(original_camera_order):
            if cam_idx >= len(track_2d):
                continue
            
            tracks_in_camera = track_2d[cam_idx]
            
            for bbox_idx_in_cam, track_id in enumerate(tracks_in_camera):
                if track_id not in track_3d:
                    continue
                
                obj_idx = track_3d.index(track_id)
                
                if track_id not in track_to_cameras:
                    track_to_cameras[track_id] = []
                # Store camera name, object index in track_3d, and bbox index within this camera
                track_to_cameras[track_id].append((cam_name, obj_idx, bbox_idx_in_cam))
        
        # Find objects that appear in multiple cameras (for "same object" questions)
        multi_view_objects = {track_id: cameras for track_id, cameras in track_to_cameras.items() if len(cameras) >= 2}
        
        # Find objects that appear in single camera (for "different object" questions)
        single_view_objects = {track_id: cameras for track_id, cameras in track_to_cameras.items() if len(cameras) == 1}
        
        # Decide whether to create same/different question
        # Give 75% probability to same-object questions (objects in multiple views)
        create_same_question = random.random() < 0.6 and len(multi_view_objects) > 0
        
        if create_same_question:
            # Select same object in 2 different views
            selected_track = random.choice(list(multi_view_objects.keys()))
            cameras = multi_view_objects[selected_track]
            
            # Select 2 different cameras
            if len(cameras) < 2:
                logger.warning("Not enough cameras for same object question")
                return None
            
            cam1, obj_idx1, bbox_idx1 = random.choice(cameras)
            remaining_cameras = [c for c in cameras if c[0] != cam1]
            cam2, obj_idx2, bbox_idx2 = random.choice(remaining_cameras)
            
            # Both have same track_id but different camera views
            object1_info = {
                'object_id': obj_idx1 + 1,
                'track_id': selected_track,
                'label': label_3d[obj_idx1] if obj_idx1 < len(label_3d) else "unknown",
                'bbox_3d': bbox_3d[obj_idx1] if obj_idx1 < len(bbox_3d) else None,
                'camera': cam1,
                'bbox_idx_in_camera': bbox_idx1
            }
            
            object2_info = {
                'object_id': obj_idx2 + 1,
                'track_id': selected_track,
                'label': label_3d[obj_idx2] if obj_idx2 < len(label_3d) else "unknown",
                'bbox_3d': bbox_3d[obj_idx2] if obj_idx2 < len(bbox_3d) else None,
                'camera': cam2,
                'bbox_idx_in_camera': bbox_idx2
            }
            
            are_same = True
            logger.info(f"Selected same object: track_id={selected_track}, cameras={cam1}, {cam2}")
            
        else:
            # Select 2 different objects in different views
            if len(track_to_cameras) < 2:
                logger.warning("Not enough objects for different object question")
                return None
            
            # Select 2 different track_ids
            track_ids = list(track_to_cameras.keys())
            random.shuffle(track_ids)
            
            track1 = track_ids[0]
            cameras1 = track_to_cameras[track1]
            cam1, obj_idx1, bbox_idx1 = random.choice(cameras1)
            
            # Find track2 that doesn't appear in cam1
            track2 = None
            bbox_idx2 = None
            for tid in track_ids[1:]:
                cameras2 = track_to_cameras[tid]
                # Check if this track doesn't appear in cam1
                if not any(c[0] == cam1 for c in cameras2):
                    track2 = tid
                    cam2, obj_idx2, bbox_idx2 = random.choice(cameras2)
                    break
            
            if not track2:
                logger.warning("Could not find objects in different cameras")
                return None
            
            object1_info = {
                'object_id': obj_idx1 + 1,
                'track_id': track1,
                'label': label_3d[obj_idx1] if obj_idx1 < len(label_3d) else "unknown",
                'bbox_3d': bbox_3d[obj_idx1] if obj_idx1 < len(bbox_3d) else None,
                'camera': cam1,
                'bbox_idx_in_camera': bbox_idx1
            }
            
            object2_info = {
                'object_id': obj_idx2 + 1,
                'track_id': track2,
                'label': label_3d[obj_idx2] if obj_idx2 < len(label_3d) else "unknown",
                'bbox_3d': bbox_3d[obj_idx2] if obj_idx2 < len(bbox_3d) else None,
                'camera': cam2,
                'bbox_idx_in_camera': bbox_idx2
            }
            
            are_same = False
            logger.info(f"Selected different objects: track1={track1}, track2={track2}, cameras={cam1}, {cam2}")
        
        return object1_info, object2_info, are_same
