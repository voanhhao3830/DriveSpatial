import os
import json
import random
import logging
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
from uuid import uuid4

from sources.question_generation.enum import CausualityReasoningQuestionType
from sources.question_generation.utils import save_frames_to_folder
from sources.db.sql import SQLiteController

logger = logging.getLogger(__name__)

# Visibility level definitions (from occlusion_extractor.py)
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


class OcclusionAwarenessGenerator:
    """
    Generates occlusion awareness questions from video sequences.
    
    Shows a video context and asks which objects are occluded in the final frame.
    
    Question format:
    "Given this video sequence, are there any objects occluded in the final frame? What are they?"
    
    Answer format:
    "Object-X, Object-Y" (list of occluded object IDs) or "None" if no objects are occluded
    """
    
    QUESTION_TYPE = CausualityReasoningQuestionType.OCCLUSION_AWARENESS
    
    # Near object selection parameters
    NEAR_OBJECT_THRESHOLD = 20.0  # meters
    NEAR_OBJECT_PROBABILITY = 0.8  # 80% chance to select from near objects
    
    QUESTION_TEMPLATE = """Given this video sequence, are there any objects occluded in the final frame? What are they?
"""
    
    @staticmethod
    def _get_object_distance_from_ego(
        track_id: str,
        annotation: Dict[str, Any]
    ) -> float:
        """
        Calculate distance of an object from ego vehicle.
        
        Args:
            track_id: Object track ID
            annotation: Annotation data with track_3d and bbox_3d
            
        Returns:
            Distance in meters, or float('inf') if not found
        """
        import math
        
        track_3d = annotation.get('track_3d', [])
        bbox_3d = annotation.get('bbox_3d', [])
        
        if track_id not in track_3d:
            return float('inf')
        
        obj_idx = track_3d.index(track_id)
        if obj_idx >= len(bbox_3d):
            return float('inf')
        
        bbox = bbox_3d[obj_idx]
        if not bbox or len(bbox) < 3:
            return float('inf')
        
        x, y, z = bbox[0], bbox[1], bbox[2]
        return math.sqrt(x**2 + y**2 + z**2)
    
    @staticmethod
    def _weighted_choice_objects(
        objects: List[Dict[str, Any]],
        annotation: Dict[str, Any],
        near_threshold: float = 10.0,
        near_probability: float = 0.8,
        max_objects: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Select objects with 80% probability from near objects (< threshold from ego).
        
        Args:
            objects: List of object dictionaries with track_id
            annotation: Annotation data
            near_threshold: Distance threshold for near objects (default: 10m)
            near_probability: Probability to select from near objects (default: 0.8)
            max_objects: Maximum number of objects to select
            
        Returns:
            Selected list of objects
        """
        if not objects:
            return []
        
        if len(objects) <= max_objects:
            return objects
        
        # Separate objects into near and far groups
        near_objects = []
        far_objects = []
        for obj in objects:
            track_id = obj.get('track_id', '')
            distance = OcclusionAwarenessGenerator._get_object_distance_from_ego(track_id, annotation)
            if distance < near_threshold:
                near_objects.append(obj)
            else:
                far_objects.append(obj)
        
        # Select objects with probability-based approach
        selected = []
        while len(selected) < max_objects:
            # Decide whether to pick from near objects (80% chance)
            pick_near = random.random() < near_probability
            
            if pick_near and near_objects:
                chosen = random.choice(near_objects)
                selected.append(chosen)
                near_objects.remove(chosen)
            elif far_objects:
                chosen = random.choice(far_objects)
                selected.append(chosen)
                far_objects.remove(chosen)
            elif near_objects:
                chosen = random.choice(near_objects)
                selected.append(chosen)
                near_objects.remove(chosen)
            else:
                break
        
        return selected
    
    @staticmethod
    def generate_question(
        video_sequence: List[Dict[str, Any]],
        db_path: str,
        save_path: str,
        num_options: int = 4
    ) -> Optional[Dict[str, Any]]:
        """
        Generate an occlusion awareness question from a video sequence.
        
        Args:
            video_sequence: List of consecutive frame data dictionaries
            db_path: Path to the database
            save_path: Path to save questions and videos
            num_options: Number of options to generate (default: 4)
            
        Returns:
            question: Question dictionary or None if generation fails
        """
        try:
            if not video_sequence or len(video_sequence) < 2:
                logger.warning("Video sequence too short for occlusion awareness (need at least 2 frames)")
                return None
            
            # Use all frames as context
            context_frames = video_sequence
            first_frame = context_frames[0]
            last_frame = context_frames[-1]

            # Extract metadata
            metadata = last_frame.get("metadata", {})
            source = metadata.get("source")
            scene_id = last_frame.get("scene_id")

            # Step 1: Find occluded objects in the final frame (by track_id)
            occluded_objects = OcclusionAwarenessGenerator._find_occluded_objects(
                last_frame=last_frame
            )

            if occluded_objects is None:
                logger.warning("Could not determine occluded objects")
                return None

            # Map occluded objects to first-frame object IDs (so answers match IDs shown in first frame)
            occluded_objects = OcclusionAwarenessGenerator._map_to_first_frame_object_ids(
                occluded_objects=occluded_objects,
                first_frame=first_frame
            )
            if occluded_objects is None:
                return None

            # Create save directories
            question_folder = Path(save_path) / "causality" / OcclusionAwarenessGenerator.QUESTION_TYPE.value
            images_folder = question_folder / "images"
            images_folder.mkdir(parents=True, exist_ok=True)

            # Generate unique ID for this question
            question_id = str(uuid4())

            # Step 2: Save frames — only show the occluded objects mentioned in the answer
            video_folder = images_folder / question_id
            
            # Prepare highlight info for occluded objects mentioned in the answer
            # Each object gets a different color for easy identification
            colors = [
                (255, 0, 0),      # Blue
                (255, 0, 255),    # Magenta
                (0, 255, 255),    # Cyan
                (0, 255, 0),      # Green
                (255, 128, 0),    # Orange
                (128, 0, 255),    # Purple
            ]
            
            # Build highlight objects from occluded objects
            highlight_objects = []
            if occluded_objects:
                # Get track_id to object_id mapping from first frame
                first_frame_track_3d = first_frame.get('annotation', {}).get('track_3d', [])
                
                for idx, obj in enumerate(occluded_objects):
                    track_id = obj.get('track_id')
                    object_id = obj.get('object_id')
                    color = colors[idx % len(colors)]
                    
                    highlight_objects.append({
                        'track_id': track_id,
                        'color': color,
                        'object_id': object_id,  # For consistent labeling
                        'label': f'#{object_id}'
                    })

            frame_paths = save_frames_to_folder(
                frame_sequence=context_frames,
                output_folder=str(video_folder),
                source=source,
                show_object_ids=True,
                highlight_objects=highlight_objects if highlight_objects else None,
                show_only_highlighted=True if highlight_objects else False,  # Only show mentioned objects
                show_object_ids_only_on_frames=[0],
            )
            
            # Step 3: Generate correct answer
            correct_answer = OcclusionAwarenessGenerator._format_answer(occluded_objects)
            
            # Step 4: Generate distractor options
            options, correct_idx = OcclusionAwarenessGenerator._generate_occlusion_options(
                occluded_objects=occluded_objects,
                correct_answer=correct_answer,
                last_frame=last_frame,
                num_options=num_options
            )
            
            # Step 5: Build question text
            question_text = OcclusionAwarenessGenerator.QUESTION_TEMPLATE
            question_text += f"\nVideo sequence: <video>\n\nOptions:\n"
            for idx, option in enumerate(options):
                question_text += f"({chr(65+idx)}) {option}\n"
            
            # Step 6: Create question dictionary
            question_dict = {
                "question_id": question_id,
                "question_type": OcclusionAwarenessGenerator.QUESTION_TYPE.value,
                "question": question_text,
                "videos": [str(video_folder)],
                "frame_paths": frame_paths,
                "options": options,
                "correct_answer": correct_idx,
                "metadata": {
                    "scene_id": scene_id,
                    "frame_ids": [f.get("frame_id") for f in context_frames],
                    "source": source,
                    "record_ids": [f.get("id") for f in context_frames],
                    "last_frame_id": last_frame.get("frame_id"),
                    "reasoning_chain": {
                        "occluded_objects": occluded_objects,
                        "num_occluded": len(occluded_objects) if occluded_objects else 0,
                    }
                }
            }
            
            return question_dict
            
        except Exception as e:
            logger.error(f"Error generating occlusion awareness question: {e}", exc_info=True)
            return None
    
    @staticmethod
    def _map_to_first_frame_object_ids(
        occluded_objects: List[Dict[str, Any]],
        first_frame: Dict[str, Any]
    ) -> Optional[List[Dict[str, Any]]]:
        """
        Map occluded object object_ids to the first frame's numbering.
        So "Object-X" in the answer matches the ID shown in the first context frame.
        
        Only includes objects that are visible in the first frame's camera views (track_2d).

        Args:
            occluded_objects: List from _find_occluded_objects (object_id from last frame)
            first_frame: First context frame

        Returns:
            Same list with object_id set to first frame index+1, or None if mapping fails
        """
        try:
            track_3d = first_frame.get("annotation", {}).get("track_3d", [])
            track_2d = first_frame.get("annotation", {}).get("track_2d", [])
            
            # Get all track_ids visible in camera views (track_2d)
            visible_in_cameras = set()
            for camera_tracks in track_2d:
                if isinstance(camera_tracks, list):
                    visible_in_cameras.update(camera_tracks)
            
            track_to_first_frame_id = {}
            for idx, track_id in enumerate(track_3d):
                if track_id == "00000":
                    continue
                # Only include if visible in camera views
                if track_id in visible_in_cameras:
                    track_to_first_frame_id[track_id] = idx + 1

            result = []
            for obj in occluded_objects:
                track_id = obj.get("track_id")
                if track_id not in track_to_first_frame_id:
                    logger.warning(
                        "Occluded track_id %s not found in first frame or not visible in cameras, skipping for answer",
                        track_id
                    )
                    continue
                obj_copy = obj.copy()
                obj_copy["object_id"] = track_to_first_frame_id[track_id]
                result.append(obj_copy)

            return result if result else None
        except Exception as e:
            logger.error(f"Error mapping to first frame object IDs: {e}", exc_info=True)
            return None

    @staticmethod
    def _find_occluded_objects(
        last_frame: Dict[str, Any]
    ) -> Optional[List[Dict[str, Any]]]:
        """
        Find objects that are occluded in the final frame.
        
        An object is considered occluded if its visibility level is 1 or 2 (≤60% visible).
        Returns a maximum of 3 most occluded objects, prioritized by visibility level.
        
        Args:
            last_frame: Last frame data
            
        Returns:
            List of up to 3 occluded object info dictionaries or None
        """
        try:
            annotation = last_frame.get('annotation', {})
            track_3d = annotation.get('track_3d', [])
            label_3d = annotation.get('label_3d', [])
            occlusions = last_frame.get('occlusions', [])
            
            # Build map of track_id to object info
            track_to_obj = {}
            for idx, track_id in enumerate(track_3d):
                if track_id == '00000':  # Skip ego
                    continue
                track_to_obj[track_id] = {
                    'track_id': track_id,
                    'object_id': idx + 1,
                    'label': label_3d[idx] if idx < len(label_3d) else 'unknown'
                }
            
            # Find occluded objects (visibility level 1 or 2)
            occluded_objects = []
            for occ in occlusions:
                target_obj = occ.get('target_object', {})
                target_track_id = target_obj.get('track_id')
                visibility = occ.get('visibility', {})
                
                # Extract the visibility level (handle both dict and int formats)
                if isinstance(visibility, dict):
                    visibility_level = visibility.get('level', 4)
                    visibility_percent = visibility.get('visibility_percent', 100.0)
                elif isinstance(visibility, int):
                    visibility_level = visibility
                    visibility_percent = None
                else:
                    visibility_level = 4
                    visibility_percent = None
                
                # Only include objects with visibility level 1 or 2 (≤60% visible)
                if visibility_level <= 2 and target_track_id in track_to_obj:
                    obj_info = track_to_obj[target_track_id].copy()
                    obj_info['visibility_level'] = visibility_level
                    obj_info['visibility_percent'] = visibility_percent
                    
                    # Only add if not already in list (avoid duplicates)
                    if not any(o['track_id'] == target_track_id for o in occluded_objects):
                        occluded_objects.append(obj_info)
            
            # Sort by visibility level (most occluded first), then by object_id
            occluded_objects.sort(key=lambda x: (x['visibility_level'], x['object_id']))
            
            # Limit to maximum 3 objects, 80% probability for near objects (10m)
            if len(occluded_objects) > 3:
                occluded_objects = OcclusionAwarenessGenerator._weighted_choice_objects(
                    objects=occluded_objects,
                    annotation=annotation,
                    near_threshold=OcclusionAwarenessGenerator.NEAR_OBJECT_THRESHOLD,
                    near_probability=OcclusionAwarenessGenerator.NEAR_OBJECT_PROBABILITY,
                    max_objects=3
                )
            else:
                occluded_objects = occluded_objects[:3]
            
            return occluded_objects
            
        except Exception as e:
            logger.error(f"Error finding occluded objects: {e}", exc_info=True)
            return None
    
    @staticmethod
    def _format_answer(occluded_objects: List[Dict[str, Any]]) -> str:
        """
        Format the answer string.
        
        Args:
            occluded_objects: List of occluded object info (max 3 objects)
            
        Returns:
            Formatted answer string (e.g., "Object-1, Object-3, Object-5" or "None")
        """
        if not occluded_objects:
            return "None"
        
        # Format as "Object-X, Object-Y, Object-Z" (max 3 objects)
        object_ids = [f"Object-{obj['object_id']}" for obj in occluded_objects]
        return ", ".join(object_ids)
    
    @staticmethod
    def _generate_occlusion_options(
        occluded_objects: List[Dict[str, Any]],
        correct_answer: str,
        last_frame: Dict[str, Any],
        num_options: int = 4
    ) -> Tuple[List[str], int]:
        """
        Generate distractor options for occlusion awareness question.
        
        Distractors include:
        - "None" (if there are occluded objects)
        - Subset of occluded objects (missing some)
        - Mix of occluded and non-occluded objects
        - Only non-occluded objects
        - Fake object IDs
        
        Args:
            occluded_objects: List of occluded object info
            correct_answer: The correct answer string
            last_frame: Last frame data
            num_options: Number of options to generate
            
        Returns:
            Tuple of (options_list, correct_answer_index)
        """
        options = [correct_answer]
        
        # Get all objects in the frame
        annotation = last_frame.get('annotation', {})
        track_3d = annotation.get('track_3d', [])
        
        all_object_ids = [idx + 1 for idx, track_id in enumerate(track_3d) if track_id != '00000']
        occluded_object_ids = [obj['object_id'] for obj in occluded_objects]
        non_occluded_object_ids = [obj_id for obj_id in all_object_ids if obj_id not in occluded_object_ids]
        
        distractor_types = []
        
        # Type 1: "None" (if there are occluded objects)
        if occluded_objects:
            distractor_types.append("None")
        
        # Type 2: Subset of occluded objects (missing some)
        if len(occluded_objects) > 1:
            for i in range(len(occluded_objects)):
                subset = [obj for idx, obj in enumerate(occluded_objects) if idx != i]
                if subset:
                    subset_str = ", ".join([f"Object-{obj['object_id']}" for obj in subset])
                    distractor_types.append(subset_str)
        
        # Type 3: Mix of occluded and non-occluded objects
        if occluded_objects and non_occluded_object_ids:
            for non_occ_id in non_occluded_object_ids[:3]:
                mixed = occluded_object_ids[:2] + [non_occ_id]
                mixed.sort()
                mixed_str = ", ".join([f"Object-{obj_id}" for obj_id in mixed])
                distractor_types.append(mixed_str)
        
        # Type 4: Only non-occluded objects
        if non_occluded_object_ids:
            for i in range(min(3, len(non_occluded_object_ids))):
                subset = non_occluded_object_ids[:i+1]
                subset_str = ", ".join([f"Object-{obj_id}" for obj_id in subset])
                distractor_types.append(subset_str)
        
        # Type 5: Fake object IDs (never in the frame)
        max_obj_id = max(all_object_ids) if all_object_ids else 0
        for fake_id in range(max_obj_id + 1, max_obj_id + 5):
            distractor_types.append(f"Object-{fake_id}")
            if occluded_objects:
                fake_mixed = [occluded_objects[0]['object_id'], fake_id]
                fake_mixed_str = ", ".join([f"Object-{obj_id}" for obj_id in fake_mixed])
                distractor_types.append(fake_mixed_str)
        
        # Sample distractors
        random.shuffle(distractor_types)
        for distractor in distractor_types:
            if distractor != correct_answer and distractor not in options:
                options.append(distractor)
            
            if len(options) >= num_options:
                break
        
        # Ensure we have exactly num_options
        while len(options) < num_options:
            # Generate random fake object IDs
            fake_id = random.randint(50, 100)
            distractor = f"Object-{fake_id}"
            if distractor not in options:
                options.append(distractor)
        
        # Shuffle options
        random.shuffle(options)
        correct_idx = options.index(correct_answer)
        
        return options, correct_idx