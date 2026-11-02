import os
import json
import random
import logging
import numpy as np
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


class ObjectDisappearanceGenerator:
    QUESTION_TYPE = CausualityReasoningQuestionType.OBJECT_DISAPPEARANCE
    
    # Near object selection parameters
    NEAR_OBJECT_THRESHOLD = 20.0  # meters
    NEAR_OBJECT_PROBABILITY = 0.8  # 80% chance to select from near objects
    
    QUESTION_TEMPLATE = """Given this video sequence, which object was present earlier but is no longer visible, and why?
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
        return np.sqrt(x**2 + y**2 + z**2)
    
    @staticmethod
    def _weighted_choice(
        items: List[Dict[str, Any]],
        annotation: Dict[str, Any],
        near_threshold: float = 10.0,
        near_probability: float = 0.8
    ) -> Optional[Dict[str, Any]]:
        """
        Select an item with 80% probability from near objects (< threshold from ego).
        
        Args:
            items: List of object dictionaries with track_id
            annotation: Annotation data
            near_threshold: Distance threshold for near objects (default: 10m)
            near_probability: Probability to select from near objects (default: 0.8)
            
        Returns:
            Selected item or None if empty
        """
        if not items:
            return None
        
        if len(items) == 1:
            return items[0]
        
        # Separate items into near and far groups
        near_items = []
        far_items = []
        for item in items:
            track_id = item.get('track_id', '')
            distance = ObjectDisappearanceGenerator._get_object_distance_from_ego(track_id, annotation)
            if distance < near_threshold:
                near_items.append(item)
            else:
                far_items.append(item)
        
        # Select with probability-based approach
        pick_near = random.random() < near_probability
        
        if pick_near and near_items:
            return random.choice(near_items)
        elif far_items:
            return random.choice(far_items)
        elif near_items:
            return random.choice(near_items)
        else:
            return random.choice(items)
    
    @staticmethod
    def generate_question(
        video_sequence: List[Dict[str, Any]],
        db_path: str,
        save_path: str,
        num_options: int = 4
    ) -> Optional[Dict[str, Any]]:
        """
        Generate an object disappearance question from a video sequence.
        
        Args:
            video_sequence: List of consecutive frame data dictionaries
            db_path: Path to the database
            save_path: Path to save questions and videos
            num_options: Number of options to generate (default: 4)
            
        Returns:
            question: Question dictionary or None if generation fails
        """
        try:
            if not video_sequence or len(video_sequence) < 3:
                logger.warning("Video sequence too short for object disappearance (need at least 3 frames)")
                return None
            
            # Use all frames as context
            context_frames = video_sequence
            
            # Extract metadata
            metadata = context_frames[0].get("metadata", {})
            source = metadata.get("source")
            scene_id = context_frames[0].get("scene_id")
            
            # Step 1: Find objects that disappeared
            disappeared_object = ObjectDisappearanceGenerator._find_disappeared_object(
                context_frames=context_frames
            )

            if not disappeared_object:
                logger.warning("No disappeared objects found")
                return None

            # Use object_id from the first context frame (only frame with IDs shown)
            first_frame = context_frames[0]
            track_3d_first = first_frame.get("annotation", {}).get("track_3d", [])
            if disappeared_object["track_id"] in track_3d_first:
                idx = track_3d_first.index(disappeared_object["track_id"])
                disappeared_object["object_id"] = idx + 1
                label_3d = first_frame.get("annotation", {}).get("label_3d", [])
                if idx < len(label_3d):
                    disappeared_object["label"] = label_3d[idx]

            # Step 2: Determine why the object disappeared (passed by or occluded)
            reason = ObjectDisappearanceGenerator._determine_disappearance_reason(
                disappeared_object=disappeared_object,
                last_frame=context_frames[-1]
            )

            # Create save directories
            question_folder = Path(save_path) / "causality" / ObjectDisappearanceGenerator.QUESTION_TYPE.value
            images_folder = question_folder / "images"
            images_folder.mkdir(parents=True, exist_ok=True)

            # Generate unique ID for this question
            question_id = str(uuid4())

            # Step 3: Save frames — only first context frame shows object IDs; others do not
            video_folder = images_folder / question_id

            frame_paths = save_frames_to_folder(
                frame_sequence=context_frames,
                output_folder=str(video_folder),
                source=source,
                show_object_ids=True,
                highlight_objects=None,
                show_only_highlighted=False,
                show_object_ids_only_on_frames=[0],
            )
            
            # Step 4: Generate correct answer
            correct_answer = ObjectDisappearanceGenerator._format_answer(
                disappeared_object, reason
            )
            
            # Step 5: Generate distractor options
            options, correct_idx = ObjectDisappearanceGenerator._generate_disappearance_options(
                disappeared_object=disappeared_object,
                reason=reason,
                correct_answer=correct_answer,
                context_frames=context_frames,
                num_options=num_options
            )
            
            # Step 6: Build question text
            question_text = ObjectDisappearanceGenerator.QUESTION_TEMPLATE
            question_text += f"\nVideo sequence: <video>\n\nOptions:\n"
            for idx, option in enumerate(options):
                question_text += f"({chr(65+idx)}) {option}\n"
            
            # Step 7: Create question dictionary
            question_dict = {
                "question_id": question_id,
                "question_type": ObjectDisappearanceGenerator.QUESTION_TYPE.value,
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
                    "reasoning_chain": {
                        "disappeared_object": {
                            "track_id": disappeared_object['track_id'],
                            "object_id": disappeared_object['object_id'],
                            "label": disappeared_object['label'],
                            "last_frame_idx": disappeared_object['last_frame_idx'],
                            "disappearance_category": disappeared_object.get('disappearance_category', 'unknown')
                        },
                        "reason": {
                            "type": reason.get('type'),
                            "visibility_level": reason.get('visibility_level'),
                            "visibility_percent": reason.get('visibility_percent'),
                            "visibility_token": reason.get('visibility_token'),
                            "occluder_track_id": reason.get('occluder_track_id'),
                            "occluder_label": reason.get('occluder_label')
                        }
                    }
                }
            }
            
            return question_dict
            
        except Exception as e:
            logger.error(f"Error generating object disappearance question: {e}", exc_info=True)
            return None
    
    @staticmethod
    def _find_disappeared_object(
        context_frames: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """
        Find an object that was present earlier but is no longer visible.
        
        An object is considered "disappeared" if:
        1. Not in last frame's track_3d (passed by/out of view), OR
        2. In last frame but heavily occluded (visibility level 1 or 2, ≤60% visible)
        
        Args:
            context_frames: List of context frame data
            
        Returns:
            Dictionary with disappeared object info or None
        """
        try:
            last_frame = context_frames[-1]
            last_track_3d = set(last_frame.get('annotation', {}).get('track_3d', []))
            last_label_3d = last_frame.get('annotation', {}).get('label_3d', [])
            last_track_3d.discard('00000')  # Remove ego
            
            # Build map of track_id to visibility level in last frame
            last_frame_visibility = {}
            occlusions = last_frame.get('occlusions', [])
            for occ in occlusions:
                target_obj = occ.get('target_object', {})
                target_track_id = target_obj.get('track_id')
                visibility = occ.get('visibility', {})
                
                # Extract the visibility level (handle both dict and int formats)
                if isinstance(visibility, dict):
                    visibility_level = visibility.get('level', 4)
                elif isinstance(visibility, int):
                    visibility_level = visibility
                else:
                    visibility_level = 4  # Default to fully visible
                
                if target_track_id and target_track_id != '00000':
                    # Store the lowest visibility level if multiple occlusions
                    if target_track_id not in last_frame_visibility:
                        last_frame_visibility[target_track_id] = visibility_level
                    else:
                        last_frame_visibility[target_track_id] = min(
                            last_frame_visibility[target_track_id], 
                            visibility_level
                        )
            
            # Track objects across all frames
            track_appearances = {}
            for frame_idx, frame in enumerate(context_frames):
                track_3d = frame.get('annotation', {}).get('track_3d', [])
                label_3d = frame.get('annotation', {}).get('label_3d', [])
                
                for idx, track_id in enumerate(track_3d):
                    if track_id == '00000':  # Skip ego
                        continue
                    
                    if track_id not in track_appearances:
                        track_appearances[track_id] = {
                            'track_id': track_id,
                            'object_id': idx + 1,
                            'label': label_3d[idx] if idx < len(label_3d) else 'unknown',
                            'first_frame_idx': frame_idx,
                            'last_frame_idx': frame_idx,
                            'frame_count': 0
                        }
                    
                    track_appearances[track_id]['last_frame_idx'] = frame_idx
                    track_appearances[track_id]['frame_count'] += 1
                    
                    # Only update label (keep original object_id from first appearance for consistency)
                    track_appearances[track_id]['label'] = label_3d[idx] if idx < len(label_3d) else 'unknown'
            
            # Find objects that disappeared - separate into two categories.
            # Only consider objects that appear in the first context frame (frame 0), so the
            # object_id we show in the answer is the one visible in the only frame with IDs.
            passed_by_objects = []  # Objects that left the scene
            occluded_objects = []   # Objects that are heavily occluded
            
            # Get track_ids visible in first frame's camera views (track_2d)
            first_frame = context_frames[0]
            track_2d_first = first_frame.get('annotation', {}).get('track_2d', [])
            visible_in_first_frame = set()
            for camera_tracks in track_2d_first:
                if isinstance(camera_tracks, list):
                    visible_in_first_frame.update(camera_tracks)

            for track_id, info in track_appearances.items():
                # Must have appeared in at least 2 frames, in the first context frame (frame 0),
                # AND visible in the first frame's camera views
                if info['frame_count'] < 2 or info['first_frame_idx'] != 0:
                    continue
                if track_id not in visible_in_first_frame:
                    continue

                # Case 1: Object not in last frame (passed by)
                if track_id not in last_track_3d:
                    # Must have disappeared recently (not more than 2 frames ago)
                    if len(context_frames) - info['last_frame_idx'] <= 3:
                        passed_by_objects.append(info)

                # Case 2: Object in last frame but heavily occluded (visibility level 1 or 2)
                elif track_id in last_frame_visibility:
                    visibility_level = last_frame_visibility[track_id]
                    if visibility_level <= 2:  # ≤60% visible
                        # Keep object_id from first frame (frame 0) for answer consistency
                        occluded_objects.append(info)
            
            # 50/50 split: randomly choose between passed_by and occluded
            # Only use categories that have objects available
            available_categories = []
            if passed_by_objects:
                available_categories.append(('passed_by', passed_by_objects))
            if occluded_objects:
                available_categories.append(('occluded', occluded_objects))
            
            if not available_categories:
                return None
            
            # Randomly select category (50/50 if both available)
            category_name, objects_list = random.choice(available_categories)
            
            # Use weighted selection preferring near objects (10m)
            first_frame_annotation = context_frames[0].get('annotation', {})
            selected_object = ObjectDisappearanceGenerator._weighted_choice(
                items=objects_list,
                annotation=first_frame_annotation,
                near_threshold=ObjectDisappearanceGenerator.NEAR_OBJECT_THRESHOLD,
                near_probability=ObjectDisappearanceGenerator.NEAR_OBJECT_PROBABILITY
            )
            
            if not selected_object:
                return None
            
            # Mark the category for metadata
            selected_object['disappearance_category'] = category_name
            
            return selected_object
            
        except Exception as e:
            logger.error(f"Error finding disappeared object: {e}", exc_info=True)
            return None
    
    @staticmethod
    def _determine_disappearance_reason(
        disappeared_object: Dict[str, Any],
        last_frame: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Determine why the object disappeared (passed by or occluded).
        
        Uses visibility levels to determine if an object is occluded:
        - Level 1 (0-40% visible): Heavily occluded -> disappeared
        - Level 2 (41-60% visible): Moderately occluded -> disappeared
        - Level 3 (61-80% visible): Slightly occluded -> still visible
        - Level 4 (81-100% visible): Mostly visible -> still visible
        
        Args:
            disappeared_object: Info about the disappeared object
            last_frame: Last frame data
            
        Returns:
            Dictionary with reason info
        """
        try:
            track_id = disappeared_object['track_id']
            occlusions = last_frame.get('occlusions', [])
            
            # Check if object is occluded
            for occ in occlusions:
                target_obj = occ.get('target_object', {})
                if target_obj.get('track_id') == track_id:
                    # Object has occlusion information
                    source_obj = occ.get('source_object', {})
                    visibility = occ.get('visibility', {})
                    visibility = visibility.get('level', 4)
                    
                    # Check visibility level (1-4)
                    # Level 1 or 2 means significantly occluded (<=60% visible)
                    if visibility <= 1:
                        return {
                            'type': 'occluded',
                            'occluder_track_id': source_obj.get('track_id'),
                            'occluder_label': source_obj.get('label'),
                            'visibility_level': visibility,
                        }
            
            # Object passed by (moved out of view)
            return {
                'type': 'passed_by'
            }
            
        except Exception as e:
            logger.error(f"Error determining disappearance reason: {e}", exc_info=True)
            return {'type': 'passed_by'}
    
    @staticmethod
    def _format_answer(
        disappeared_object: Dict[str, Any],
        reason: Dict[str, Any]
    ) -> str:
        """
        Format the answer string.
        
        Args:
            disappeared_object: Info about the disappeared object
            reason: Reason for disappearance
            
        Returns:
            Formatted answer string
        """
        obj_id = disappeared_object['object_id']
        
        if reason['type'] == 'occluded':
            # Find occluder object ID (need to look it up in the frame)
            return f"Object-{obj_id}: occluded"
        else:
            return f"Object-{obj_id}: passed by"
    
    @staticmethod
    def _generate_disappearance_options(
        disappeared_object: Dict[str, Any],
        reason: Dict[str, Any],
        correct_answer: str,
        context_frames: List[Dict[str, Any]],
        num_options: int = 4
    ) -> Tuple[List[str], int]:
        """
        Generate distractor options for object disappearance question.
        
        Distractors include:
        - Objects that never appeared
        - Objects that are still visible in the last frame
        - Wrong reasons for the disappeared object
        
        Args:
            disappeared_object: Info about the disappeared object
            reason: Correct reason for disappearance
            correct_answer: The correct answer string
            context_frames: List of context frames
            num_options: Number of options to generate
            
        Returns:
            Tuple of (options_list, correct_answer_index)
        """
        options = [correct_answer]
        
        # Get all objects that appeared and objects still visible
        last_frame = context_frames[-1]
        last_track_3d = last_frame.get('annotation', {}).get('track_3d', [])
        last_label_3d = last_frame.get('annotation', {}).get('label_3d', [])
        
        all_appeared_tracks = set()
        for frame in context_frames:
            track_3d = frame.get('annotation', {}).get('track_3d', [])
            all_appeared_tracks.update(track_3d)
        all_appeared_tracks.discard('00000')
        
        # Objects still visible in last frame
        still_visible = []
        for idx, track_id in enumerate(last_track_3d):
            if track_id != '00000' and track_id != disappeared_object['track_id']:
                still_visible.append({
                    'track_id': track_id,
                    'object_id': idx + 1,
                    'label': last_label_3d[idx] if idx < len(last_label_3d) else 'unknown'
                })
        
        # Generate distractors
        distractor_types = []
        
        # Type 1: Objects that are still visible (with passed by/occluded)
        for obj in still_visible[:3]:
            distractor_types.append(f"Object-{obj['object_id']}: passed by")
            distractor_types.append(f"Object-{obj['object_id']}: occluded")
        
        # Type 2: Wrong reason for the disappeared object
        if reason['type'] == 'occluded':
            distractor_types.append(f"Object-{disappeared_object['object_id']}: passed by")
        else:
            distractor_types.append(f"Object-{disappeared_object['object_id']}: occluded")
        
        # Type 3: Fake object IDs (never appeared)
        max_obj_id = max([idx + 1 for idx, _ in enumerate(last_track_3d)])
        for fake_id in range(max_obj_id + 1, max_obj_id + 5):
            distractor_types.append(f"Object-{fake_id}: passed by")
            distractor_types.append(f"Object-{fake_id}: occluded")
        
        # Sample distractors
        random.shuffle(distractor_types)
        for distractor in distractor_types:
            if distractor != correct_answer and distractor not in options:
                options.append(distractor)
            
            if len(options) >= num_options:
                break
        
        # Ensure we have exactly num_options
        while len(options) < num_options:
            fake_id = random.randint(50, 100)
            fake_reason = random.choice(['passed by', 'occluded'])
            distractor = f"Object-{fake_id}: {fake_reason}"
            if distractor not in options:
                options.append(distractor)
        
        # Shuffle options
        random.shuffle(options)
        correct_idx = options.index(correct_answer)
        
        return options, correct_idx