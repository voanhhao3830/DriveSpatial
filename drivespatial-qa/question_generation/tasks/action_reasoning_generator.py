import os
import json
import random
import logging
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional, Set
from uuid import uuid4

from sources.question_generation.enum import CausualityReasoningQuestionType
from sources.question_generation.utils import save_frames_to_folder
from sources.db.sql import SQLiteController
from sources.data_collection.enum import IntrinsicInteraction

logger = logging.getLogger(__name__)


class ActionReasoningGenerator:
    """
    Generates action reasoning questions from video sequences.
    
    Shows a video and asks about the action of a specific object.
    Uses internal_interactions data from the interaction extractor.
    Selects 3 objects that appear in different camera views.
    
    Question format:
    "Given this video sequence, what is the action of Object-X?"
    
    Answer: One of the IntrinsicInteraction types (e.g., "moving forward", "turning left")
    """
    
    QUESTION_TYPE = CausualityReasoningQuestionType.ACTION_REASONING
    
    # Near object selection parameters
    NEAR_OBJECT_THRESHOLD = 20.0  # meters
    NEAR_OBJECT_PROBABILITY = 0.8  # 80% chance to select from near objects
    
    QUESTION_TEMPLATE = """Given this video sequence, what are the actions of the three highlighted objects?
"""
    
    # Map action enums to natural language
    ACTION_DESCRIPTIONS = {
        IntrinsicInteraction.STOPPED: "stopped",
        IntrinsicInteraction.MOVING_FORWARD: "moving forward",
        IntrinsicInteraction.MOVING_BACKWARD: "moving backward",
        IntrinsicInteraction.U_TURN: "making a U-turn",
        IntrinsicInteraction.TURN_LEFT: "turning left",
        IntrinsicInteraction.TURN_RIGHT: "turning right",
        IntrinsicInteraction.LANE_CHANGE_LEFT: "changing lane to the left",
        IntrinsicInteraction.LANE_CHANGE_RIGHT: "changing lane to the right",
        IntrinsicInteraction.ACCELERATE: "accelerating",
        IntrinsicInteraction.DECELERATE: "decelerating",
    }
    
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
    def _weighted_choice_multi(
        candidates: List[Dict[str, Any]],
        annotation: Dict[str, Any],
        near_threshold: float = 10.0,
        near_probability: float = 0.8,
        max_objects: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Select multiple candidates with 80% probability from near objects (< threshold from ego).
        
        Args:
            candidates: List of candidate dictionaries with track_id
            annotation: Annotation data
            near_threshold: Distance threshold for near objects (default: 10m)
            near_probability: Probability to select from near objects (default: 0.8)
            max_objects: Maximum number of objects to select
            
        Returns:
            Selected list of candidates
        """
        if not candidates:
            return []
        
        if len(candidates) <= max_objects:
            return candidates
        
        # Separate candidates into near and far groups
        near_candidates = []
        far_candidates = []
        for candidate in candidates:
            track_id = candidate.get('track_id', '')
            distance = ActionReasoningGenerator._get_object_distance_from_ego(track_id, annotation)
            if distance < near_threshold:
                near_candidates.append(candidate)
            else:
                far_candidates.append(candidate)
        
        # Select candidates with probability-based approach
        selected = []
        while len(selected) < max_objects:
            # Decide whether to pick from near objects (80% chance)
            pick_near = random.random() < near_probability
            
            if pick_near and near_candidates:
                chosen = random.choice(near_candidates)
                selected.append(chosen)
                near_candidates.remove(chosen)
            elif far_candidates:
                chosen = random.choice(far_candidates)
                selected.append(chosen)
                far_candidates.remove(chosen)
            elif near_candidates:
                chosen = random.choice(near_candidates)
                selected.append(chosen)
                near_candidates.remove(chosen)
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
        Generate an action reasoning question from a video sequence.
        
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
                logger.warning("Video sequence too short for action reasoning (need at least 3 frames)")
                return None
            
            # Extract metadata from first frame
            metadata = video_sequence[0].get("metadata", {})
            source = metadata.get("source")
            scene_id = video_sequence[0].get("scene_id")
            
            # Step 1: Find exactly 3 objects with clear actions in different cameras
            objects_data = ActionReasoningGenerator._find_objects_in_different_cameras(video_sequence)
            
            if not objects_data or len(objects_data) < 3:
                logger.warning(f"Need 3 objects in different cameras, found only {len(objects_data) if objects_data else 0}")
                return None
            
            # Use the first 3 objects
            selected_objects = objects_data[:3]
            
            # Create save directories
            question_folder = Path(save_path) / "causality" / ActionReasoningGenerator.QUESTION_TYPE.value
            images_folder = question_folder / "images"
            images_folder.mkdir(parents=True, exist_ok=True)
            
            # Generate unique ID for this question
            question_id = str(uuid4())
            
            # Step 2: Save frames to folder with all 3 objects highlighted
            video_folder = images_folder / question_id
            
            # Use different colors for each object
            colors = [
                (255, 0, 0),      # Blue
                (255, 0, 255),    # Magenta
                (0, 255, 255),    # Cyan
            ]
            
            highlight_tracks = []
            for idx, obj in enumerate(selected_objects):
                highlight_tracks.append({
                    'track_id': obj['track_id'],
                    'color': colors[idx],
                    'label': f'Object-{idx + 1}'
                })
            
            frame_paths = save_frames_to_folder(
                frame_sequence=video_sequence,
                output_folder=str(video_folder),
                source=source,
                show_object_ids=True,
                highlight_objects=highlight_tracks,
                show_only_highlighted=True,  # Only show the 3 objects
                hide_object_ids_for_highlighted=True  # Only show custom labels (Object-1, etc.)
            )
            
            # Step 3: Generate correct answer (all 3 objects)
            correct_answer = ActionReasoningGenerator._format_actions_string(selected_objects)
            
            # Step 4: Generate distractor options
            options, correct_idx = ActionReasoningGenerator._generate_multi_action_options(
                selected_objects, correct_answer, num_options
            )
            
            # Step 5: Build question text
            question_text = ActionReasoningGenerator.QUESTION_TEMPLATE
            question_text += f"\nVideo sequence: <video>\n\nOptions:\n"
            for idx, option in enumerate(options):
                question_text += f"({chr(65+idx)}) {option}\n"
            
            # Step 6: Create question dictionary
            question_dict = {
                "question_id": question_id,
                "question_type": ActionReasoningGenerator.QUESTION_TYPE.value,
                "question": question_text,
                "videos": [str(video_folder)],
                "frame_paths": frame_paths,
                "options": options,
                "correct_answer": correct_idx,
                "metadata": {
                    "scene_id": scene_id,
                    "frame_ids": [f.get("frame_id") for f in video_sequence],
                    "source": source,
                    "record_ids": [f.get("id") for f in video_sequence],
                    "reasoning_chain": {
                        "num_frames": len(video_sequence),
                        "objects": [
                            {
                                "object_track": obj['track_id'],
                                "object_id": obj['object_id'],
                                "action": obj['action'].value,
                                "cameras": list(obj['cameras'])
                            }
                            for obj in selected_objects
                        ]
                    }
                }
            }
            
            return question_dict
            
        except Exception as e:
            logger.error(f"Error generating action reasoning question: {e}", exc_info=True)
            return None
    
    @staticmethod
    def _find_objects_in_different_cameras(
        video_sequence: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Find up to 3 objects with clear actions that appear in different cameras.
        
        Args:
            video_sequence: List of frame dictionaries
            
        Returns:
            List of object dictionaries with track_id, action, and camera info
        """
        # Collect all objects with actions across the video
        object_actions = {}  # track_id -> {action: count, label: str, cameras: set}
        
        for frame in video_sequence:
            internal_interactions = frame.get("internal_interactions", [])
            track_3d = frame.get('annotation', {}).get('track_3d', [])
            track_2d = frame.get('annotation', {}).get('track_2d', [])
            
            # Build track_id to cameras mapping
            track_to_cameras = {}
            for cam_idx, camera_tracks in enumerate(track_2d):
                if isinstance(camera_tracks, list):
                    for track in camera_tracks:
                        if track not in track_to_cameras:
                            track_to_cameras[track] = set()
                        track_to_cameras[track].add(cam_idx)
            
            for internal_data in internal_interactions:
                track_id = internal_data.get('track_id')
                label = internal_data.get('label', 'unknown')
                actions_list = internal_data.get('actions', [])
                
                # Skip ego vehicle
                if track_id == '00000':
                    continue
                
                # Must be in track_3d and visible in cameras
                if track_id not in track_3d or track_id not in track_to_cameras:
                    continue
                
                if track_id not in object_actions:
                    object_actions[track_id] = {
                        'actions': {},
                        'label': label,
                        'cameras': set()
                    }
                
                # Update cameras where this object appears
                object_actions[track_id]['cameras'].update(track_to_cameras[track_id])
                
                # Count actions
                for action_str in actions_list:
                    try:
                        clean_action = action_str.replace('IntrinsicInteraction.', '').upper()
                        action_enum = IntrinsicInteraction[clean_action]
                        
                        # Skip "stopped" actions - not interesting
                        if action_enum == IntrinsicInteraction.STOPPED:
                            continue
                        
                        if action_enum not in object_actions[track_id]['actions']:
                            object_actions[track_id]['actions'][action_enum] = 0
                        object_actions[track_id]['actions'][action_enum] += 1
                    except (KeyError, AttributeError):
                        continue
        
        # Filter objects with actions and select up to 3 from different cameras
        candidates = []
        for track_id, data in object_actions.items():
            if data['actions']:
                # Get most frequent action
                most_common_action = max(data['actions'].items(), key=lambda x: x[1])
                action, count = most_common_action
                
                candidates.append({
                    'track_id': track_id,
                    'action': action,
                    'label': data['label'],
                    'cameras': data['cameras'],
                    'action_count': count
                })
        
        if not candidates:
            return []
        
        # Select up to 3 objects from different cameras
        selected_objects = ActionReasoningGenerator._select_objects_from_different_cameras(
            candidates, video_sequence, max_objects=3
        )
        
        return selected_objects
    
    @staticmethod
    def _select_objects_from_different_cameras(
        candidates: List[Dict[str, Any]],
        video_sequence: List[Dict[str, Any]],
        max_objects: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Select objects that appear in different cameras, preferring near objects.
        
        Args:
            candidates: List of candidate objects with camera info
            video_sequence: Video sequence to get object IDs
            max_objects: Maximum number of objects to select
            
        Returns:
            List of selected objects with object_id added
        """
        # Get annotation from first frame for distance calculation and ID consistency
        first_frame = video_sequence[0] if video_sequence else {}
        first_frame_annotation = first_frame.get('annotation', {})
        first_track_3d = first_frame_annotation.get('track_3d', [])
        
        # Get visible track IDs in first frame's camera views (for ID consistency)
        first_frame_track_2d = first_frame_annotation.get('track_2d', [])
        visible_in_first_frame = set()
        for camera_tracks in first_frame_track_2d:
            if isinstance(camera_tracks, list):
                visible_in_first_frame.update(camera_tracks)
        
        # Build mapping from track_id to object_id using first frame (only for visible objects)
        track_to_first_frame_id = {}
        for idx, tid in enumerate(first_track_3d):
            if tid != '00000' and tid in visible_in_first_frame:
                track_to_first_frame_id[tid] = idx + 1
        
        # First pass: collect candidates that don't overlap with used cameras
        # We'll use weighted selection among these
        used_cameras: Set[int] = set()
        non_overlapping = []
        overlapping = []
        
        for candidate in candidates:
            candidate_cameras = candidate['cameras']
            track_id = candidate['track_id']
            
            # Only include objects visible in first frame's cameras
            if track_id not in track_to_first_frame_id:
                continue
            
            # Get object_id from first frame for consistency
            candidate['object_id'] = track_to_first_frame_id[track_id]
            
            if not candidate_cameras.intersection(used_cameras):
                non_overlapping.append(candidate)
            else:
                overlapping.append(candidate)
        
        # Use weighted selection preferring near objects (10m) from non-overlapping
        selected = ActionReasoningGenerator._weighted_choice_multi(
            candidates=non_overlapping,
            annotation=first_frame_annotation,
            near_threshold=ActionReasoningGenerator.NEAR_OBJECT_THRESHOLD,
            near_probability=ActionReasoningGenerator.NEAR_OBJECT_PROBABILITY,
            max_objects=max_objects
        )
        
        # Update used cameras
        for obj in selected:
            used_cameras.update(obj.get('cameras', set()))
        
        # If we don't have enough objects, add from overlapping with weighted selection
        if len(selected) < max_objects:
            remaining_needed = max_objects - len(selected)
            remaining_candidates = [c for c in overlapping if c not in selected]
            
            additional = ActionReasoningGenerator._weighted_choice_multi(
                candidates=remaining_candidates,
                annotation=first_frame_annotation,
                near_threshold=ActionReasoningGenerator.NEAR_OBJECT_THRESHOLD,
                near_probability=ActionReasoningGenerator.NEAR_OBJECT_PROBABILITY,
                max_objects=remaining_needed
            )
            selected.extend(additional)
        
        return selected
    
    @staticmethod
    def _format_actions_string(objects: List[Dict[str, Any]]) -> str:
        """
        Format actions of multiple objects into a string.
        
        Args:
            objects: List of object dictionaries with action info
            
        Returns:
            Formatted string like "Object-1: moving forward, Object-2: turning left, Object-3: accelerating"
        """
        action_parts = []
        for idx, obj in enumerate(objects):
            action_desc = ActionReasoningGenerator.ACTION_DESCRIPTIONS.get(
                obj['action'],
                obj['action'].value.replace('_', ' ')
            )
            action_parts.append(f"Object-{idx + 1}: {action_desc}")
        
        return ", ".join(action_parts)
    
    @staticmethod
    def _generate_multi_action_options(
        objects: List[Dict[str, Any]],
        correct_answer: str,
        num_options: int = 4
    ) -> Tuple[List[str], int]:
        """
        Generate distractor options for multi-object action question.
        
        Args:
            objects: List of 3 object dictionaries
            correct_answer: The correct formatted answer string
            num_options: Number of options to generate
            
        Returns:
            Tuple of (options_list, correct_answer_index)
        """
        options = [correct_answer]
        
        # Get all action types
        all_actions = list(ActionReasoningGenerator.ACTION_DESCRIPTIONS.keys())
        
        # Generate distractors by modifying actions
        for _ in range(num_options - 1):
            # Create a modified version
            distractor_objects = []
            for idx, obj in enumerate(objects):
                # Randomly decide whether to change this object's action
                if random.random() < 0.5:  # 50% chance to modify
                    # Pick a different action
                    other_actions = [a for a in all_actions if a != obj['action']]
                    new_action = random.choice(other_actions)
                    action_desc = ActionReasoningGenerator.ACTION_DESCRIPTIONS.get(
                        new_action,
                        new_action.value.replace('_', ' ')
                    )
                else:
                    # Keep original
                    action_desc = ActionReasoningGenerator.ACTION_DESCRIPTIONS.get(
                        obj['action'],
                        obj['action'].value.replace('_', ' ')
                    )
                
                distractor_objects.append(f"Object-{idx + 1}: {action_desc}")
            
            distractor = ", ".join(distractor_objects)
            
            # Ensure it's different from correct answer and existing options
            if distractor != correct_answer and distractor not in options:
                options.append(distractor)
        
        # If we don't have enough unique distractors, generate more variations
        while len(options) < num_options:
            # Shuffle the order of objects or mix actions more
            shuffled_actions = []
            for idx, obj in enumerate(objects):
                random_action = random.choice(all_actions)
                action_desc = ActionReasoningGenerator.ACTION_DESCRIPTIONS.get(
                    random_action,
                    random_action.value.replace('_', ' ')
                )
                shuffled_actions.append(f"Object-{idx + 1}: {action_desc}")
            
            distractor = ", ".join(shuffled_actions)
            if distractor not in options:
                options.append(distractor)
        
        # Shuffle options
        random.shuffle(options)
        correct_idx = options.index(correct_answer)
        
        return options, correct_idx
    
    @staticmethod
    def _generate_action_options(
        correct_action: IntrinsicInteraction,
        correct_answer: str,
        num_options: int = 4
    ) -> Tuple[List[str], int]:
        """
        Generate distractor options for action question.
        
        Args:
            correct_action: The correct action enum
            correct_answer: The correct formatted answer string
            num_options: Number of options to generate
            
        Returns:
            Tuple of (options_list, correct_answer_index)
        """
        options = [correct_answer]
        
        # Get all other action types as distractors
        all_actions = list(ActionReasoningGenerator.ACTION_DESCRIPTIONS.keys())
        other_actions = [a for a in all_actions if a != correct_action]
        
        # Shuffle and select distractors
        random.shuffle(other_actions)
        
        for action in other_actions[:num_options - 1]:
            distractor = ActionReasoningGenerator.ACTION_DESCRIPTIONS.get(
                action,
                action.value.replace('_', ' ')
            )
            options.append(distractor)
        
        # If we don't have enough unique actions, add generic options
        while len(options) < num_options:
            options.append("no action")
        
        # Shuffle options
        random.shuffle(options)
        correct_idx = options.index(correct_answer)
        
        return options, correct_idx
