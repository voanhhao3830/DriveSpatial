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
from sources.data_collection.enum import ExtrinsicInteraction

logger = logging.getLogger(__name__)


class ObjectInteractionGenerator:
    QUESTION_TYPE = CausualityReasoningQuestionType.OBJECT_INTERACTION
    
    QUESTION_TEMPLATE = """Given this video sequence, what is the interaction between Object-1 and Object-2?
"""
    
    # Map interaction enums to natural language
    INTERACTION_DESCRIPTIONS = {
        ExtrinsicInteraction.LEAD: "Object-1 is leading Object-2",
        ExtrinsicInteraction.FOLLOW: "Object-1 is following Object-2",
        ExtrinsicInteraction.OVERTAKE: "Object-1 is overtaking Object-2",
        ExtrinsicInteraction.PASSING: "Object-1 is passing Object-2",
        ExtrinsicInteraction.CO_MOVING: "Object-1 is co-moving with Object-2",
        ExtrinsicInteraction.APPROACHING: "Object-1 is approaching Object-2",
        ExtrinsicInteraction.CROSSING: "Object-1 is crossing Object-2",
        ExtrinsicInteraction.WAITING: "Object-1 is waiting for Object-2",
        ExtrinsicInteraction.YIELDING: "Object-1 is yielding to Object-2",
    }
    
    @staticmethod
    def generate_question(
        video_sequence: List[Dict[str, Any]],
        db_path: str,
        save_path: str,
        num_options: int = 4
    ) -> Optional[Dict[str, Any]]:
        """
        Generate an object interaction question from a video sequence.
        
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
                logger.warning("Video sequence too short for object interaction (need at least 3 frames)")
                return None
            
            # Extract metadata from first frame
            metadata = video_sequence[0].get("metadata", {})
            source = metadata.get("source")
            scene_id = video_sequence[0].get("scene_id")
            
            # Step 1: Find object pair with interaction across the video sequence
            interaction_data = ObjectInteractionGenerator._find_object_interaction(video_sequence)
            
            if not interaction_data:
                logger.warning("No valid object interaction found in video sequence")
                return None
            
            object1_track = interaction_data['object1_track']
            object2_track = interaction_data['object2_track']
            object1_id = interaction_data['object1_id']
            object2_id = interaction_data['object2_id']
            interaction_type = interaction_data['interaction']
            
            # Create save directories
            question_folder = Path(save_path) / "causality" / ObjectInteractionGenerator.QUESTION_TYPE.value
            images_folder = question_folder / "images"
            images_folder.mkdir(parents=True, exist_ok=True)
            
            # Generate unique ID for this question
            question_id = str(uuid4())
            
            # Step 2: Save frames to folder with both objects highlighted
            video_folder = images_folder / question_id
            
            highlight_tracks = [
                {
                    'track_id': object1_track,
                    'color': (255, 0, 0),  # Blue
                    'label': 'Object-1'
                },
                {
                    'track_id': object2_track,
                    'color': (255, 0, 255),  # Magenta
                    'label': 'Object-2'
                }
            ]
            
            frame_paths = save_frames_to_folder(
                frame_sequence=video_sequence,
                output_folder=str(video_folder),
                source=source,
                show_object_ids=True,
                highlight_objects=highlight_tracks,
                show_only_highlighted=True
            )
            
            # Step 3: Generate correct answer
            correct_answer = ObjectInteractionGenerator.INTERACTION_DESCRIPTIONS.get(
                interaction_type,
                f"Object-1 is {interaction_type.value.replace('_', ' ')} Object-2"
            )
            
            # Step 4: Generate distractor options
            options, correct_idx = ObjectInteractionGenerator._generate_interaction_options(
                interaction_type, correct_answer, num_options
            )
            
            # Step 5: Build question text
            question_text = ObjectInteractionGenerator.QUESTION_TEMPLATE
            question_text += f"\nVideo sequence: <video>\n\nOptions:\n"
            for idx, option in enumerate(options):
                question_text += f"({chr(65+idx)}) {option}\n"
            
            # Step 6: Create question dictionary
            question_dict = {
                "question_id": question_id,
                "question_type": ObjectInteractionGenerator.QUESTION_TYPE.value,
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
                        "object1_track": object1_track,
                        "object2_track": object2_track,
                        "object1_id": object1_id,
                        "object2_id": object2_id,
                        "interaction": interaction_type.value
                    }
                }
            }
            
            return question_dict
            
        except Exception as e:
            logger.error(f"Error generating object interaction question: {e}", exc_info=True)
            return None
    
    @staticmethod
    def _find_object_interaction(
        video_sequence: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """
        Find a pair of objects with a clear interaction across the video sequence.
        75% of the time, select objects in different cameras.
        
        Args:
            video_sequence: List of frame dictionaries
            
        Returns:
            Dictionary with object pair and interaction, or None if not found
        """
        # Collect all interactions across the video
        interaction_counts = {}  # (source_track, target_track, interaction) -> count
        
        for frame in video_sequence:
            external_interactions = frame.get("external_interactions", [])
            track_3d = frame.get('annotation', {}).get('track_3d', [])
            track_2d = frame.get('annotation', {}).get('track_2d', [])
            
            # Collect track IDs visible in cameras
            visible_tracks = set()
            for camera_tracks in track_2d:
                if isinstance(camera_tracks, list):
                    visible_tracks.update(camera_tracks)
            
            for ext_data in external_interactions:
                source_obj = ext_data.get('source_object', {})
                target_obj = ext_data.get('target_object', {})
                source_track = source_obj.get('track_id')
                target_track = target_obj.get('track_id')
                interaction_str = ext_data.get('interaction_type', '')
                
                # Skip ego vehicle
                if source_track == '00000' or target_track == '00000':
                    continue
                
                # Both must be in track_3d and visible in cameras
                if (source_track not in track_3d or target_track not in track_3d or
                    source_track not in visible_tracks or target_track not in visible_tracks):
                    continue
                
                if interaction_str:
                    try:
                        interaction_enum = ExtrinsicInteraction[interaction_str.upper()]
                        key = (source_track, target_track, interaction_enum)
                        interaction_counts[key] = interaction_counts.get(key, 0) + 1
                    except (KeyError, AttributeError):
                        continue
        
        if not interaction_counts:
            return None
        
        # Categorize interactions by camera configuration
        different_camera_interactions = []
        same_camera_interactions = []
        
        for (source_track, target_track, interaction), count in interaction_counts.items():
            # Check camera configuration in first frame where both appear
            in_different_cameras = ObjectInteractionGenerator._check_different_cameras(
                video_sequence, source_track, target_track
            )
            
            entry = ((source_track, target_track, interaction), count)
            
            if in_different_cameras:
                different_camera_interactions.append(entry)
            else:
                same_camera_interactions.append(entry)
        
        # 75% probability to select from different cameras
        use_different_cameras = random.random() < 0.75
        
        # Select interaction based on probability
        selected_list = different_camera_interactions if (use_different_cameras and different_camera_interactions) else same_camera_interactions
        
        # If preferred list is empty, use the other
        if not selected_list:
            selected_list = same_camera_interactions if use_different_cameras else different_camera_interactions
        
        if not selected_list:
            return None
        
        # Find the interaction that appears most frequently in the selected list
        best_interaction = max(selected_list, key=lambda x: x[1])
        (source_track, target_track, interaction), count = best_interaction
        
        # Get object IDs from the first frame where both appear
        for frame in video_sequence:
            track_3d = frame.get('annotation', {}).get('track_3d', [])
            if source_track in track_3d and target_track in track_3d:
                object1_id = track_3d.index(source_track) + 1
                object2_id = track_3d.index(target_track) + 1
                
                return {
                    'object1_track': source_track,
                    'object2_track': target_track,
                    'object1_id': object1_id,
                    'object2_id': object2_id,
                    'interaction': interaction,
                    'frame_count': count
                }
        
        return None
    
    @staticmethod
    def _check_different_cameras(
        video_sequence: List[Dict[str, Any]],
        track1: str,
        track2: str
    ) -> bool:
        """
        Check if two objects appear in different cameras in any frame.
        
        Args:
            video_sequence: List of frame dictionaries
            track1: Track ID of first object
            track2: Track ID of second object
            
        Returns:
            True if objects are in different cameras in at least one frame
        """
        for frame in video_sequence:
            track_2d = frame.get('annotation', {}).get('track_2d', [])
            
            # Find which cameras each object appears in
            track1_cameras = set()
            track2_cameras = set()
            
            for cam_idx, camera_tracks in enumerate(track_2d):
                if isinstance(camera_tracks, list):
                    if track1 in camera_tracks:
                        track1_cameras.add(cam_idx)
                    if track2 in camera_tracks:
                        track2_cameras.add(cam_idx)
            
            # Check if they appear in different cameras (no overlap)
            # or if one appears in multiple cameras including different ones
            if track1_cameras and track2_cameras:
                # If they don't share any camera, they're in different cameras
                if not track1_cameras.intersection(track2_cameras):
                    return True
        
        return False
    
    @staticmethod
    def _generate_interaction_options(
        correct_interaction: ExtrinsicInteraction,
        correct_answer: str,
        num_options: int = 4
    ) -> Tuple[List[str], int]:
        """
        Generate distractor options for interaction question.
        
        Args:
            correct_interaction: The correct interaction enum
            correct_answer: The correct formatted answer string
            num_options: Number of options to generate
            
        Returns:
            Tuple of (options_list, correct_answer_index)
        """
        options = [correct_answer]
        
        # Get all other interaction types as distractors
        all_interactions = list(ObjectInteractionGenerator.INTERACTION_DESCRIPTIONS.keys())
        other_interactions = [i for i in all_interactions if i != correct_interaction]
        
        # Shuffle and select distractors
        random.shuffle(other_interactions)
        
        for interaction in other_interactions[:num_options - 1]:
            distractor = ObjectInteractionGenerator.INTERACTION_DESCRIPTIONS.get(
                interaction,
                f"Object-1 is {interaction.value.replace('_', ' ')} Object-2"
            )
            options.append(distractor)
        
        # If we don't have enough unique interactions, add "No interaction"
        while len(options) < num_options:
            options.append("Object-1 and Object-2 have no interaction")
        
        # Shuffle options
        random.shuffle(options)
        correct_idx = options.index(correct_answer)
        
        return options, correct_idx