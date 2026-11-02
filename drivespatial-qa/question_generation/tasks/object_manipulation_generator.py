import os
import json
import re
import random
import logging
import numpy as np
from uuid import uuid4
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

from sources.db.sql import SQLiteController
from sources.question_generation.utils.constants import CAMERA_ORDER
from sources.question_generation.utils import save_frames_to_folder
from sources.question_generation.enum import CausualityReasoningQuestionType

logger = logging.getLogger(__name__)

def invert_matrix_egopose_numpy(egopose: np.ndarray) -> np.ndarray:
    """
    Compute the inverse transformation of a 4x4 egopose numpy matrix.
    
    Args:
        egopose: 4x4 transformation matrix
        
    Returns:
        4x4 inverse transformation matrix
    """
    try:
        inverse_matrix = np.zeros((4, 4), dtype=np.float32)
        rotation = egopose[:3, :3]
        translation = egopose[:3, 3]
        inverse_matrix[:3, :3] = rotation.T
        inverse_matrix[:3, 3] = -np.dot(rotation.T, translation)
        inverse_matrix[3, 3] = 1.0
        return inverse_matrix
    except Exception as e:
        logger.error(f"Error inverting egopose matrix: {e}", exc_info=True)
        raise


class ObjectManipulationGenerator:
    """
    Generates trajectory prediction questions from video sequences.
    
    Shows a video context (all frames except the last) and asks about the trajectory
    of a specific object in the next 0.5 seconds given a velocity.
    
    Question format:
    "Given current scene. What if Object-X continues its trajectory with velocity: Y km/h,
    what would be the trajectory of it in the next 0.5 seconds?"
    
    Answer types:
    - Camera transition (e.g., "Disappear from Camera Front Left then appear in Camera Rear Left")
    - Relative position to another object (e.g., "Moving on the left side of Object-Y")
    - Camera appearance (e.g., "Appear in Camera Front")
    """
    
    QUESTION_TYPE = CausualityReasoningQuestionType.OBJECT_MANIPULATION
    
    # Near object selection parameters
    NEAR_OBJECT_THRESHOLD = 20.0  # meters
    NEAR_OBJECT_PROBABILITY = 0.8  # 80% chance to select from near objects
    
    QUESTION_TEMPLATE = """Given this video sequence, what if Object-{object_id} rotates {rotation_angle} degrees (note: current ego front camera is at +90 degrees) and continues with velocity: {velocity} km/h, what would be the trajectory of it in the next 0.5 seconds?
"""
    
    # Velocity options in km/h
    VELOCITY_OPTIONS = [5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    
    # Rotation angle options in degrees (relative to current direction)
    # Positive angles are counter-clockwise, negative are clockwise
    ROTATION_OPTIONS = [-90, -45, -30, -15, 0, 15, 30, 45, 90, 180]
    
    @staticmethod
    def _get_object_distance_from_ego(
        bbox_3d: List[float]
    ) -> float:
        """
        Calculate distance of an object from ego vehicle using bbox_3d.
        
        Args:
            bbox_3d: 3D bounding box [x, y, z, ...]
            
        Returns:
            Distance in meters, or float('inf') if invalid
        """
        if not bbox_3d or len(bbox_3d) < 3:
            return float('inf')
        
        x, y, z = bbox_3d[0], bbox_3d[1], bbox_3d[2]
        return np.sqrt(x**2 + y**2 + z**2)
    
    @staticmethod
    def _weighted_choice(
        candidates: List[Dict[str, Any]],
        near_threshold: float = 10.0,
        near_probability: float = 0.8
    ) -> Optional[Dict[str, Any]]:
        """
        Select a candidate with 80% probability from near objects (< threshold from ego).
        
        Args:
            candidates: List of candidate dictionaries with current_position
            near_threshold: Distance threshold for near objects (default: 10m)
            near_probability: Probability to select from near objects (default: 0.8)
            
        Returns:
            Selected candidate or None if empty
        """
        if not candidates:
            return None
        
        if len(candidates) == 1:
            return candidates[0]
        
        # Separate candidates into near and far groups
        near_candidates = []
        far_candidates = []
        for candidate in candidates:
            position = candidate.get('current_position', None)
            if position is not None:
                distance = np.linalg.norm(position[:2])  # Distance in XY plane
            else:
                distance = float('inf')
            if distance < near_threshold:
                near_candidates.append(candidate)
            else:
                far_candidates.append(candidate)
        
        # Select with probability-based approach
        pick_near = random.random() < near_probability
        
        if pick_near and near_candidates:
            return random.choice(near_candidates)
        elif far_candidates:
            return random.choice(far_candidates)
        elif near_candidates:
            return random.choice(near_candidates)
        else:
            return random.choice(candidates)
    
    @staticmethod
    def generate_question(
        video_sequence: List[Dict[str, Any]],
        db_path: str,
        save_path: str,
        num_options: int = 4
    ) -> Optional[Dict[str, Any]]:
        """
        Generate a trajectory prediction question from a video sequence.
        
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
                logger.warning("Video sequence too short for trajectory prediction (need at least 2 frames)")
                return None
            
            # Context frames are all except the last one
            context_frames = video_sequence[:-1]
            last_frame = video_sequence[-1]
            
            # Extract metadata
            metadata = context_frames[0].get("metadata", {})
            source = metadata.get("source")
            scene_id = context_frames[0].get("scene_id")
            
            # Get the last context frame's record_id to query next frame
            last_context_record_id = context_frames[-1].get("id")
            
            # Step 1: Query next frame
            next_frame = ObjectManipulationGenerator._query_next_frame(
                db_path=db_path,
                scene_id=scene_id,
                source=source,
                last_record_id=last_context_record_id
            )
            
            # If no next frame found, use the last frame from video_sequence
            if not next_frame:
                logger.info("No next frame in database, using last frame from video sequence")
                next_frame = last_frame
            
            # Step 2: Select an object with velocity for prediction
            selected_object = ObjectManipulationGenerator._select_object_for_prediction(
                current_frame=context_frames[-1],
                next_frame=next_frame,
                context_frames=context_frames
            )
            
            if not selected_object:
                logger.warning("No suitable object found for trajectory prediction")
                return None
            
            # Step 3: Select a velocity and rotation angle for the question
            velocity_kmh = random.choice(ObjectManipulationGenerator.VELOCITY_OPTIONS)
            rotation_angle = random.choice(ObjectManipulationGenerator.ROTATION_OPTIONS)
            
            # Step 4: Simulate trajectory with the given velocity and rotation
            simulated_position = ObjectManipulationGenerator._simulate_trajectory(
                current_position=selected_object['current_position'],
                current_velocity=selected_object['current_velocity'],
                given_velocity_kmh=velocity_kmh,
                rotation_angle=rotation_angle,
                time_delta=0.5  # 0.5 seconds
            )
            # Step 5: Determine trajectory description
            correct_answer = ObjectManipulationGenerator._determine_trajectory_description(
                track_id=selected_object['track_id'],
                current_frame=context_frames[-1],
                next_frame=next_frame,
                simulated_position=simulated_position,
                source=source,
                current_position=selected_object['current_position']
            )
            
            if not correct_answer:
                logger.warning("Could not determine trajectory description")
                return None
            
            # Create save directories
            question_folder = Path(save_path) / "causality" / ObjectManipulationGenerator.QUESTION_TYPE.value
            images_folder = question_folder / "images"
            images_folder.mkdir(parents=True, exist_ok=True)
            
            # Generate unique ID for this question
            question_id = str(uuid4())
            
            # Step 6: Save frames to folder (only context frames)
            video_folder = images_folder / question_id
            
            # Highlight the selected object
            colors = [(255, 0, 0)]  # Blue for the main object
            
            highlight_tracks = [{
                'track_id': selected_object['track_id'],
                'color': colors[0]
            }]
            
            frame_paths = save_frames_to_folder(
                frame_sequence=context_frames,
                output_folder=str(video_folder),
                source=source,
                show_object_ids=True,
                highlight_objects=highlight_tracks,
                show_only_highlighted=True  # Only show the objects mentioned in the question
            )
            
            # Step 7: Generate distractor options
            options, correct_idx = ObjectManipulationGenerator._generate_trajectory_options(
                correct_answer=correct_answer,
                current_frame=context_frames[-1],
                next_frame=next_frame,
                selected_object=selected_object,
                source=source,
                num_options=num_options
            )
            
            # Step 8: Build question text
            question_text = ObjectManipulationGenerator.QUESTION_TEMPLATE.format(
                object_id=selected_object['object_id'],
                velocity=velocity_kmh,
                rotation_angle=rotation_angle
            )
            question_text += f"\nVideo sequence (context): <video>\n\nOptions:\n"
            for idx, option in enumerate(options):
                question_text += f"({chr(65+idx)}) {option}\n"
            
            # Step 9: Create question dictionary
            question_dict = {
                "question_id": question_id,
                "question_type": ObjectManipulationGenerator.QUESTION_TYPE.value,
                "question": question_text,
                "videos": [str(video_folder)],
                "frame_paths": frame_paths,
                "options": options,
                "correct_answer": correct_idx,
                "metadata": {
                    "scene_id": scene_id,
                    "context_frame_ids": [f.get("frame_id") for f in context_frames],
                    "next_frame_id": next_frame.get("frame_id"),
                    "source": source,
                    "context_record_ids": [f.get("id") for f in context_frames],
                    "next_record_id": next_frame.get("id"),
                    "reasoning_chain": {
                        "selected_object": {
                            "track_id": selected_object['track_id'],
                            "object_id": selected_object['object_id'],
                            "label": selected_object['label'],
                            "current_position": selected_object['current_position'].tolist(),
                            "current_velocity": selected_object['current_velocity'].tolist(),
                        },
                        "given_velocity_kmh": velocity_kmh,
                        "rotation_angle_degrees": rotation_angle,
                        "simulated_position": simulated_position.tolist(),
                        "correct_trajectory": correct_answer
                    }
                }
            }
            
            return question_dict
            
        except Exception as e:
            logger.error(f"Error generating trajectory prediction question: {e}", exc_info=True)
            return None
    
    @staticmethod
    def _query_next_frame(
        db_path: str,
        scene_id: str,
        source: str,
        last_record_id: int
    ) -> Optional[Dict[str, Any]]:
        """
        Query the next frame from the database.
        
        Uses timestamp for proper temporal ordering.
        
        Args:
            db_path: Path to SQLite database
            scene_id: Scene identifier
            source: Dataset source
            last_record_id: The last record_id in the context
            
        Returns:
            Next frame dictionary or None if not found
        """
        try:
            db = SQLiteController(db_path=db_path)
            
            # Query frames from the same scene with id greater than last_record_id
            query = """
                SELECT * FROM av_data 
                WHERE scene_id = ? 
                AND source = ?
                AND id > ?
            """
            
            with db._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(query, (scene_id, source, last_record_id))
                rows = cursor.fetchall()
                
                if not rows:
                    return None
                
                # Parse JSON fields and extract timestamps
                frames = []
                json_fields = ['views', 'lidar', 'annotation', 'calibration',
                               'spatial_relations', 'internal_interactions',
                               'external_interactions', 'metadata']
                
                for row in rows:
                    frame = dict(row)
                    for field in json_fields:
                        if frame.get(field):
                            try:
                                frame[field] = json.loads(frame[field])
                            except json.JSONDecodeError:
                                pass
                    
                    # Extract timestamp from calibration for sorting
                    calibration = frame.get('calibration', {})
                    img_timestamp = calibration.get('img_timestamp', [])
                    
                    # Use first camera timestamp, or record ID as fallback
                    if img_timestamp and len(img_timestamp) > 0:
                        timestamp = img_timestamp[0]
                    else:
                        timestamp = frame.get('id', 0)
                    
                    frame['_timestamp'] = timestamp
                    frames.append(frame)
                
                # Sort frames by timestamp and return the first (next frame)
                frames.sort(key=lambda f: f['_timestamp'])
                
                return frames[0] if frames else None
            
        except Exception as e:
            logger.error(f"Error querying next frame: {e}", exc_info=True)
            return None
    
    @staticmethod
    def _select_object_for_prediction(
        current_frame: Dict[str, Any],
        next_frame: Dict[str, Any],
        context_frames: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """
        Select an object suitable for trajectory prediction.
        
        Object must:
        - Be visible in current frame
        - Exist in both current and next frame
        - Not be stopped (check internal_interactions)
        
        Args:
            current_frame: Current frame data
            next_frame: Next frame data
            context_frames: All context frames
            
        Returns:
            Dictionary with object info or None
        """
        try:
            from sources.data_collection.enum import IntrinsicInteraction
            
            current_annotation = current_frame.get('annotation', {})
            next_annotation = next_frame.get('annotation', {})
            
            track_3d_current = current_annotation.get('track_3d', [])
            track_3d_next = next_annotation.get('track_3d', [])
            bbox_3d_current = current_annotation.get('bbox_3d', [])
            label_3d_current = current_annotation.get('label_3d', [])
            velocity_current = current_annotation.get('velocity', [])
            
            # Use FIRST context frame's track_3d for consistent object IDs (matching visualization)
            first_frame = context_frames[0] if context_frames else current_frame
            first_track_3d = first_frame.get('annotation', {}).get('track_3d', [])
            
            # Get visible track IDs in FIRST frame's camera views (for ID consistency)
            first_frame_track_2d = first_frame.get('annotation', {}).get('track_2d', [])
            visible_in_first_frame = set()
            for camera_tracks in first_frame_track_2d:
                if isinstance(camera_tracks, list):
                    visible_in_first_frame.update(camera_tracks)
            
            # Only include objects visible in first frame's cameras
            track_to_first_frame_id = {}
            for idx, tid in enumerate(first_track_3d):
                if tid != '00000' and tid in visible_in_first_frame:
                    track_to_first_frame_id[tid] = idx + 1
            
            if len(bbox_3d_current) == 0:
                return None
            
            # Get internal interactions to check if objects are stopped
            internal_interactions = current_frame.get('internal_interactions', [])
            
            # Build map of track_id -> (actions, details)
            track_to_interaction_data = {}
            for interaction_data in internal_interactions:
                track_id = interaction_data.get('track_id')
                actions = interaction_data.get('actions', [])
                details = interaction_data.get('details', {})
                if track_id:
                    track_to_interaction_data[track_id] = {
                        'actions': actions,
                        'details': details
                    }
            
            # Get visible track IDs in current frame's camera views
            visible_track_ids = set()
            track_2d = current_frame.get('annotation', {}).get('track_2d', [])
            for camera_tracks in track_2d:
                if isinstance(camera_tracks, list):
                    visible_track_ids.update(camera_tracks)
            
            # Find objects that are not stopped and exist in both frames
            candidates = []
            for idx, track_id in enumerate(track_3d_current):
                # Skip ego
                if track_id == '00000':
                    continue
                
                # Must be visible
                if track_id not in visible_track_ids:
                    continue
                
                # Must exist in next frame
                if track_id not in track_3d_next:
                    continue
                
                # Check if object is stopped using internal_interactions
                interaction_data = track_to_interaction_data.get(track_id, {})
                actions = interaction_data.get('actions', [])
                details = interaction_data.get('details', {})
                
                is_stopped = False
                for action_str in actions:
                    # Actions might have "IntrinsicInteraction." prefix
                    clean_action = action_str.replace('IntrinsicInteraction.', '').upper()
                    if clean_action == 'STOPPED':
                        is_stopped = True
                        break
                # Skip stopped objects
                if is_stopped:
                    continue
                
                # Get velocity from internal_interactions details first, fallback to annotation
                velocity_vec = np.array([0.0, 0.0])
                
                # Try to get velocity from internal_interactions details
                if 'velocity_x' in details and 'velocity_y' in details:
                    velocity_vec = np.array([
                        float(details['velocity_x']),
                        float(details['velocity_y'])
                    ])
                # Fallback to annotation velocity if available
                elif velocity_current is not None and len(velocity_current) > idx:
                    velocity_vec = np.array(velocity_current[idx][:2])  # vx, vy
                
                # Use first frame's object_id for consistency with visualization
                # Only include objects that exist in first frame
                if track_id not in track_to_first_frame_id:
                    continue
                    
                candidates.append({
                    'track_id': track_id,
                    'object_id': track_to_first_frame_id[track_id],
                    'label': label_3d_current[idx] if idx < len(label_3d_current) else 'unknown',
                    'current_position': np.array(bbox_3d_current[idx][:3]),  # x, y, z
                    'current_velocity': velocity_vec,
                    'bbox_idx_current': idx
                })
            
            if not candidates:
                return None
            
            # Select a candidate with weighted preference for near objects (10m)
            return ObjectManipulationGenerator._weighted_choice(
                candidates=candidates,
                near_threshold=ObjectManipulationGenerator.NEAR_OBJECT_THRESHOLD,
                near_probability=ObjectManipulationGenerator.NEAR_OBJECT_PROBABILITY
            )
            
        except Exception as e:
            logger.error(f"Error selecting object for prediction: {e}", exc_info=True)
            return None
    
    @staticmethod
    def _simulate_trajectory(
        current_position: np.ndarray,
        current_velocity: np.ndarray,
        given_velocity_kmh: float,
        rotation_angle: float,
        time_delta: float
    ) -> np.ndarray:
        """
        Simulate object trajectory with given velocity and rotation.
        
        Args:
            current_position: Current 3D position [x, y, z]
            current_velocity: Current velocity [vx, vy] in m/s
            given_velocity_kmh: Given velocity magnitude in km/h
            rotation_angle: Rotation angle in degrees (counter-clockwise, relative to current direction)
            time_delta: Time to simulate in seconds
            
        Returns:
            Simulated 3D position after time_delta
        """
        # Convert km/h to m/s
        given_velocity_ms = given_velocity_kmh / 3.6
        
        # Get direction from current velocity
        current_velocity_magnitude = np.linalg.norm(current_velocity)
        if current_velocity_magnitude > 0.01:
            direction = current_velocity / current_velocity_magnitude
        else:
            # Default to moving forward (positive x)
            direction = np.array([1.0, 0.0])
        
        # Apply rotation to direction
        # Convert angle to radians (counter-clockwise)
        angle_rad = np.radians(rotation_angle)
        cos_angle = np.cos(angle_rad)
        sin_angle = np.sin(angle_rad)
        
        # 2D rotation matrix
        rotation_matrix = np.array([
            [cos_angle, -sin_angle],
            [sin_angle, cos_angle]
        ])
        
        # Rotate direction
        rotated_direction = rotation_matrix @ direction
        
        # Calculate displacement
        displacement = rotated_direction * given_velocity_ms * time_delta
        
        # Add displacement to current position (x, y only, keep z the same)
        simulated_position = current_position.copy()
        simulated_position[0] += displacement[0]
        simulated_position[1] += displacement[1]
        
        return simulated_position
    
    @staticmethod
    def _determine_trajectory_description(
        track_id: str,
        current_frame: Dict[str, Any],
        next_frame: Dict[str, Any],
        simulated_position: np.ndarray,
        source: str,
        current_position: np.ndarray
    ) -> Optional[str]:
        """
        Determine the trajectory description based on simulated position.
        
        Args:
            track_id: Track ID of the object
            current_frame: Current frame data
            next_frame: Next frame data
            simulated_position: Simulated 3D position
            source: Dataset source
            
        Returns:
            Trajectory description string or None
        """
        try:
            # Get camera information
            calibration = current_frame.get('calibration', {})
            
            # Check which cameras the object appears in after simulation
            projected_cameras = ObjectManipulationGenerator._project_to_cameras(
                position_3d=simulated_position,
                calibration=calibration,
                source=source,
                track_id=track_id,
                current_frame= current_frame,
                current_position=current_position
            )
            
            # Get current cameras
            current_cameras = ObjectManipulationGenerator._get_object_cameras(
                track_id=track_id,
                frame=current_frame
            )
            
            # Determine trajectory type
            if len(projected_cameras) == 0:
                return "Disappears from all cameras"
            
            # Check for camera transition
            if current_cameras and projected_cameras:
                disappeared_from = [cam for cam in current_cameras if cam not in projected_cameras]
                appeared_in = [cam for cam in projected_cameras if cam not in current_cameras]
                
                if disappeared_from and appeared_in:
                    return f"Disappear from {disappeared_from[0]} then appear in {appeared_in[0]}"
                elif appeared_in:
                    return f"Appear in {appeared_in[0]}"
            
            # Check relative position to another object
            next_annotation = next_frame.get('annotation', {})
            track_3d_next = next_annotation.get('track_3d', [])
            bbox_3d_next = next_annotation.get('bbox_3d', [])
            
            if len(bbox_3d_next) > 0:
                # Find a nearby object
                nearby_object = ObjectManipulationGenerator._find_nearby_object(
                    simulated_position=simulated_position,
                    track_id=track_id,
                    track_3d=track_3d_next,
                    bbox_3d=bbox_3d_next
                )
                
                if nearby_object:
                    relative_pos = ObjectManipulationGenerator._get_relative_position(
                        simulated_position, nearby_object['position']
                    )
                    return f"Moving on the {relative_pos} side of Object-{nearby_object['object_id']}"
            
            # Default: just mention the camera
            if projected_cameras:
                return f"Moving in {projected_cameras[0]}"
            
            return None
            
        except Exception as e:
            logger.error(f"Error determining trajectory description: {e}", exc_info=True)
            return None
    
    @staticmethod
    def _project_to_cameras(
        position_3d: np.ndarray,
        calibration: Dict[str, Any],
        source: str,
        track_id: str,
        current_frame: Dict[str, Any],
        current_position: np.ndarray
    ) -> List[str]:
        """
        Project 3D position to cameras and determine which cameras it appears in.
        
        Uses camera extrinsics to determine camera FOV and visibility.
        
        Args:
            position_3d: 3D position [x, y, z] in LiDAR coordinates
            calibration: Calibration data with extrinsics and intrinsics
            source: Dataset source
            
        Returns:
            List of camera names where the object is visible
        """
        visible_cameras = []
        
        extrinsics = calibration.get('extrinsics', [])
        intrinsics = calibration.get('intrinsics', [])
        
        # Camera name mapping (this should match the dataset's camera order)
        camera_names = CAMERA_ORDER
        cam_names = camera_names.get(source.lower(), [])

        # Add homogeneous coordinate
        position_homogeneous = np.array([position_3d[0], position_3d[1], position_3d[2], 1.0])
        # transform to ego pose
        for idx, (extrinsic, intrinsic) in enumerate(zip(extrinsics, intrinsics)):
            if idx >= len(cam_names):
                break
            # (intrinsic @ (np.array(extrinsic)  @ (rotation_matrix @ position_homogeneous_2.T).T))).tolist()
            # Transform to camera coordinates
            # if source == 'waymo' and idx != 0:
            #     point_cam = (T_front_cam_to_ref @ extrinsic )  @ position_homogeneous
            # else:
            point_cam = (extrinsic  @ position_homogeneous)
            # ipdb> ((intrinsic @ T_front_cam_to_ref @ extrinsic  @ invert_matrix_egopose_numpy(Rz) @ position_homogeneous_2 )/ 5.97717584e+01).tolist()
            # if track_id == 'Old1LFoduHRZa8jsHrR8qA'and idx == 2:
            #     import ipdb; ipdb.set_trace()
            # Check if point is in front of camera (positive z in camera frame)
            if point_cam[2] > 0:
                # Project to image plane
                point_img = intrinsic @ point_cam
                
                # Normalize by depth
                if point_img[2] != 0:
                    u = point_img[0] / point_img[2]
                    v = point_img[1] / point_img[2]
                    #  ((intrinsic @ (np.array(extrinsic) @ ( position_homogeneous_2))) / 11.717437824609343).tolist()
                    # Get image dimensions (typically 1600x900 for nuScenes, 1920x1280 for Waymo)
                    # Use a reasonable estimate based on intrinsic matrix
                    img_width = intrinsic[0][2] * 2  # cx is typically at half width
                    img_height = intrinsic[1][2] * 2  # cy is typically at half height
                    # Check if projection is within image bounds
                    if 0 <= u <= img_width and 0 <= v <= img_height:
                        # Format camera name nicely
                        cam_name = cam_names[idx].replace('_', ' ').replace('CAM ', 'Camera ').title()
                        visible_cameras.append(cam_name)

        
        return visible_cameras 
    
    @staticmethod
    def _get_object_cameras(
        track_id: str,
        frame: Dict[str, Any]
    ) -> List[str]:
        """
        Get the list of cameras where an object is visible.
        
        Args:
            track_id: Track ID of the object
            frame: Frame data
            
        Returns:
            List of camera names
        """
        cameras = []
        track_2d = frame.get('annotation', {}).get('track_2d', [])
        views = frame.get('views', {})
        camera_names = list(views.keys())
        
        for idx, camera_tracks in enumerate(track_2d):
            if isinstance(camera_tracks, list) and track_id in camera_tracks:
                if idx < len(camera_names):
                    cameras.append(f"Camera {camera_names[idx].replace('_', ' ').title()}")
        
        return cameras
    
    @staticmethod
    def _find_nearby_object(
        simulated_position: np.ndarray,
        track_id: str,
        track_3d: List[str],
        bbox_3d: List
    ) -> Optional[Dict[str, Any]]:
        """
        Find a nearby object (not self) for relative position description.
        
        Args:
            simulated_position: Simulated position of target object
            track_id: Track ID of target object
            track_3d: List of track IDs in frame
            bbox_3d: List of 3D bounding boxes
            
        Returns:
            Dictionary with nearby object info or None
        """
        try:
            min_distance = float('inf')
            nearest_object = None
            
            for idx, tid in enumerate(track_3d):
                # Skip self and ego
                if tid == track_id or tid == '00000':
                    continue
                
                if idx >= len(bbox_3d):
                    continue
                
                other_position = np.array(bbox_3d[idx][:3])
                distance = np.linalg.norm(simulated_position[:2] - other_position[:2])
                
                # Only consider objects within 30 meters
                if distance < 30 and distance < min_distance:
                    min_distance = distance
                    nearest_object = {
                        'track_id': tid,
                        'object_id': idx + 1,
                        'position': other_position
                    }
            
            return nearest_object
            
        except Exception as e:
            logger.error(f"Error finding nearby object: {e}", exc_info=True)
            return None
    
    @staticmethod
    def _get_relative_position(pos1: np.ndarray, pos2: np.ndarray) -> str:
        """
        Get relative position (left/right) based on y-coordinate.
        
        Args:
            pos1: Position of first object [x, y, z]
            pos2: Position of second object [x, y, z]
            
        Returns:
            "left" or "right"
        """
        # In nuScenes-style coordinates: positive y is left
        if pos1[1] > pos2[1]:
            return "left"
        else:
            return "right"
    
    @staticmethod
    def _generate_trajectory_options(
        correct_answer: str,
        current_frame: Dict[str, Any],
        next_frame: Dict[str, Any],
        selected_object: Dict[str, Any],
        source: str,
        num_options: int = 4
    ) -> Tuple[List[str], int]:
        """
        Generate distractor options for trajectory prediction.
        
        Args:
            correct_answer: The correct trajectory description
            current_frame: Current frame data
            next_frame: Next frame data
            selected_object: Selected object info
            source: Dataset source
            num_options: Number of options to generate
            
        Returns:
            Tuple of (options_list, correct_answer_index)
        """
        options = [correct_answer]
        
        # Generate distractor types
        distractor_types = [
            "Disappear from Camera Front Left then appear in Camera Rear Left",
            "Disappear from Camera Front Right then appear in Camera Rear Right",
            "Moving on the left side of Object-{obj_id}",
            "Moving on the right side of Object-{obj_id}",
            "Appear in Camera Front",
            "Appear in Camera Rear",
            "Appear in Camera Side Left",
            "Appear in Camera Side Right",
            "Disappears from all cameras",
            "Moving in Camera Front Left",
            "Moving in Camera Front Right"
        ]
        
        # Get available object IDs for relative position distractors
        next_annotation = next_frame.get('annotation', {})
        track_3d_next = next_annotation.get('track_3d', [])
        available_obj_ids = [idx + 1 for idx, tid in enumerate(track_3d_next) 
                            if tid != selected_object['track_id'] and tid != '00000']
        
        # Generate distractors
        for _ in range(num_options - 1):
            distractor = random.choice(distractor_types)
            
            # Replace placeholder with actual object ID
            if '{obj_id}' in distractor and available_obj_ids:
                obj_id = random.choice(available_obj_ids)
                distractor = distractor.format(obj_id=obj_id)
            elif '{obj_id}' in distractor:
                # Skip this distractor if no objects available
                continue
            
            # Ensure it's different from correct answer and existing options
            if distractor != correct_answer and distractor not in options:
                options.append(distractor)
        
        # If we don't have enough unique distractors, add more generic ones
        while len(options) < num_options:
            generic_distractors = [
                "Continue moving forward in the same camera",
                "Stop and remain stationary",
                "Change direction significantly"
            ]
            distractor = random.choice(generic_distractors)
            if distractor not in options:
                options.append(distractor)
        
        # Shuffle options
        random.shuffle(options)
        correct_idx = options.index(correct_answer)
        
        return options, correct_idx

