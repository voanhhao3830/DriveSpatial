import os
import json
import re
import random
import logging
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
from uuid import uuid4

from sources.question_generation.enum import CausualityReasoningQuestionType
from sources.question_generation.utils import save_frames_to_folder
from sources.db.sql import SQLiteController
from sources.data_collection.enum import IntrinsicInteraction, ExtrinsicInteraction

logger = logging.getLogger(__name__)


class EventOrderingGenerator:
    QUESTION_TYPE = CausualityReasoningQuestionType.EVENT_ORDERING
    
    # Near object selection parameters
    NEAR_OBJECT_THRESHOLD = 20.0  # meters
    NEAR_OBJECT_PROBABILITY = 0.8  # 80% chance to select from near objects
    
    QUESTION_TEMPLATE = """Given this video sequence, what is the correct order of events in the next 3 seconds?
"""
    
    # Map interaction enums to natural language
    INTRINSIC_ACTION_DESCRIPTIONS = {
        IntrinsicInteraction.STOPPED: "stopped",
        IntrinsicInteraction.MOVING_FORWARD: "moving forward",
        IntrinsicInteraction.MOVING_BACKWARD: "moving backward",
        IntrinsicInteraction.U_TURN: "making U-turn",
        IntrinsicInteraction.TURN_LEFT: "turning left",
        IntrinsicInteraction.TURN_RIGHT: "turning right",
        IntrinsicInteraction.LANE_CHANGE_LEFT: "changing lane left",
        IntrinsicInteraction.LANE_CHANGE_RIGHT: "changing lane right",
        IntrinsicInteraction.ACCELERATE: "accelerating",
        IntrinsicInteraction.DECELERATE: "decelerating",
    }
    
    EXTRINSIC_INTERACTION_DESCRIPTIONS = {
        ExtrinsicInteraction.LEAD: "leading",
        ExtrinsicInteraction.FOLLOW: "following",
        ExtrinsicInteraction.OVERTAKE: "overtaking",
        ExtrinsicInteraction.PASSING: "passing",
        ExtrinsicInteraction.CO_MOVING: "co-moving with",
        ExtrinsicInteraction.APPROACHING: "approaching",
        ExtrinsicInteraction.CROSSING: "crossing",
        ExtrinsicInteraction.WAITING: "waiting for",
        ExtrinsicInteraction.YIELDING: "yielding to",
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
    def _weighted_choice_events(
        events: List[Dict[str, Any]],
        annotation: Dict[str, Any],
        near_threshold: float = 10.0,
        near_probability: float = 0.8,
        max_events: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Select events with 80% probability from near objects (< threshold from ego).
        
        Args:
            events: List of event dictionaries with track_id
            annotation: Annotation data
            near_threshold: Distance threshold for near objects (default: 10m)
            near_probability: Probability to select from near objects (default: 0.8)
            max_events: Maximum number of events to select
            
        Returns:
            Selected list of events
        """
        import random
        
        if not events:
            return []
        
        if len(events) <= max_events:
            return events
        
        # Separate events into near and far groups
        near_events = []
        far_events = []
        for event in events:
            track_id = event.get('track_id', '')
            distance = EventOrderingGenerator._get_object_distance_from_ego(track_id, annotation)
            if distance < near_threshold:
                near_events.append(event)
            else:
                far_events.append(event)
        
        # Select events with probability-based approach
        selected = []
        while len(selected) < max_events:
            # Decide whether to pick from near objects (80% chance)
            pick_near = random.random() < near_probability
            
            if pick_near and near_events:
                # Pick from near events
                chosen = random.choice(near_events)
                selected.append(chosen)
                near_events.remove(chosen)
            elif far_events:
                # Pick from far events
                chosen = random.choice(far_events)
                selected.append(chosen)
                far_events.remove(chosen)
            elif near_events:
                # No far events left, pick from near
                chosen = random.choice(near_events)
                selected.append(chosen)
                near_events.remove(chosen)
            else:
                # No events left
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
        Generate an event ordering question from a video sequence.
        
        The question shows the video context (all frames except last) and asks to order events 
        in the next 3 seconds. Queries up to 6 future frames (0.5s each) from the database.
        If no future frames are available, uses the last frame from video_sequence.
        
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
                logger.warning("Video sequence too short for event ordering (need at least 2 frames)")
                return None
            
            # Context frames are all except the last one
            context_frames = video_sequence[:-1]
            last_frame = video_sequence[-1]
            
            # Extract metadata
            metadata = context_frames[0].get("metadata", {})
            source = metadata.get("source")
            scene_id = context_frames[0].get("scene_id")
            
            # Get the last context frame's record_id to query future frames
            last_context_record_id = context_frames[-1].get("id")
            
            # Step 1: Query future frames (next 3 seconds = up to 6 frames at 0.5s each)
            future_frames = EventOrderingGenerator._query_future_frames(
                db_path=db_path,
                scene_id=scene_id,
                source=source,
                last_record_id=last_context_record_id,
                max_frames=6
            )
            
            # If no future frames found, use the last frame from video_sequence
            if not future_frames:
                logger.info("No future frames in database, using last frame from video sequence")
                future_frames = [last_frame]
            
            # Calculate actual duration based on available frames (0.5s per frame)
            actual_duration = len(future_frames) * 0.5
            
            # Step 2: Extract ordered events from all future frames
            ordered_events = EventOrderingGenerator._extract_ordered_events(
                future_frames=future_frames,
                context_frames=context_frames
            )
            
            if not ordered_events or len(ordered_events) < 2:
                logger.warning("Not enough events in future frames (need at least 2)")
                return None
            
            # Create save directories
            question_folder = Path(save_path) / "causality" / EventOrderingGenerator.QUESTION_TYPE.value
            images_folder = question_folder / "images"
            images_folder.mkdir(parents=True, exist_ok=True)
            
            # Generate unique ID for this question
            question_id = str(uuid4())
            
            # Step 3: Create video visualization (only context frames, excluding last)
            # Step 3: Create folder for frames
            video_folder = images_folder / question_id
            video_folder.mkdir(parents=True, exist_ok=True)
            
            # Prepare highlight info for all objects mentioned in events
            # Each unique object gets a different color
            colors = [
                (255, 0, 0),      # Blue
                (255, 0, 255),    # Magenta
                (0, 255, 255),    # Cyan
                (0, 255, 0),      # Green
                (255, 128, 0),    # Orange
                (128, 0, 255),    # Purple
            ]
            
            # Build CONSISTENT track_to_object_id mapping from FIRST context frame
            # This ensures object IDs are stable across all frames
            first_context_track_3d = context_frames[0].get('annotation', {}).get('track_3d', [])
            reference_track_to_object_id = {}
            for idx, track_id in enumerate(first_context_track_3d):
                if track_id != '00000':
                    reference_track_to_object_id[track_id] = idx + 1
            
            # Collect unique track_ids from all events
            # Each unique object will get a different color and consistent object_id
            unique_tracks = {}
            for event in ordered_events:
                track_id = event['track_id']
                if track_id not in unique_tracks:
                    # Use reference mapping for consistent object_id
                    object_id = reference_track_to_object_id.get(track_id, event['object_id'])
                    unique_tracks[track_id] = object_id
            
            # Also collect objects mentioned in interactions (e.g., "leading Object-25")
            # We need to find the track_id for these mentioned objects
            mentioned_object_ids = EventOrderingGenerator._extract_mentioned_object_ids(ordered_events)
            
            # Get track IDs for all mentioned objects using the REFERENCE mapping
            for obj_id in mentioned_object_ids:
                if obj_id not in [oid for oid in unique_tracks.values()]:
                    # Find the track_id for this object_id from REFERENCE mapping
                    track_id = EventOrderingGenerator._find_track_id_by_object_id_from_mapping(
                        obj_id, reference_track_to_object_id
                    )
                    if track_id:
                        unique_tracks[track_id] = obj_id
            
            highlight_tracks = []
            for idx, (track_id, object_id) in enumerate(unique_tracks.items()):
                # Assign a unique color to each object
                # Include object_id for consistent labeling across frames
                color = colors[idx % len(colors)]
                highlight_tracks.append({
                    'track_id': track_id,
                    'color': color,
                    'object_id': object_id  # Pass consistent object_id for visualization
                })
            
            # Save frames to folder instead of video
            frame_paths = save_frames_to_folder(
                frame_sequence=context_frames,
                output_folder=str(video_folder),
                source=source,
                show_object_ids=True,
                highlight_objects=highlight_tracks,
                show_only_highlighted=True  # Only show the objects mentioned in events
            )
            
            # Step 4: Generate correct answer (chronologically ordered events)
            correct_answer = EventOrderingGenerator._format_ordered_events(ordered_events)
            
            # Step 5: Generate distractor options (shuffled orderings)
            options, correct_idx = EventOrderingGenerator._generate_ordering_options(
                ordered_events, correct_answer, num_options
            )
            
            # Step 6: Build question text (always asks about 3 seconds)
            question_text = EventOrderingGenerator.QUESTION_TEMPLATE
            question_text += f"\nVideo sequence (context): <video>\n\nOptions:\n"
            for idx, option in enumerate(options):
                question_text += f"({chr(65+idx)}) {option}\n"
            
            # Step 7: Create question dictionary
            question_dict = {
                "question_id": question_id,
                "question_type": EventOrderingGenerator.QUESTION_TYPE.value,
                "question": question_text,
                "videos": [str(video_folder)],  # Absolute path to folder containing frames
                "frame_paths": frame_paths,  # List of frame filenames
                "options": options,
                "correct_answer": correct_idx,
                "metadata": {
                    "scene_id": scene_id,
                    "context_frame_ids": [f.get("frame_id") for f in context_frames],
                    "future_frame_ids": [f.get("frame_id") for f in future_frames],
                    "source": source,
                    "context_record_ids": [f.get("id") for f in context_frames],
                    "future_record_ids": [f.get("id") for f in future_frames],
                    "actual_duration_seconds": actual_duration,
                    "used_last_frame_fallback": len(future_frames) == 1 and future_frames[0].get("id") == last_frame.get("id"),
                    "reasoning_chain": {
                        "num_context_frames": len(context_frames),
                        "num_future_frames": len(future_frames),
                        "num_events": len(ordered_events),
                        "ordered_events": ordered_events
                    }
                }
            }
            
            return question_dict
            
        except Exception as e:
            logger.error(f"Error generating event ordering question: {e}", exc_info=True)
            return None
    
    @staticmethod
    def _extract_mentioned_object_ids(ordered_events: List[Dict[str, Any]]) -> set:
        """
        Extract all object IDs mentioned in event descriptions (including interaction targets).
        
        Args:
            ordered_events: List of event dictionaries
            
        Returns:
            Set of object IDs mentioned in the events
        """
        import re
        mentioned_ids = set()
        
        for event in ordered_events:
            event_desc = event.get('event_description', '')
            # Find patterns like "Object-25" or "Object-4"
            matches = re.findall(r'Object-(\d+)', event_desc)
            for match in matches:
                mentioned_ids.add(int(match))
        
        return mentioned_ids
    
    @staticmethod
    def _find_track_id_by_object_id(
        object_id: int, 
        frames: List[Dict[str, Any]]
    ) -> Optional[str]:
        """
        Find the track_id corresponding to an object_id in the given frames.
        
        DEPRECATED: Use _find_track_id_by_object_id_from_mapping for consistent IDs.
        This method has issues because track_3d order can vary between frames.
        
        Args:
            object_id: The object ID to find (1-indexed)
            frames: List of frame dictionaries
            
        Returns:
            Track ID string or None if not found
        """
        for frame in frames:
            track_3d = frame.get('annotation', {}).get('track_3d', [])
            # object_id is 1-indexed, so subtract 1 to get the track_3d index
            if 0 < object_id <= len(track_3d):
                return track_3d[object_id - 1]
        
        return None
    
    @staticmethod
    def _find_track_id_by_object_id_from_mapping(
        object_id: int,
        track_to_object_id: Dict[str, int]
    ) -> Optional[str]:
        """
        Find the track_id corresponding to an object_id using a consistent mapping.
        
        This method ensures consistent object IDs by using a pre-built mapping
        from the reference frame (first context frame).
        
        Args:
            object_id: The object ID to find (1-indexed)
            track_to_object_id: Mapping from track_id to object_id
            
        Returns:
            Track ID string or None if not found
        """
        for track_id, obj_id in track_to_object_id.items():
            if obj_id == object_id:
                return track_id
        return None
    
    @staticmethod
    def _extract_object_events(
        frame: Dict[str, Any],
        context_frames: List[Dict[str, Any]] = None,
        track_to_object_id: Dict[str, int] = None
    ) -> List[Dict[str, Any]]:
        """
        Extract all object events (actions and interactions) from a frame.
        
        Args:
            frame: Frame dictionary with internal_interactions and external_interactions
            context_frames: Optional list of context frames to check object visibility
            track_to_object_id: Optional mapping from track_id to object_id for consistent numbering
            
        Returns:
            List of object event dictionaries with track_id, label, actions, interactions
        """
        internal_interactions = frame.get("internal_interactions", [])
        external_interactions = frame.get("external_interactions", [])
        track_3d = frame['annotation'].get('track_3d', [])
        label_3d = frame['annotation'].get('label_3d', [])
        
        # If no track_to_object_id provided, create one from this frame's track_3d
        if track_to_object_id is None:
            track_to_object_id = {tid: idx + 1 for idx, tid in enumerate(track_3d) if tid != '00000'}
        
        # Build a map of track_id -> events
        track_to_events = {}
        
        # Process internal interactions (intrinsic actions)
        for internal_data in internal_interactions:
            track_id = internal_data.get('track_id')
            label = internal_data.get('label', 'unknown')
            
            # Skip ego vehicle (we don't ask about ego's actions)
            if track_id == '00000':
                continue
            
            if not track_id or track_id not in track_to_object_id:
                continue
            
            if track_id not in track_to_events:
                track_to_events[track_id] = {
                    'object_id': track_to_object_id[track_id],
                    'track_id': track_id,
                    'label': label,
                    'actions': [],
                    'interactions': []
                }
            
            # Extract actions from internal interactions (already in 'actions' field)
            actions_list = internal_data.get('actions', [])
            for action_str in actions_list:
                try:
                    # Actions might have "IntrinsicInteraction." prefix, remove it
                    clean_action = action_str.replace('IntrinsicInteraction.', '').upper()
                    action_enum = IntrinsicInteraction[clean_action]
                    action_desc = EventOrderingGenerator.INTRINSIC_ACTION_DESCRIPTIONS.get(
                        action_enum, clean_action.lower().replace('_', ' ')
                    )
                    if action_desc not in track_to_events[track_id]['actions']:
                        track_to_events[track_id]['actions'].append(action_desc)
                except (KeyError, AttributeError, ValueError):
                    # Try without enum conversion
                    action_desc = action_str.lower().replace('_', ' ')
                    if action_desc not in track_to_events[track_id]['actions']:
                        track_to_events[track_id]['actions'].append(action_desc)
        
        # Process external interactions (extrinsic interactions)
        for external_data in external_interactions:
            source_obj = external_data.get('source_object', {})
            source_track = source_obj.get('track_id')
            source_label = source_obj.get('label', 'unknown')
            
            target_obj = external_data.get('target_object', {})
            target_track = target_obj.get('track_id')
            target_label = target_obj.get('label', 'unknown')
            
            # Skip if source is ego (we don't ask about ego's interactions)
            if source_track == '00000':
                continue
            
            if not source_track or source_track not in track_to_object_id:
                continue
            
            if source_track not in track_to_events:
                track_to_events[source_track] = {
                    'object_id': track_to_object_id[source_track],
                    'track_id': source_track,
                    'label': source_label,
                    'actions': [],
                    'interactions': []
                }
            
            # Extract interaction from external interactions (single 'interaction_type' field)
            interaction_str = external_data.get('interaction_type', '')
            if interaction_str:
                try:
                    # Convert to enum
                    interaction_enum = ExtrinsicInteraction[interaction_str.upper()]
                    interaction_desc = EventOrderingGenerator.EXTRINSIC_INTERACTION_DESCRIPTIONS.get(
                        interaction_enum, interaction_str.replace('_', ' ')
                    )
                    
                    # Build interaction string with target
                    if target_track == '00000':
                        # Target is ego
                        interaction_full = f"{interaction_desc} ego vehicle"
                    elif target_track in track_to_object_id:
                        interaction_full = f"{interaction_desc} Object-{track_to_object_id[target_track]}"
                    else:
                        # Target not in reference track_3d, use generic description
                        interaction_full = f"{interaction_desc} {target_label}"
                    
                    if interaction_full not in track_to_events[source_track]['interactions']:
                        track_to_events[source_track]['interactions'].append(interaction_full)
                except (KeyError, AttributeError, ValueError):
                    # Fallback to raw string
                    interaction_desc = interaction_str.replace('_', ' ')
                    if target_track == '00000':
                        interaction_full = f"{interaction_desc} ego vehicle"
                    elif target_track in track_to_object_id:
                        interaction_full = f"{interaction_desc} Object-{track_to_object_id[target_track]}"
                    else:
                        interaction_full = f"{interaction_desc} {target_label}"
                    
                    if interaction_full not in track_to_events[source_track]['interactions']:
                        track_to_events[source_track]['interactions'].append(interaction_full)
        
        # Convert to list and filter objects with events
        object_events = []
        for track_id, event_data in track_to_events.items():
            if event_data['actions'] or event_data['interactions']:
                object_events.append(event_data)
        
        # Sort by object_id for consistent ordering
        object_events.sort(key=lambda x: x['object_id'])
        
        # Prioritize and limit to maximum 3 events, ensuring visibility and preferring near objects
        annotation = frame.get('annotation', {})
        object_events = EventOrderingGenerator._prioritize_and_limit_events(
            object_events, 
            max_events=3,
            video_sequence=context_frames,
            annotation=annotation
        )
        
        return object_events
    
    @staticmethod
    def _prioritize_and_limit_events(
        object_events: List[Dict[str, Any]], 
        max_events: int = 3,
        video_sequence: List[Dict[str, Any]] = None,
        annotation: Dict[str, Any] = None
    ) -> List[Dict[str, Any]]:
        """
        Prioritize and limit object events to maximum count, preferring near objects.
        
        Priority:
        1. Objects visible in at least one camera view in the video sequence
        2. Objects with actions that are NOT "stopped"
        3. Objects with interactions
        4. Near objects (< 10m from ego) have higher selection probability
        5. Objects with "stopped" action (only if no other objects available)
        
        Args:
            object_events: List of object event dictionaries
            max_events: Maximum number of events to return (default: 3)
            video_sequence: Optional list of context frames to check visibility
            annotation: Optional annotation data for distance calculation
            
        Returns:
            Filtered list of object events (max length = max_events)
        """
        # Filter objects by visibility in camera views
        if video_sequence:
            visible_objects = EventOrderingGenerator._filter_visible_objects(
                object_events, video_sequence
            )
            # If we have enough visible objects, use only those
            if len(visible_objects) >= max_events:
                object_events = visible_objects
            # Otherwise keep all and hope for the best
        
        if len(object_events) <= max_events:
            return object_events
        
        # Categorize objects
        non_stopped_objects = []
        stopped_only_objects = []
        
        for obj_event in object_events:
            actions = obj_event.get('actions', [])
            interactions = obj_event.get('interactions', [])
            
            # Check if object has non-stopped actions or any interactions
            has_non_stopped = any(action != 'stopped' for action in actions)
            has_interactions = len(interactions) > 0
            
            if has_non_stopped or has_interactions:
                # Priority: objects with interesting actions or interactions
                non_stopped_objects.append(obj_event)
            elif 'stopped' in actions:
                # Low priority: stopped-only objects
                stopped_only_objects.append(obj_event)
        
        # Select up to max_events using weighted selection (prefer near objects)
        selected_events = []
        
        if annotation is None and video_sequence:
            annotation = video_sequence[0].get('annotation', {})
        
        # Helper to convert object_events to event format for weighted selection
        def to_event_format(obj_events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
            return [{'track_id': e.get('track_id', ''), **e} for e in obj_events]
        
        if non_stopped_objects:
            # Use weighted selection for non-stopped objects
            if len(non_stopped_objects) > max_events and annotation:
                selected_events = EventOrderingGenerator._weighted_choice_events(
                    events=to_event_format(non_stopped_objects),
                    annotation=annotation,
                    near_threshold=EventOrderingGenerator.NEAR_OBJECT_THRESHOLD,
                    near_probability=EventOrderingGenerator.NEAR_OBJECT_PROBABILITY,
                    max_events=max_events
                )
            else:
                selected_events = non_stopped_objects[:max_events]
        
        # If we still have space and no other objects, add stopped objects
        remaining_slots = max_events - len(selected_events)
        if remaining_slots > 0 and len(non_stopped_objects) == 0:
            if len(stopped_only_objects) > remaining_slots and annotation:
                selected_events.extend(EventOrderingGenerator._weighted_choice_events(
                    events=to_event_format(stopped_only_objects),
                    annotation=annotation,
                    near_threshold=EventOrderingGenerator.NEAR_OBJECT_THRESHOLD,
                    near_probability=EventOrderingGenerator.NEAR_OBJECT_PROBABILITY,
                    max_events=remaining_slots
                ))
            else:
                selected_events.extend(stopped_only_objects[:remaining_slots])
        
        return selected_events
    
    @staticmethod
    def _filter_visible_objects(
        object_events: List[Dict[str, Any]],
        video_sequence: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Filter objects to only those visible in at least one camera view in any frame.
        
        Args:
            object_events: List of object event dictionaries with track_id
            video_sequence: List of frame dictionaries
            
        Returns:
            Filtered list of objects that appear in camera views
        """
        visible_track_ids = set()
        
        # Collect all track IDs that appear in 2D annotations (camera views)
        for frame in video_sequence:
            track_2d = frame.get('annotation', {}).get('track_2d', [])
            # track_2d is a list of lists (one per camera)
            for camera_tracks in track_2d:
                if isinstance(camera_tracks, list):
                    visible_track_ids.update(camera_tracks)
        
        # Filter object_events to only include visible objects
        visible_objects = []
        for obj_event in object_events:
            if obj_event['track_id'] in visible_track_ids:
                visible_objects.append(obj_event)
        
        return visible_objects
    
    @staticmethod
    def _query_future_frames(
        db_path: str,
        scene_id: str,
        source: str,
        last_record_id: int,
        max_frames: int = 6
    ) -> List[Dict[str, Any]]:
        """
        Query future frames from the database after the given record_id.
        
        Uses timestamp for proper temporal ordering.
        
        Args:
            db_path: Path to SQLite database
            scene_id: Scene identifier
            source: Dataset source
            last_record_id: The last record_id in the context
            max_frames: Maximum number of future frames to retrieve (default: 6 for 3 seconds at 0.5s each)
            
        Returns:
            List of future frame dictionaries in chronological order
        """
        try:
            from sources.db.sql import SQLiteController
            
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
                    return []
                
                # Parse JSON fields and extract timestamps
                future_frames = []
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
                        # Fallback: use record ID as pseudo-timestamp
                        timestamp = frame.get('id', 0)
                    
                    frame['_timestamp'] = timestamp
                    future_frames.append(frame)
                
                # Sort frames by timestamp
                future_frames.sort(key=lambda f: f['_timestamp'])
                
                # Limit to max_frames
                future_frames = future_frames[:max_frames]
                
                return future_frames
            
        except Exception as e:
            logger.error(f"Error querying future frames: {e}", exc_info=True)
            return []
    
    @staticmethod
    def _extract_ordered_events(
        future_frames: List[Dict[str, Any]],
        context_frames: List[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Extract events from future frames in chronological order.
        
        Each event represents one action or interaction at a specific time.
        Events are ordered by frame, then by object.
        
        Args:
            future_frames: List of future frame dictionaries
            context_frames: Optional context frames to check object visibility
            
        Returns:
            List of event dictionaries with: frame_index, time_offset, track_id, object_id, event_description
        """
        ordered_events = []
        
        # Use the FIRST CONTEXT FRAME's track_3d as reference for consistent object IDs
        # This ensures object IDs match the visualization (which uses context frames)
        if context_frames and context_frames[0].get('annotation', {}).get('track_3d'):
            reference_track_3d = context_frames[0]['annotation']['track_3d']
        else:
            # Fallback: use first future frame's track_3d
            reference_track_3d = future_frames[0]['annotation'].get('track_3d', []) if future_frames else []
        
        # Get all track_ids visible in context frames' cameras (track_2d)
        # This ensures we only ask about objects that appear in the visualization
        visible_in_context = EventOrderingGenerator._get_visible_track_ids(context_frames) if context_frames else set()
        
        # Create object ID mapping based on reference track_3d (1-indexed)
        # Only include objects that are BOTH in track_3d AND visible in context frames' cameras
        track_to_object_id = {}
        for idx, track_id in enumerate(reference_track_3d):
            if track_id != '00000' and (not visible_in_context or track_id in visible_in_context):
                track_to_object_id[track_id] = idx + 1
        
        # Extract events from each frame
        for frame_idx, frame in enumerate(future_frames):
            time_offset = (frame_idx + 1) * 0.5  # Each frame is 0.5s ahead
            
            # Extract events from this frame using consistent object ID mapping
            frame_events = EventOrderingGenerator._extract_object_events(
                frame=frame,
                context_frames=context_frames,
                track_to_object_id=track_to_object_id
            )
            
            # Add frame and time information to each event
            for event in frame_events:
                track_id = event['track_id']
                object_id = track_to_object_id.get(track_id, event['object_id'])
                
                # Create individual events for actions and interactions
                for action in event.get('actions', []):
                    ordered_events.append({
                        'frame_index': frame_idx,
                        'time_offset': time_offset,
                        'track_id': track_id,
                        'object_id': object_id,
                        'label': event['label'],
                        'event_type': 'action',
                        'event_description': action
                    })
                
                for interaction in event.get('interactions', []):
                    ordered_events.append({
                        'frame_index': frame_idx,
                        'time_offset': time_offset,
                        'track_id': track_id,
                        'object_id': object_id,
                        'label': event['label'],
                        'event_type': 'interaction',
                        'event_description': interaction
                    })
        
        # Limit to maximum 3 events total, prioritizing non-stopped actions and near objects
        # Also ensure objects appear in context frames
        if context_frames:
            # Filter to only include objects visible in context
            visible_track_ids = EventOrderingGenerator._get_visible_track_ids(context_frames)
            ordered_events = [e for e in ordered_events if e['track_id'] in visible_track_ids]
        
        if len(ordered_events) > 3:
            # Separate events
            non_stopped = [e for e in ordered_events if 'stopped' not in e['event_description'].lower()]
            stopped = [e for e in ordered_events if 'stopped' in e['event_description'].lower()]
            
            # Get annotation from first future frame for distance calculation
            annotation = future_frames[0].get('annotation', {}) if future_frames else {}
            
            if len(non_stopped) >= 3:
                # Use 80% probability to select near objects (10m)
                ordered_events = EventOrderingGenerator._weighted_choice_events(
                    events=non_stopped,
                    annotation=annotation,
                    near_threshold=EventOrderingGenerator.NEAR_OBJECT_THRESHOLD,
                    near_probability=EventOrderingGenerator.NEAR_OBJECT_PROBABILITY,
                    max_events=3
                )
            elif non_stopped:
                # Select from non-stopped with 80% near probability
                selected_non_stopped = EventOrderingGenerator._weighted_choice_events(
                    events=non_stopped,
                    annotation=annotation,
                    near_threshold=EventOrderingGenerator.NEAR_OBJECT_THRESHOLD,
                    near_probability=EventOrderingGenerator.NEAR_OBJECT_PROBABILITY,
                    max_events=len(non_stopped)
                )
                # Fill remaining with stopped
                remaining = 3 - len(selected_non_stopped)
                ordered_events = selected_non_stopped + stopped[:remaining]
            else:
                # Only stopped events - use 80% near probability
                ordered_events = EventOrderingGenerator._weighted_choice_events(
                    events=stopped,
                    annotation=annotation,
                    near_threshold=EventOrderingGenerator.NEAR_OBJECT_THRESHOLD,
                    near_probability=EventOrderingGenerator.NEAR_OBJECT_PROBABILITY,
                    max_events=3
                )
        
        return ordered_events
    
    @staticmethod
    def _get_visible_track_ids(frames: List[Dict[str, Any]]) -> set:
        """
        Get all track IDs that are visible in at least one camera view across the given frames.
        
        Args:
            frames: List of frame dictionaries
            
        Returns:
            Set of visible track IDs
        """
        visible_track_ids = set()
        
        for frame in frames:
            track_2d = frame.get('annotation', {}).get('track_2d', [])
            # track_2d is a list of lists (one per camera)
            for camera_tracks in track_2d:
                if isinstance(camera_tracks, list):
                    visible_track_ids.update(camera_tracks)
        
        return visible_track_ids
    
    @staticmethod
    def _format_ordered_events(ordered_events: List[Dict[str, Any]]) -> str:
        """
        Format ordered events into a chronological sequence string.
        
        Args:
            ordered_events: List of event dictionaries in chronological order
            
        Returns:
            Formatted string like "1. Object-1: moving forward → 2. Object-2: turning left → 3. Object-1: following Object-2"
        """
        event_strs = []
        
        for idx, event in enumerate(ordered_events):
            obj_id = event['object_id']
            description = event['event_description']
            event_str = f"{idx + 1}. Object-{obj_id}: {description}"
            event_strs.append(event_str)
        
        return " → ".join(event_strs)
    
    @staticmethod
    def _generate_ordering_options(
        ordered_events: List[Dict[str, Any]],
        correct_answer: str,
        num_options: int = 4
    ) -> Tuple[List[str], int]:
        """
        Generate distractor options by modifying actions/interactions (not just shuffling order).
        
        This ensures only 1 correct answer exists, and distractors have different events
        that are clearly distinguishable from the correct answer.
        
        Args:
            ordered_events: List of event dictionaries in correct chronological order
            correct_answer: The correct formatted answer string
            num_options: Number of options to generate
            
        Returns:
            Tuple of (options_list, correct_answer_index)
        """
        options = [correct_answer]
        
        # All available actions and interactions for creating distractors
        all_actions = list(EventOrderingGenerator.INTRINSIC_ACTION_DESCRIPTIONS.values())
        all_interactions = list(EventOrderingGenerator.EXTRINSIC_INTERACTION_DESCRIPTIONS.values())
        
        # Generate distractors by modifying actions/interactions
        attempts = 0
        max_attempts = num_options * 10
        
        while len(options) < num_options and attempts < max_attempts:
            attempts += 1
            
            # Create a modified version of events
            modified_events = []
            for event in ordered_events:
                modified_event = event.copy()
                event_desc = event['event_description']
                event_type = event.get('event_type', 'action')
                
                # Randomly modify the action/interaction
                modification_type = random.choice(['replace_action', 'replace_interaction', 'change_object'])
                
                if modification_type == 'replace_action' or event_type == 'action':
                    # Replace with a different action
                    current_action = event_desc
                    available_actions = [a for a in all_actions if a != current_action]
                    if available_actions:
                        new_action = random.choice(available_actions)
                        modified_event['event_description'] = new_action
                
                elif modification_type == 'replace_interaction' or event_type == 'interaction':
                    # Replace interaction verb while keeping target
                    for verb in all_interactions:
                        if event_desc.startswith(verb):
                            target_part = event_desc[len(verb):].strip()
                            available_verbs = [v for v in all_interactions if v != verb]
                            if available_verbs:
                                new_verb = random.choice(available_verbs)
                                modified_event['event_description'] = f"{new_verb} {target_part}"
                            break
                    else:
                        # Couldn't parse, replace with random action
                        modified_event['event_description'] = random.choice(all_actions)
                
                elif modification_type == 'change_object':
                    # Change the object ID slightly
                    current_obj_id = event['object_id']
                    new_obj_id = current_obj_id + random.choice([-2, -1, 1, 2, 3])
                    if new_obj_id > 0:
                        modified_event['object_id'] = new_obj_id
                
                modified_events.append(modified_event)
            
            # Format the distractor
            distractor = EventOrderingGenerator._format_ordered_events(modified_events)
            
            # Ensure it's different from correct answer and existing options
            if distractor != correct_answer and distractor not in options:
                options.append(distractor)
        
        # Fill remaining slots with more diverse distractors
        while len(options) < num_options:
            # Generate completely different events
            distractor_events = []
            for idx, event in enumerate(ordered_events):
                fake_event = event.copy()
                # Randomly pick different action
                fake_event['event_description'] = random.choice(all_actions)
                # Optionally change object ID
                if random.random() > 0.5:
                    fake_event['object_id'] = event['object_id'] + random.randint(1, 5)
                distractor_events.append(fake_event)
            
            distractor = EventOrderingGenerator._format_ordered_events(distractor_events)
            if distractor not in options:
                options.append(distractor)
            else:
                # Last resort: create with random object and action
                random_obj_id = random.randint(1, 20)
                random_action = random.choice(all_actions)
                single_event = [{
                    'object_id': random_obj_id,
                    'event_description': random_action
                }]
                for i in range(len(ordered_events) - 1):
                    single_event.append({
                        'object_id': random.randint(1, 20),
                        'event_description': random.choice(all_actions)
                    })
                distractor = EventOrderingGenerator._format_ordered_events(single_event)
                if distractor not in options:
                    options.append(distractor)
                else:
                    break  # Give up if we can't create unique options
        
        # Shuffle options to randomize correct answer position
        random.shuffle(options)
        correct_idx = options.index(correct_answer)
        
        return options, correct_idx
    
    @staticmethod
    def _format_event_string(object_events: List[Dict[str, Any]]) -> str:
        """
        Format object events into a human-readable string.
        Maximum 3 events (objects), each with their actions and interactions combined.
        
        Args:
            object_events: List of object event dictionaries (should be pre-filtered to max 3)
            
        Returns:
            Formatted string like "Object-1: moving forward, following Object-2; Object-2: turning left"
        """
        event_parts = []
        
        for obj_event in object_events:
            obj_id = obj_event['object_id']
            parts = []
            
            # Add actions (limit to first 2 to keep concise)
            if obj_event['actions']:
                parts.extend(obj_event['actions'][:2])
            
            # Add interactions (limit to first 1 to keep concise)
            if obj_event['interactions']:
                parts.extend(obj_event['interactions'][:1])
            
            if parts:
                event_str = f"Object-{obj_id}: {', '.join(parts)}"
                event_parts.append(event_str)
        
        return "; ".join(event_parts)
    
    @staticmethod
    def _generate_event_options(
        object_events: List[Dict[str, Any]],
        correct_answer: str,
        num_options: int = 4
    ) -> Tuple[List[str], int]:
        """
        Generate distractor options by modifying actions/interactions.
        
        Args:
            object_events: List of object event dictionaries
            correct_answer: The correct formatted answer string
            num_options: Number of options to generate
            
        Returns:
            Tuple of (options_list, correct_answer_index)
        """
        options = [correct_answer]
        
        # Generate distractors
        all_actions = list(EventOrderingGenerator.INTRINSIC_ACTION_DESCRIPTIONS.values())
        all_interactions = list(EventOrderingGenerator.EXTRINSIC_INTERACTION_DESCRIPTIONS.values())
        
        for _ in range(num_options - 1):
            # Create a modified version
            distractor_events = []
            for obj_event in object_events:
                modified_event = obj_event.copy()
                modified_event['actions'] = obj_event['actions'].copy()
                modified_event['interactions'] = obj_event['interactions'].copy()
                
                # Randomly modify actions or interactions
                modification_type = random.choice(['action', 'interaction', 'both', 'none'])
                
                if modification_type in ['action', 'both'] and modified_event['actions']:
                    # Replace one action with a different one
                    if len(modified_event['actions']) > 0:
                        idx_to_replace = random.randint(0, len(modified_event['actions']) - 1)
                        new_action = random.choice([a for a in all_actions if a not in modified_event['actions']])
                        modified_event['actions'][idx_to_replace] = new_action
                
                if modification_type in ['interaction', 'both'] and modified_event['interactions']:
                    # Replace interaction verb
                    modified_interactions = []
                    for interaction_str in modified_event['interactions']:
                        # Extract the object reference
                        if ' Object-' in interaction_str or ' ego vehicle' in interaction_str:
                            # Split verb and target
                            for verb in all_interactions:
                                if interaction_str.startswith(verb):
                                    target_part = interaction_str[len(verb):].strip()
                                    new_verb = random.choice([i for i in all_interactions if i != verb])
                                    modified_interactions.append(f"{new_verb} {target_part}")
                                    break
                            else:
                                # Couldn't parse, keep original
                                modified_interactions.append(interaction_str)
                        else:
                            modified_interactions.append(interaction_str)
                    modified_event['interactions'] = modified_interactions
                
                distractor_events.append(modified_event)
            
            distractor_str = EventOrderingGenerator._format_event_string(distractor_events)
            
            # Ensure it's different from correct answer and existing options
            if distractor_str != correct_answer and distractor_str not in options:
                options.append(distractor_str)
        
        # If we couldn't generate enough unique distractors, add simple variations
        while len(options) < num_options:
            # Remove a random object or add/remove an action
            if len(object_events) > 1:
                shuffled_events = object_events.copy()
                # Remove a random object
                shuffled_events.pop(random.randint(0, len(shuffled_events) - 1))
                distractor_str = EventOrderingGenerator._format_event_string(shuffled_events)
                if distractor_str not in options:
                    options.append(distractor_str)
                else:
                    # Add generic distractor
                    random_obj_id = random.randint(1, len(object_events) + 5)
                    random_action = random.choice(all_actions)
                    options.append(f"Object-{random_obj_id}: {random_action}")
            else:
                # Single object case
                random_action = random.choice(all_actions)
                options.append(f"Object-{object_events[0]['object_id']}: {random_action}")
        
        # Shuffle options
        random.shuffle(options)
        correct_idx = options.index(correct_answer)
        
        return options, correct_idx
