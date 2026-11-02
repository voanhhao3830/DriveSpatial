import os
import json
import random
import logging
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
from uuid import uuid4

from sources.question_generation.enum import ConstructionQuestionType
from sources.question_generation.utils import save_frames_to_folder

logger = logging.getLogger(__name__)


class CameraMemoryGenerator:
    QUESTION_TYPE = ConstructionQuestionType.CAMERA_MEMORY
    
    QUESTION_TEMPLATE = """Given this video sequence where one camera view is masked after the first frame, base on the video sequence, what objects are present in that masked camera view in the final frame?"""
    
    @staticmethod
    def generate_question(
        sampled_data: Dict[str, Any],
        db_path: str,
        save_path: str,
        num_options: int = 4
    ) -> Optional[Dict[str, Any]]:
        """
        Generate a camera memory question.
        
        Args:
            sampled_data: Single frame data dictionary
            db_path: Path to the database
            save_path: Path to save questions and images
            num_options: Number of options to generate (default: 4)
            
        Returns:
            question: Question dictionary or None if generation fails
        """
        try:
            # Query consecutive frames for this scene
            from sources.db.sql import SQLiteController
            db = SQLiteController(db_path)
            
            scene_id = sampled_data.get('scene_id')
            source = sampled_data.get('metadata', {}).get('source')
            frame_id = sampled_data.get('frame_id')
            record_id = sampled_data.get('id')
            
            # Query next 5-6 frames from this scene
            with db._get_connection() as conn:
                cursor = conn.cursor()
                query = """
                    SELECT * FROM av_data 
                    WHERE scene_id = ? AND source = ? AND id >= ?
                """
                cursor.execute(query, (scene_id, source, record_id))
                rows = cursor.fetchall()
                
                video_sequence = []
                json_fields = ['views', 'lidar', 'annotation', 'calibration',
                               'spatial_relations', 'internal_interactions',
                               'occlusions', 'external_interactions', 'metadata']
                
                for row in rows:
                    result = dict(row)
                    for field in json_fields:
                        if result.get(field):
                            result[field] = json.loads(result[field])
                    
                    # Extract timestamp from calibration for sorting
                    calibration = result.get('calibration', {})
                    img_timestamp = calibration.get('img_timestamp', [])
                    
                    # Use first camera timestamp, or record ID as fallback
                    if img_timestamp and len(img_timestamp) > 0:
                        timestamp = img_timestamp[0]
                    else:
                        # Fallback: use record ID as pseudo-timestamp
                        timestamp = result.get('id', 0)
                    
                    result['_timestamp'] = timestamp
                    video_sequence.append(result)
                
                # Sort frames by timestamp
                video_sequence.sort(key=lambda f: f['_timestamp'])
                
                # Limit to 6 frames
                video_sequence = video_sequence[:6]
            
            if len(video_sequence) < 3:
                logger.warning(f"Not enough frames for camera memory (need at least 3, got {len(video_sequence)})")
                return None
            
            # Select a camera to mask (not the front camera)
            views = list(video_sequence[0].get('views', {}).keys())
            if len(views) < 2:
                logger.warning("Not enough camera views")
                return None
            
            # Exclude front camera if possible
            front_camera_names = ['CAM_FRONT', 'CAMERA_FRONT', 'camera_front', 'front']
            non_front_cameras = [v for v in views if not any(fc in v.upper() for fc in front_camera_names)]
            
            if non_front_cameras:
                masked_camera = random.choice(non_front_cameras)
            else:
                # If all are front, just pick any camera except first
                masked_camera = random.choice(views[1:])
            
            masked_camera_idx = views.index(masked_camera)
            
            # Get objects in the masked camera in the final frame
            final_frame = video_sequence[-1]
            objects_in_masked_camera = CameraMemoryGenerator._get_objects_in_camera(
                final_frame, masked_camera_idx
            )
            
            if not objects_in_masked_camera:
                logger.warning("No objects in masked camera in final frame")
                return None
            
            # Create save directories
            question_folder = Path(save_path) / "construct" / CameraMemoryGenerator.QUESTION_TYPE.value
            images_folder = question_folder / "images"
            images_folder.mkdir(parents=True, exist_ok=True)
            
            # Generate unique ID for this question
            question_id = str(uuid4())
            
            # Create folder for frames with masked camera
            video_folder = images_folder / question_id
            video_folder.mkdir(parents=True, exist_ok=True)
            # Save frames with masking applied
            frame_paths = CameraMemoryGenerator._save_masked_frames(
                video_sequence=video_sequence,
                output_folder=str(video_folder),
                masked_camera=masked_camera,
                source=source
            )
            
            # Generate correct answer
            correct_answer = CameraMemoryGenerator._format_answer(objects_in_masked_camera)
            
            # Generate distractor options
            options, correct_idx = CameraMemoryGenerator._generate_options(
                correct_objects=objects_in_masked_camera,
                all_frames=video_sequence,
                masked_camera_idx=masked_camera_idx,
                num_options=num_options
            )
            
            # Build question text
            question_text = CameraMemoryGenerator.QUESTION_TEMPLATE
            question_text += f"\nVideo sequence: <video>\n\nOptions:\n"
            for idx, option in enumerate(options):
                question_text += f"({chr(65+idx)}) {option}\n"
            
            # Create question dictionary
            question_dict = {
                "question_id": question_id,
                "question_type": CameraMemoryGenerator.QUESTION_TYPE.value,
                "question": question_text,
                "videos": [str(video_folder)],  # Absolute path to folder
                "frame_paths": frame_paths,
                "options": options,
                "correct_answer": correct_idx,
                "metadata": {
                    "scene_id": scene_id,
                    "frame_ids": [f.get("frame_id") for f in video_sequence],
                    "source": source,
                    "record_ids": [f.get("id") for f in video_sequence],
                    "masked_camera": masked_camera,
                    "masked_camera_idx": masked_camera_idx,
                    "num_frames": len(video_sequence),
                    "objects_in_masked_camera": objects_in_masked_camera
                }
            }
            
            return question_dict
            
        except Exception as e:
            logger.error(f"Error generating camera memory question: {e}", exc_info=True)
            return None
    
    @staticmethod
    def _get_objects_in_camera(
        frame: Dict[str, Any],
        camera_idx: int
    ) -> List[Dict[str, Any]]:
        """
        Get list of objects visible in a specific camera.
        
        Args:
            frame: Frame data
            camera_idx: Index of the camera
            
        Returns:
            List of object dictionaries with track_id and label
        """
        annotation = frame.get('annotation', {})
        track_2d = annotation.get('track_2d', [])
        bbox_2d = annotation.get('bbox_2d', [])
        track_3d = annotation.get('track_3d', [])
        label_3d = annotation.get('label_3d', [])
        
        if camera_idx >= len(track_2d):
            return []
        
        camera_tracks = track_2d[camera_idx]
        camera_boxes = bbox_2d[camera_idx] if camera_idx < len(bbox_2d) else []
        
        objects = []
        for track_id in camera_tracks:
            if track_id == '00000':  # Skip ego
                continue
            
            # Find label from track_3d
            if track_id in track_3d:
                obj_idx = track_3d.index(track_id)
                label = label_3d[obj_idx] if obj_idx < len(label_3d) else 'unknown'
                
                objects.append({
                    'track_id': track_id,
                    'label': label
                })
        
        return objects
    
    @staticmethod
    def _format_answer(objects: List[Dict[str, Any]]) -> str:
        """
        Format answer as "count label; count label; ..."
        
        Args:
            objects: List of object dictionaries
            
        Returns:
            Formatted answer string
        """
        # Count objects by label
        label_counts = {}
        for obj in objects:
            label = obj['label']
            label_counts[label] = label_counts.get(label, 0) + 1
        
        # Format as "2 Pedestrian; 3 Car; 1 Bicycle"
        parts = []
        for label, count in sorted(label_counts.items()):
            parts.append(f"{count} {label}")
        
        return "; ".join(parts)
    
    @staticmethod
    def _get_camera_region_in_visualization(
        source: str,
        camera_name: str,
        views: Dict[str, Any],
        target_height: int = 480
    ) -> Optional[Tuple[int, int, int, int]]:
        """
        Get the pixel region (x_start, y_start, x_end, y_end) for a camera in the visualization.
        
        This must match the layout logic in visualize_multiview exactly.
        
        Args:
            source: Data source name
            camera_name: Name of camera to find
            views: Dict of camera views (to calculate image sizes)
            target_height: Target height used in visualization (default: 480)
            
        Returns:
            Tuple of (x_start, y_start, x_end, y_end) or None if camera not found
        """
        import cv2
        from sources.question_generation.utils.constants import CAMERA_ORDER
        
        source_lower = source.lower() if source else None
        camera_order = CAMERA_ORDER.get(source_lower, [])
        
        if not camera_order:
            # Fallback to views order for unknown sources
            camera_order = list(views.keys())
        
        # Calculate max_width by loading ALL images and finding the maximum width
        # This matches the logic in visualize_multiview exactly
        max_width = 0
        camera_widths = {}
        for cam_name in camera_order:
            if cam_name not in views:
                continue
            cam_path = views[cam_name]
            if isinstance(cam_path, dict):
                cam_path = cam_path.get("path")
            if cam_path and os.path.exists(cam_path):
                img = cv2.imread(cam_path)
                if img is not None:
                    h, w = img.shape[:2]
                    aspect_ratio = w / h
                    new_width = int(target_height * aspect_ratio)
                    camera_widths[cam_name] = new_width
                    max_width = max(max_width, new_width)
        
        if max_width == 0:
            # Default width if no images found
            max_width = int(target_height * (16 / 9))  # Assume 16:9 aspect ratio
        
        # Get the actual width of this specific camera (for proper masking)
        actual_cam_width = camera_widths.get(camera_name, max_width)
        
        # Calculate position based on source-specific layout
        if source_lower == 'waymo':
            # Waymo layout:
            # Row 0: FRONT_LEFT, FRONT, FRONT_RIGHT
            # Row 1: SIDE_LEFT, [empty], SIDE_RIGHT
            positions = {
                'FRONT_LEFT': (0, 0),
                'FRONT': (max_width, 0),
                'FRONT_RIGHT': (max_width * 2, 0),
                'SIDE_LEFT': (0, target_height),
                'SIDE_RIGHT': (max_width * 2, target_height),
            }
            if camera_name in positions:
                x, y = positions[camera_name]
                # Use actual camera width for proper masking
                return (x, y, x + actual_cam_width, y + target_height)
                
        elif source_lower == 'argoverse2':
            # AV2 layout:
            # Row 0: ring_front_left, ring_front_center, ring_front_right
            # Row 1: ring_side_left, ring_side_right (centered)
            # Row 2: ring_rear_left, ring_rear_right (centered)
            canvas_width = max_width * 3
            row2_offset_x = (canvas_width - 2 * max_width) // 2
            row3_offset_x = row2_offset_x
            
            positions = {
                'ring_front_left': (0, 0),
                'ring_front_center': (max_width, 0),
                'ring_front_right': (max_width * 2, 0),
                'ring_side_left': (row2_offset_x, target_height),
                'ring_side_right': (row2_offset_x + max_width, target_height),
                'ring_rear_left': (row3_offset_x, target_height * 2),
                'ring_rear_right': (row3_offset_x + max_width, target_height * 2),
            }
            if camera_name in positions:
                x, y = positions[camera_name]
                return (x, y, x + actual_cam_width, y + target_height)
                
        elif source_lower == 'once':
            # ONCE layout:
            # Row 0: cam06, cam03, cam01, cam08
            # Row 1: cam05, cam09, cam07 (centered)
            canvas_width = max_width * 4
            row2_offset_x = (canvas_width - 3 * max_width) // 2
            
            positions = {
                'cam06': (0, 0),
                'cam03': (max_width, 0),
                'cam01': (max_width * 2, 0),
                'cam08': (max_width * 3, 0),
                'cam05': (row2_offset_x, target_height),
                'cam09': (row2_offset_x + max_width, target_height),
                'cam07': (row2_offset_x + max_width * 2, target_height),
            }
            if camera_name in positions:
                x, y = positions[camera_name]
                return (x, y, x + actual_cam_width, y + target_height)
                
        elif source_lower == 'truckscenes':
            # Truckscenes: 2 columns grid
            cols = 2
            cam_idx = camera_order.index(camera_name) if camera_name in camera_order else -1
            if cam_idx >= 0:
                row = cam_idx // cols
                col = cam_idx % cols
                x = col * max_width
                y = row * target_height
                return (x, y, x + actual_cam_width, y + target_height)
                
        else:
            # Default: 3-column grid layout (nuscenes and others)
            cols = 3
            cam_idx = camera_order.index(camera_name) if camera_name in camera_order else -1
            if cam_idx >= 0:
                row = cam_idx // cols
                col = cam_idx % cols
                # Center-align each row
                num_images = len([c for c in camera_order if c in views])
                num_in_row = min(cols, num_images - row * cols)
                canvas_width = cols * max_width
                offset_x = (canvas_width - num_in_row * max_width) // 2
                x = offset_x + col * max_width
                y = row * target_height
                return (x, y, x + actual_cam_width, y + target_height)
        
        return None
    
    @staticmethod
    def _save_masked_frames(
        video_sequence: List[Dict[str, Any]],
        output_folder: str,
        masked_camera: str,
        source: str
    ) -> List[str]:
        """
        Save frames with one camera masked (except first frame).
        
        Args:
            video_sequence: List of frame data
            output_folder: Output folder path
            masked_camera: Name of camera to mask
            source: Data source
            
        Returns:
            List of saved frame paths
        """
        import cv2
        import numpy as np
        from pathlib import Path
        from sources.question_generation.utils import visualize_multiview
        
        output_path = Path(output_folder)
        output_path.mkdir(parents=True, exist_ok=True)
        
        frame_paths = []
        
        for frame_idx, frame_data in enumerate(video_sequence):
            # Use record ID (database id) if available for unique filenames
            record_id = frame_data.get('id')
            
            if record_id is not None:
                filename = f"frame_{frame_idx:04d}_record_{record_id}.jpg"
            else:
                filename = f"frame_{frame_idx:04d}.jpg"
            
            frame_path = output_path / filename
            
            # First frame: show all cameras normally
            if frame_idx == 0:
                visualize_multiview(
                    data=frame_data,
                    output_path=str(frame_path),
                    source=source,
                    show_object_ids=True
                )
            else:
                # Other frames: mask the selected camera
                # Create visualization first
                temp_path = output_path / f"temp_{filename}"
                visualize_multiview(
                    data=frame_data,
                    output_path=str(temp_path),
                    source=source,
                    show_object_ids=True
                )
                
                # Load and apply mask
                img = cv2.imread(str(temp_path))
                if img is not None:
                    views = frame_data.get('views', {})
                    
                    # Get camera region using layout-aware calculation
                    region = CameraMemoryGenerator._get_camera_region_in_visualization(
                        source=source,
                        camera_name=masked_camera,
                        views=views
                    )
                    
                    if region is not None:
                        x_start, y_start, x_end, y_end = region
                        
                        # Clamp to image bounds
                        img_h, img_w = img.shape[:2]
                        x_start = max(0, min(x_start, img_w))
                        x_end = max(0, min(x_end, img_w))
                        y_start = max(0, min(y_start, img_h))
                        y_end = max(0, min(y_end, img_h))
                        
                        # Apply black mask
                        img[y_start:y_end, x_start:x_end] = 0
                        
                        # Add "MASKED" text in the center of the masked region
                        text = "MASKED"
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        font_scale = 1.5
                        thickness = 3
                        (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)
                        
                        text_x = x_start + (x_end - x_start - text_width) // 2
                        text_y = y_start + (y_end - y_start + text_height) // 2
                        cv2.putText(img, text, (text_x, text_y), font, font_scale, (255, 255, 255), thickness)
                    else:
                        logger.warning(f"Could not find camera region for {masked_camera} in {source}")
                        
                    # Save masked image
                    cv2.imwrite(str(frame_path), img)
                
                # Remove temp file
                if temp_path.exists():
                    temp_path.unlink()
            
            frame_paths.append(filename)
        
        return frame_paths
    
    @staticmethod
    def _generate_options(
        correct_objects: List[Dict[str, Any]],
        all_frames: List[Dict[str, Any]],
        masked_camera_idx: int,
        num_options: int = 4
    ) -> Tuple[List[str], int]:
        """
        Generate distractor options.
        
        Args:
            correct_objects: Correct objects in masked camera
            all_frames: All frames in sequence
            masked_camera_idx: Index of masked camera
            num_options: Number of options
            
        Returns:
            Tuple of (options_list, correct_answer_index)
        """
        correct_answer = CameraMemoryGenerator._format_answer(correct_objects)
        options = [correct_answer]
        
        # Get objects from other cameras in final frame
        final_frame = all_frames[-1]
        annotation = final_frame.get('annotation', {})
        track_2d = annotation.get('track_2d', [])
        track_3d = annotation.get('track_3d', [])
        label_3d = annotation.get('label_3d', [])
        
        distractor_types = []
        
        # Type 1: Objects from other cameras
        for cam_idx, camera_tracks in enumerate(track_2d):
            if cam_idx == masked_camera_idx:
                continue
            
            other_objects = []
            for track_id in camera_tracks:
                if track_id == '00000':
                    continue
                if track_id in track_3d:
                    obj_idx = track_3d.index(track_id)
                    label = label_3d[obj_idx] if obj_idx < len(label_3d) else 'unknown'
                    other_objects.append({'track_id': track_id, 'label': label})
            
            if other_objects:
                distractor_types.append(CameraMemoryGenerator._format_answer(other_objects))
        
        # Type 2: Subset of correct objects
        if len(correct_objects) > 1:
            for i in range(len(correct_objects)):
                subset = [obj for idx, obj in enumerate(correct_objects) if idx != i]
                distractor_types.append(CameraMemoryGenerator._format_answer(subset))
        
        # Type 3: Incorrect counts
        label_counts = {}
        for obj in correct_objects:
            label = obj['label']
            label_counts[label] = label_counts.get(label, 0) + 1
        
        for label in label_counts:
            wrong_count = label_counts[label] + random.choice([-1, 1, 2])
            if wrong_count > 0:
                wrong_counts = label_counts.copy()
                wrong_counts[label] = wrong_count
                parts = [f"{count} {label}" for label, count in sorted(wrong_counts.items())]
                distractor_types.append("; ".join(parts))
        
        # Sample distractors
        random.shuffle(distractor_types)
        for distractor in distractor_types:
            if distractor != correct_answer and distractor not in options:
                options.append(distractor)
            if len(options) >= num_options:
                break
        
        # Fill with random variations if needed
        while len(options) < num_options:
            fake_count = random.randint(1, 5)
            fake_label = random.choice(['Car', 'Pedestrian', 'Bicycle', 'Truck', 'Bus'])
            fake_option = f"{fake_count} {fake_label}"
            if fake_option not in options:
                options.append(fake_option)
        
        # Shuffle and return
        random.shuffle(options)
        correct_idx = options.index(correct_answer)
        
        return options, correct_idx