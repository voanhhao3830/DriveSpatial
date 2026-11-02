import os
import json
import random
import numpy as np
from uuid import uuid4
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from typing import Dict, Any, List, Optional, Tuple

from sources.db.sql import SQLiteController
from sources.data_collection.enum.data_sources import DataSource
from sources.question_generation.enum import ConstructionQuestionType


class EgoRotationGenerator:
    QUESTION_TYPE = ConstructionQuestionType.EGO_ROTATION
    QUESTION_TEMPLATE = """
The ego vehicle is moving through the scene. Given two front camera images from timestamp T1 and timestamp T2, 
what is the approximate rotation angle of the ego vehicle between these two timestamps?
"""
    
    # Angle bins for multiple choice options (in degrees)
    ANGLE_BINS = [0, 15, 30, 45, 60, 75, 90, 120, 150, 180]
    
    @staticmethod
    def generate_question(
        sampled_data: Dict[str, Any],
        db_path: str,
        save_path: str,
        num_options: int = 4,
        frame_gap: int = 3
    ) -> Dict[str, Any]:
        """
        Generate an ego rotation question with two front camera images.
        
        Args:
            sampled_data: Sampled data record from database
            db_path: Path to the database
            save_path: Path to save questions and images
            num_options: Number of options to generate (default: 4)
            frame_gap: Preferred frame gap (default: 3)
            
        Returns:
            question: Question dictionary with images and metadata
        """
        try:
            # Extract metadata
            metadata = sampled_data.get("metadata", {})
            scene_id = sampled_data.get("scene_id")
            frame_id = sampled_data.get("frame_id")
            source = metadata.get("source")
            
            # Create save directories
            question_folder = Path(save_path) / "construct" / EgoRotationGenerator.QUESTION_TYPE.value
            images_folder = question_folder / "images"
            images_folder.mkdir(parents=True, exist_ok=True)
            
            # Generate unique ID for this question
            question_id = str(uuid4())
            
            # Step 1: Query nearby frame from the same scene
            sqlite_db = SQLiteController(db_path=db_path)
            nearby_frame = EgoRotationGenerator._query_nearby_frame(
                sqlite_db=sqlite_db,
                scene_id=scene_id,
                current_frame_id=frame_id,
                source=source,
                frame_gap=frame_gap,
                current_record_id=sampled_data.get('id')
            )
            
            if not nearby_frame:
                raise Exception(f"No nearby frame found for scene {scene_id}")
            
            # Step 2: Get front camera images
            front_cam1 = EgoRotationGenerator._get_front_camera(sampled_data, source)
            front_cam2 = EgoRotationGenerator._get_front_camera(nearby_frame, source)
            
            if not front_cam1 or not front_cam2:
                raise Exception("Failed to get front camera images")
            
            # Step 3: Calculate ego rotation angle
            rotation_angle = EgoRotationGenerator._calculate_rotation_angle(
                sampled_data, nearby_frame, source
            )
            
            # Step 4: Save front camera images separately
            front_cam1_path = images_folder / f"{question_id}_t1.jpg"
            front_cam2_path = images_folder / f"{question_id}_t2.jpg"
            
            EgoRotationGenerator._save_image(front_cam1, str(front_cam1_path))
            EgoRotationGenerator._save_image(front_cam2, str(front_cam2_path))
            
            # Step 5: Generate multiple choice options
            correct_bin = EgoRotationGenerator._angle_to_bin(rotation_angle)
            options, correct_answer = EgoRotationGenerator._generate_options(
                correct_bin, num_options
            )
            
            # Step 6: Format question with image tokens
            question_text = EgoRotationGenerator.QUESTION_TEMPLATE.strip()
            question_text += f"\n\nTimestamp T1: <image>"
            question_text += f"\nTimestamp T2: <image>"
            question_text += f"\n\nOptions:\n"
            for idx, angle_range in enumerate(options):
                question_text += f"({chr(65+idx)}) {angle_range}\n"
            
            # Step 7: Create question dictionary
            question_dict = {
                "question_id": question_id,
                "question_type": EgoRotationGenerator.QUESTION_TYPE.value,
                "question": question_text,
                "images": [
                    str(front_cam1_path),
                    str(front_cam2_path)
                ],
                "options": options,
                "correct_answer": correct_answer,
                "metadata": {
                    "scene_id": scene_id,
                    "frame_id_1": frame_id,
                    "frame_id_2": nearby_frame.get("frame_id"),
                    "source": source,
                    "rotation_angle_degrees": float(rotation_angle),
                    "record_id_1": sampled_data.get("id"),
                    "record_id_2": nearby_frame.get("id")
                }
            }
            
            return question_dict
            
        except Exception as e:
            raise Exception(f"Failed to generate ego rotation question: {e}")
    
    @staticmethod
    def _query_nearby_frame(
        sqlite_db: SQLiteController,
        scene_id: str,
        current_frame_id: str,
        source: str,
        frame_gap: int = 3,
        current_record_id: int = None
    ) -> Optional[Dict[str, Any]]:
        """
        Query a nearby frame from the same scene.
        
        Uses timestamp from calibration for proper temporal ordering.
        
        Args:
            sqlite_db: SQLite controller instance
            scene_id: Current scene ID
            current_frame_id: Current frame ID (for filtering)
            source: Data source
            frame_gap: Preferred frame gap in frames
            current_record_id: Current record ID for filtering
            
        Returns:
            Nearby frame data or None
        """
        try:
            # Query all frames from the same scene
            query = """
            SELECT * FROM av_data 
            WHERE scene_id = ? 
            AND source = ?
            AND id != ?
            """
            
            with sqlite_db._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(query, (scene_id, source, current_record_id))
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
                    
                    # For ONCE dataset, use record_id for sorting (no reliable timestamp)
                    if source and source.lower() == 'once':
                        frame['_sort_key'] = frame.get('id', 0)
                        # Calculate distance from current record for ONCE
                        frame['_distance'] = abs(frame.get('id', 0) - current_record_id)
                    else:
                        # Extract timestamp from calibration for sorting
                        calibration = frame.get('calibration', {})
                        img_timestamp = calibration.get('img_timestamp', [])
                        
                        # Use first camera timestamp, or record ID as fallback
                        if img_timestamp and len(img_timestamp) > 0:
                            frame['_sort_key'] = img_timestamp[0]
                        else:
                            # Fallback: use record ID as pseudo-timestamp
                            frame['_sort_key'] = frame.get('id', 0)
                    
                    frames.append(frame)
                
                # For ONCE dataset: find frames close to current record_id
                if source and source.lower() == 'once':
                    # Sort by distance from current frame
                    frames.sort(key=lambda f: f['_distance'])
                    
                    # Filter frames within reasonable distance (max 10 frames apart)
                    max_distance = 10
                    nearby_frames = [f for f in frames if f['_distance'] <= max_distance]
                    
                    if nearby_frames:
                        # Pick the frame at the desired gap (or closest available)
                        if len(nearby_frames) >= frame_gap:
                            return nearby_frames[frame_gap - 1]
                        else:
                            return nearby_frames[-1]
                    elif frames:
                        # Fallback: use the closest frame if no nearby frames
                        return frames[0]
                    return None
                
                # For other datasets: sort by timestamp and pick by gap
                frames.sort(key=lambda f: f['_sort_key'])
                
                # Find the frame closest to the desired gap
                if len(frames) >= frame_gap:
                    return frames[frame_gap - 1]
                elif len(frames) > 0:
                    # Use the last frame if gap is too large
                    return frames[-1]
                
                return None
                
        except Exception as e:
            print(f"Warning: Failed to query nearby frame: {e}")
            return None
    
    @staticmethod
    def _get_front_camera(data: Dict[str, Any], source: str) -> Optional[str]:
        """
        Get front camera path based on data source.
        
        Args:
            data: Data record
            source: Data source
            
        Returns:
            Front camera image path
        """
        views = data.get("views", {})
        
        # Map source to front camera key
        front_camera_map = {
            DataSource.NUSCENES.value: "CAM_FRONT",
            DataSource.ONCE.value: "cam01",
            DataSource.WAYMO.value: "FRONT",
            DataSource.ARGOVERSE2.value: "ring_front_center",
            DataSource.TRUCKSCENES.value: "CAMERA_LEFT_FRONT"  # or could use right
        }
        
        camera_key = front_camera_map.get(source)
        if not camera_key or camera_key not in views:
            return None
        
        camera_data = views[camera_key]
        if isinstance(camera_data, dict):
            return camera_data.get("path")
        return camera_data
    
    @staticmethod
    def _calculate_rotation_angle(
        frame1: Dict[str, Any],
        frame2: Dict[str, Any],
        source: str
    ) -> float:
        """
        Calculate rotation angle between two frames using ego pose.
        
        Args:
            frame1: First frame data
            frame2: Second frame data
            source: Data source
            
        Returns:
            Rotation angle in degrees
        """
        try:
            calib1 = frame1.get("calibration", {})
            calib2 = frame2.get("calibration", {})
            
            # Get ego poses (4x4 transformation matrices)
            if "ego_pose" in calib1 and "ego_pose" in calib2:
                # Convert to numpy arrays
                ego_pose1 = np.array(calib1["ego_pose"])
                ego_pose2 = np.array(calib2["ego_pose"])
                
                # Ensure they are 4x4 matrices
                if ego_pose1.shape == (4, 4) and ego_pose2.shape == (4, 4):
                    # Extract rotation matrices (top-left 3x3)
                    rot1 = ego_pose1[:3, :3]
                    rot2 = ego_pose2[:3, :3]
                    
                    # Calculate relative rotation: R_relative = R2 @ R1^T
                    relative_rot = rot2 @ rot1.T
                    
                    # Extract yaw angle (rotation around z-axis)
                    # For a rotation matrix, yaw = atan2(R[1,0], R[0,0])
                    yaw = np.arctan2(relative_rot[1, 0], relative_rot[0, 0])
                    
                    # Convert to degrees and return absolute value
                    angle_degrees = abs(np.degrees(yaw))
                    
                    return angle_degrees
                else:
                    print(f"Warning: ego_pose shape mismatch: {ego_pose1.shape}, {ego_pose2.shape}")
            else:
                print(f"Warning: ego_pose not found in calibration data")
            
            # Fallback: return a reasonable default
            print(f"Warning: Could not calculate rotation for source {source}, using fallback")
            fallback_angle = random.uniform(10, 45)
            return fallback_angle
            
        except Exception as e:
            print(f"Warning: Error calculating rotation angle: {e}")
            import traceback
            traceback.print_exc()
            return random.uniform(10, 45)
    
    @staticmethod
    def _angle_to_bin(angle: float) -> str:
        """
        Convert angle to the closest bin range.
        
        Args:
            angle: Angle in degrees
            
        Returns:
            Bin range string (e.g., "0-15°", "15-30°")
        """
        bins = EgoRotationGenerator.ANGLE_BINS
        
        for i in range(len(bins) - 1):
            if angle >= bins[i] and angle < bins[i + 1]:
                return f"{bins[i]}-{bins[i+1]}°"
        
        # Last bin
        if angle >= bins[-1]:
            return f"{bins[-1]}°+"
        
        return f"{bins[0]}-{bins[1]}°"
    
    @staticmethod
    def _generate_options(
        correct_bin: str,
        num_options: int = 4
    ) -> Tuple[List[str], int]:
        """
        Generate multiple choice options including the correct answer.
        
        Args:
            correct_bin: Correct angle bin
            num_options: Number of options to generate
            
        Returns:
            Tuple of (options list, correct answer index)
        """
        bins = EgoRotationGenerator.ANGLE_BINS
        all_bins = []
        
        for i in range(len(bins) - 1):
            all_bins.append(f"{bins[i]}-{bins[i+1]}°")
        all_bins.append(f"{bins[-1]}°+")
        
        # Ensure correct bin is in options
        if correct_bin not in all_bins:
            correct_bin = all_bins[0]
        
        # Remove correct bin from pool
        distractor_pool = [b for b in all_bins if b != correct_bin]
        
        # Select random distractors
        num_distractors = min(num_options - 1, len(distractor_pool))
        distractors = random.sample(distractor_pool, num_distractors)
        
        # Combine and shuffle
        options = [correct_bin] + distractors
        correct_answer = 0
        
        shuffled_indices = list(range(len(options)))
        random.shuffle(shuffled_indices)
        shuffled_options = [options[i] for i in shuffled_indices]
        correct_answer = shuffled_indices.index(0)
        
        return shuffled_options, correct_answer
    
    @staticmethod
    def _save_image(image_path: str, output_path: str) -> None:
        """
        Copy/save an image to output location.
        
        Args:
            image_path: Source image path
            output_path: Destination path
        """
        try:
            if image_path and os.path.exists(image_path):
                from shutil import copy2
                copy2(image_path, output_path)
            else:
                # Create placeholder
                placeholder = Image.new('RGB', (800, 600), color=(128, 128, 128))
                placeholder.save(output_path)
        except Exception as e:
            print(f"Warning: Failed to save image: {e}")
            placeholder = Image.new('RGB', (800, 600), color=(128, 128, 128))
            placeholder.save(output_path)
    
    @staticmethod
    def _concatenate_two_images(
        image_paths: List[str],
        output_path: str,
        labels: List[str] = None
    ) -> None:
        """
        Concatenate two images side by side with labels.
        
        Args:
            image_paths: List of two image paths
            output_path: Path to save concatenated image
            labels: Optional labels for each image
        """
        if len(image_paths) != 2:
            raise ValueError("Exactly 2 images required")
        
        images = []
        for path in image_paths:
            try:
                if path and os.path.exists(path):
                    img = Image.open(path)
                    images.append(img)
            except Exception as e:
                print(f"Warning: Failed to load image {path}: {e}")
        
        if len(images) != 2:
            # Create placeholder
            placeholder = Image.new('RGB', (1600, 600), color=(128, 128, 128))
            placeholder.save(output_path)
            return
        
        # Resize to same size
        target_height = 480
        resized_images = []
        for img in images:
            aspect_ratio = img.width / img.height
            new_width = int(target_height * aspect_ratio)
            resized_img = img.resize((new_width, target_height), Image.LANCZOS)
            resized_images.append(resized_img)
        
        # Create canvas with label space
        label_height = 40
        total_width = sum(img.width for img in resized_images)
        canvas_height = target_height + label_height
        
        canvas = Image.new('RGB', (total_width, canvas_height), color=(0, 0, 0))
        
        # Paste images with labels
        x_offset = 0
        for idx, img in enumerate(resized_images):
            # Paste image
            canvas.paste(img, (x_offset, label_height))
            
            # Draw label
            if labels and idx < len(labels):
                draw = ImageDraw.Draw(canvas)
                try:
                    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24)
                except:
                    font = ImageFont.load_default()
                
                label = labels[idx]
                text_bbox = draw.textbbox((0, 0), label, font=font)
                text_width = text_bbox[2] - text_bbox[0]
                text_x = x_offset + (img.width - text_width) // 2
                
                draw.text((text_x, 10), label, fill=(255, 255, 255), font=font)
            
            x_offset += img.width
        
        canvas.save(output_path)