import os
import json
import random
from uuid import uuid4
from pathlib import Path
from typing import Dict, Any, List, Tuple
from PIL import Image, ImageDraw, ImageFont

from sources.db.sql import SQLiteController
from sources.question_generation.utils import render_bev_map
from sources.data_collection.enum.data_sources import DataSource
from sources.question_generation.enum import ConstructionQuestionType


class PerspectiveCameraMatchingGenerator:
    QUESTION_TYPE = ConstructionQuestionType.PERSPECTIVE_CAMERA_MATCHING
    QUESTION_TEMPLATE = """
Given the top-down perspective map of a driving scene, 
identify which front camera view corresponds to this map."""
    
    @staticmethod
    def generate_question(
        sampled_data: Dict[str, Any],
        db_path: str,
        save_path: str,
        dataset_configs: Dict[str, Dict[str, str]] = None,
        num_options: int = 4
    ) -> Dict[str, Any]:
        """
        Generate a perspective-to-camera matching question.
        
        Args:
            sampled_data: Sampled data record from database
            db_path: Path to the database
            save_path: Path to save questions and images
            dataset_configs: Dictionary with dataset paths for initialization
            num_options: Number of options to generate (default: 4)
            
        Returns:
            question: Question dictionary with images and metadata
        """
        try:
            # Extract metadata
            metadata = sampled_data.get("metadata", {})
            scene_type = metadata.get("scene_type", "unknown")
            source = metadata.get("source")
            scene_id = sampled_data.get("scene_id")
            frame_id = sampled_data.get("frame_id")
            
            # Create save directories
            question_folder = Path(save_path) / "construct" / PerspectiveCameraMatchingGenerator.QUESTION_TYPE.value
            images_folder = question_folder / "images"
            images_folder.mkdir(parents=True, exist_ok=True)
            
            # Generate unique ID for this question
            question_id = str(uuid4())
            
            # Step 1: Render BEV map for the current frame
            bev_map_path = images_folder / f"{question_id}_bev_map.png"
            render_bev_map(
                data=sampled_data,
                output_path=str(bev_map_path),
                dataset_configs=dataset_configs
            )
            
            # Step 2: Get front camera view for correct answer
            correct_front_view = PerspectiveCameraMatchingGenerator._get_front_camera(
                sampled_data, source
            )
            
            if not correct_front_view:
                raise Exception("Failed to get front camera view")
            
            # Step 3: Query similar scenes for distractor options
            sqlite_db = SQLiteController(db_path=db_path)
            similar_records = PerspectiveCameraMatchingGenerator._query_similar_scenes(
                sqlite_db=sqlite_db,
                scene_type=scene_type,
                source=source,
                exclude_id=sampled_data.get("id"),
                limit=num_options * 2
            )
            
            # Collect distractor front camera views
            distractor_views = []
            for record in similar_records:
                front_view = PerspectiveCameraMatchingGenerator._get_front_camera(record, source)
                if front_view and os.path.exists(front_view):
                    distractor_views.append(front_view)
                if len(distractor_views) >= num_options - 1:
                    break
            
            # Ensure we have enough distractors
            if len(distractor_views) < num_options - 1:
                raise Exception(f"Not enough distractor views: {len(distractor_views)}")
            
            # Step 4: Save front camera views as separate option images
            camera_option_paths = []
            
            # Save correct camera view
            correct_cam_path = images_folder / f"{question_id}_camera_option_0.jpg"
            PerspectiveCameraMatchingGenerator._save_image(correct_front_view, str(correct_cam_path))
            camera_option_paths.append(str(correct_cam_path))
            
            # Save distractor camera views
            for idx, distractor_view in enumerate(distractor_views[:num_options - 1]):
                distractor_cam_path = images_folder / f"{question_id}_camera_option_{idx+1}.jpg"
                PerspectiveCameraMatchingGenerator._save_image(distractor_view, str(distractor_cam_path))
                camera_option_paths.append(str(distractor_cam_path))
            
            # Step 5: Shuffle camera options
            correct_answer = 0
            shuffled_indices = list(range(len(camera_option_paths)))
            random.shuffle(shuffled_indices)
            shuffled_camera_paths = [camera_option_paths[i] for i in shuffled_indices]
            correct_answer = shuffled_indices.index(0)
            
            # Step 6: Format question with image tokens
            question_text = PerspectiveCameraMatchingGenerator.QUESTION_TEMPLATE.strip()
            question_text += f"\n\nBEV Map: <image>"
            question_text += f"\n\nCamera Options:"
            for idx in range(len(shuffled_camera_paths)):
                question_text += f"\n({chr(65+idx)}) <image>"
            
            # Step 7: Create question dictionary
            question_dict = {
                "question_id": question_id,
                "question_type": PerspectiveCameraMatchingGenerator.QUESTION_TYPE.value,
                "question": question_text,
                "images": [str(bev_map_path)] + shuffled_camera_paths,
                "options": [f"Camera {chr(65+i)}" for i in range(len(shuffled_camera_paths))],
                "correct_answer": correct_answer,
                "metadata": {
                    "scene_id": scene_id,
                    "frame_id": frame_id,
                    "source": source,
                    "scene_type": scene_type,
                    "record_id": sampled_data.get("id")
                }
            }
            
            return question_dict
            
        except Exception as e:
            raise Exception(f"Failed to generate perspective-camera matching question: {e}")
    
    @staticmethod
    def _get_front_camera(data: Dict[str, Any], source: str) -> str:
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
            DataSource.TRUCKSCENES.value: "CAMERA_LEFT_FRONT"
        }
        
        camera_key = front_camera_map.get(source)
        if not camera_key or camera_key not in views:
            return None
        
        camera_data = views[camera_key]
        if isinstance(camera_data, dict):
            return camera_data.get("path")
        return camera_data
    
    @staticmethod
    def _query_similar_scenes(
        sqlite_db: SQLiteController,
        scene_type: str,
        source: str,
        exclude_id: int,
        limit: int
    ) -> List[Dict[str, Any]]:
        """
        Query database for similar scenes (same scene_type and source).
        
        Args:
            sqlite_db: SQLite controller instance
            scene_type: Scene type to match
            source: Data source to match
            exclude_id: Record ID to exclude
            limit: Maximum number of records
            
        Returns:
            List of similar scene records
        """
        try:
            query = """
            SELECT * FROM av_data 
            WHERE id != ? 
            AND source = ?
            AND json_extract(metadata, '$.scene_type') = ?
            ORDER BY RANDOM() 
            LIMIT ?
            """
            
            with sqlite_db._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(query, (exclude_id, source, scene_type, limit))
                rows = cursor.fetchall()
                
                results = []
                json_fields = ['views', 'lidar', 'annotation', 'calibration',
                               'spatial_relations', 'internal_interactions',
                               'external_interactions', 'metadata']
                
                for row in rows:
                    result = dict(row)
                    # Parse JSON fields
                    for field in json_fields:
                        if result.get(field):
                            try:
                                result[field] = json.loads(result[field])
                            except json.JSONDecodeError:
                                pass
                    results.append(result)
                
                return results
                
        except Exception as e:
            print(f"Warning: Failed to query similar scenes: {e}")
            return []
    
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