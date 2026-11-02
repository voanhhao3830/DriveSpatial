import json
import random
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from uuid import uuid4

from sources.db.sql import SQLiteController
from sources.question_generation.enum import ConstructionQuestionType
from sources.question_generation.utils import render_bev_map
from sources.question_generation.utils.visualization import concatenate_camera_views

logger = logging.getLogger(__name__)


class CognitiveMapGenerator:
    QUESTION_TYPE = ConstructionQuestionType.COGNITIVE_SCENE
    QUERY = "SELECT * FROM av_data WHERE metadata LIKE '%\"scene_type\": \"{scene_type}\"%' AND metadata LIKE '%\"source\": \"{source}\"%' ORDER BY RANDOM() LIMIT {limit};"
    QUESTION_TEMPLATE = """
Given the current driving scene, construct a bird's-eye-view (BEV) map of the scene and select the correct map.
"""
    
    # Class-level cache: {(source, scene_type): [list of record dicts]}
    _scene_cache: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    _cache_initialized: bool = False

    @classmethod
    def generate_question(
        cls,
        sampled_data: Dict[str, Any], 
        db_path: str, 
        save_path: str,
        dataset_configs: Dict[str, Dict[str, str]] = None,
        num_options: int = 4,
        auto_init_cache: bool = True
    ) -> Dict[str, Any]:
        """
        Generate a cognitive map question with BEV map options.
        
        Args:
            sampled_data: Sampled data record from database
            db_path: Path to the database
            save_path: Path to save questions and images
            dataset_configs: Dictionary with dataset paths for initialization
            num_options: Number of options to generate (default: 4)
            auto_init_cache: If True, auto-initialize cache on first call (default: True).
                             For best performance, call preload_scene_cache() before generating
                             multiple questions.
            
        Returns:
            question: Question dictionary with images and metadata
        """
        # Extract metadata
        metadata = sampled_data.get("metadata", {})
        scene_type = metadata.get("scene_type", "unknown")
        source = metadata.get("source", "unknown")
        
        # Create database connection
        sqlite_db = SQLiteController(db_path=db_path)
        
        # Auto-initialize cache if not done and enabled
        if auto_init_cache and not cls._cache_initialized:
            logger.info("Auto-initializing scene cache (first call)...")
            cls.preload_scene_cache(sqlite_db, sources=[source])
        
        # Create save directories
        question_folder = Path(save_path) / "construct" / cls.QUESTION_TYPE.value
        images_folder = question_folder / "images"
        images_folder.mkdir(parents=True, exist_ok=True)
        
        # Generate unique ID for this question
        question_id = str(uuid4())
        
        # Step 1: Get camera views and save concatenated image
        views = sampled_data.get("views", {})
        concat_image_path = images_folder / f"{question_id}_cameras.jpg"
        concatenate_camera_views(
            camera_views=views,
            output_path=str(concat_image_path),
            source=source,
            add_labels=True
        )
        
        # Step 2: Render BEV map for the correct answer
        correct_map_path = images_folder / f"{question_id}_map_correct.png"
        render_bev_map(
            data=sampled_data,
            output_path=str(correct_map_path),
            dataset_configs=dataset_configs
        )
        
        # Step 3: Get candidate records for distractors (fast with cache)
        similar_records = cls._query_similar_scenes(
            sqlite_db=sqlite_db,
            scene_type=scene_type,
            source=source,
            exclude_id=sampled_data.get("id"),
            limit=num_options * 2  # Get more candidates
        )
        
        # Step 4: Save BEV maps as separate option images
        option_images = []
        
        # Save correct BEV map as option 0
        correct_option_path = images_folder / f"{question_id}_option_0.png"
        render_bev_map(
            data=sampled_data,
            output_path=str(correct_option_path),
            dataset_configs=dataset_configs
        )
        option_images.append(str(correct_option_path))
        
        # Save distractor BEV maps
        for idx, distractor_data in enumerate(similar_records[:num_options - 1]):
            distractor_map_path = images_folder / f"{question_id}_option_{idx+1}.png"
            render_bev_map(
                data=distractor_data,
                output_path=str(distractor_map_path),
                dataset_configs=dataset_configs
            )
            option_images.append(str(distractor_map_path))
        
        # Step 5: Shuffle options
        shuffled_indices = list(range(len(option_images)))
        random.shuffle(shuffled_indices)
        shuffled_option_images = [option_images[i] for i in shuffled_indices]
        correct_answer = shuffled_indices.index(0)
        
        # Step 6: Format question with image tokens
        question_text = cls.QUESTION_TEMPLATE
        question_text += f"\n\nCamera views: <image>"
        for idx in range(len(shuffled_option_images)):
            question_text += f"\nOption {chr(65+idx)}: <image>"

        # Step 7: Create question dictionary
        question_dict = {
            "question_id": question_id,
            "question_type": cls.QUESTION_TYPE.value,
            "question": question_text,
            "images": [str(concat_image_path)] + shuffled_option_images,
            "correct_answer": correct_answer,
            "metadata": {
                "scene_id": sampled_data.get("scene_id"),
                "frame_id": sampled_data.get("frame_id"),
                "source": source,
                "scene_type": scene_type,
                "record_id": sampled_data.get("id")
            }
        }
    
        return question_dict
    
    @classmethod
    def preload_scene_cache(
        cls,
        sqlite_db: SQLiteController,
        sources: Optional[List[str]] = None
    ) -> None:
        """
        Preload all scenes into memory cache for fast random sampling.
        
        Call this once before generating multiple questions to avoid slow
        repeated database queries with ORDER BY RANDOM().
        
        Args:
            sqlite_db: Database controller
            sources: Optional list of sources to preload. If None, loads all sources.
        """
        logger.info("Preloading scene cache from database...")
        
        try:
            # Build query based on sources filter
            if sources:
                placeholders = ','.join(['?' for _ in sources])
                query = f"SELECT * FROM av_data WHERE source IN ({placeholders})"
                params = tuple(sources)
            else:
                query = "SELECT * FROM av_data"
                params = ()
            
            with sqlite_db._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(query, params)
                rows = cursor.fetchall()
                
                json_fields = ['views', 'lidar', 'annotation', 'calibration',
                               'spatial_relations', 'internal_interactions',
                               'external_interactions', 'metadata']
                
                # Group records by (source, scene_type)
                temp_cache: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
                
                for row in rows:
                    result = dict(row)
                    
                    # Parse JSON fields
                    for field in json_fields:
                        if result.get(field):
                            try:
                                result[field] = json.loads(result[field])
                            except json.JSONDecodeError:
                                pass
                    
                    # Extract source and scene_type for cache key
                    metadata = result.get('metadata', {})
                    if isinstance(metadata, str):
                        try:
                            metadata = json.loads(metadata)
                        except json.JSONDecodeError:
                            metadata = {}
                    
                    source = result.get('source') or metadata.get('source', 'unknown')
                    scene_type = metadata.get('scene_type', 'unknown')
                    
                    cache_key = (source, scene_type)
                    if cache_key not in temp_cache:
                        temp_cache[cache_key] = []
                    temp_cache[cache_key].append(result)
                
                # Update class cache
                cls._scene_cache = temp_cache
                cls._cache_initialized = True
                
                total_records = sum(len(v) for v in cls._scene_cache.values())
                logger.info(
                    f"Scene cache loaded: {total_records} records across "
                    f"{len(cls._scene_cache)} (source, scene_type) groups"
                )
                
                # Log breakdown
                for key, records in sorted(cls._scene_cache.items()):
                    logger.debug(f"  {key}: {len(records)} records")
                    
        except Exception as e:
            logger.error(f"Failed to preload scene cache: {e}", exc_info=True)
            cls._scene_cache = {}
            cls._cache_initialized = False
    
    @classmethod
    def clear_cache(cls) -> None:
        """Clear the in-memory scene cache."""
        cls._scene_cache = {}
        cls._cache_initialized = False
        logger.info("Scene cache cleared")
    
    @classmethod
    def _query_similar_scenes(
        cls,
        sqlite_db: SQLiteController,
        scene_type: str,
        source: str,
        exclude_id: int,
        limit: int
    ) -> List[Dict[str, Any]]:
        """
        Get similar scenes (same scene_type and source) for distractor generation.
        
        Uses in-memory cache if available, otherwise falls back to database query.
        The cache-based approach is much faster for large databases.
        
        Args:
            sqlite_db: Database controller
            scene_type: Scene type to match
            source: Data source to match
            exclude_id: Record ID to exclude (the correct answer)
            limit: Maximum number of records to return
            
        Returns:
            List of similar scene records
        """
        cache_key = (source, scene_type)
        
        # Try to use cache first
        if cls._cache_initialized and cache_key in cls._scene_cache:
            cached_records = cls._scene_cache[cache_key]
            
            # Filter out the excluded record and sample randomly
            candidates = [r for r in cached_records if r.get('id') != exclude_id]
            
            if len(candidates) <= limit:
                return candidates
            else:
                return random.sample(candidates, limit)
        
        # Fallback to database query if cache not available
        logger.debug(f"Cache miss for {cache_key}, falling back to database query")
        
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
            logger.warning(f"Failed to query similar scenes: {e}")
            return []


    @staticmethod
    def save_questions_to_file(
        questions: List[Dict[str, Any]], 
        save_path: str, 
        filename: str = "questions.json"
    ) -> None:
        """
        Save all questions to a single JSON file.
        
        Args:
            questions: List of question dictionaries
            save_path: Base path to save the file
            filename: Name of the JSON file
        """
        try:
            question_folder = Path(save_path) / "construct" / CognitiveMapGenerator.QUESTION_TYPE.value
            question_folder.mkdir(parents=True, exist_ok=True)
            
            questions_file = question_folder / filename
            
            output = {
                "total_questions": len(questions),
                "question_type": CognitiveMapGenerator.QUESTION_TYPE.value,
                "questions": questions
            }
            
            with open(questions_file, "w") as f:
                json.dump(output, f, indent=2)
            
            print(f"Saved {len(questions)} questions to {questions_file}")
            
        except Exception as e:
            print(f"Error saving questions to file: {e}")
            raise