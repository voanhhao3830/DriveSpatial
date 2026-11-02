"""
FastAPI application for managing AV (Autonomous Vehicle) data.

This application provides endpoints to:
- Query and retrieve AV data with filtering and pagination
- Update AV data records
"""

import logging
import traceback
from typing import Optional, List, Dict, Any
from datetime import datetime
from pathlib import Path
import numpy as np
import os
import io
import base64

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
import sqlite3
import json

# Check for requests library (for HPC fetching)
try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Optional imports for AV2 feather support
try:
    import pandas as pd
    from pyarrow import feather
    FEATHER_SUPPORT = True
    logger.info("AV2 feather file support enabled (pandas and pyarrow available)")
except ImportError:
    FEATHER_SUPPORT = False
    logger.warning("pandas or pyarrow not installed. AV2 feather file support disabled.")
# Database configuration
DATABASE_PATH = "./data/output.db"
# QUESTION_DATABASE_PATH = "./data/questions-v8/question-v2.db" # NOTE: this is the large question database
QUESTION_DATABASE_PATH = "./data/questions-v8-update-neurips/question.db"
EVALUATION_DATABASE_PATH = "./data/questions-v8-update-neurips/drivespatial-small-human.db"

# Initialize FastAPI app
app = FastAPI(
    title="DriveSpatial AV Data API",
    description="API for querying and updating autonomous vehicle dataset",
    version="1.0.0"
)

# Add CORS middleware to allow requests from the frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "*",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Pydantic Schemas
class AVDataBase(BaseModel):
    """Base schema for AV data"""
    scene_id: str = Field(..., description="Scene identifier")
    frame_id: str = Field(..., description="Frame identifier")
    source: str = Field(..., description="Data source (e.g., nuScenes, Argoverse2)")
    views: Dict[str, Any] = Field(..., description="Images path data")
    lidar: Dict[str, Any] = Field(..., description="Lidar path data")
    annotation: Dict[str, Any] = Field(..., description="Annotation data")
    calibration: Dict[str, Any] = Field(..., description="Calibration data")
    spatial_relations: List[Any] = Field(..., description="Spatial relations data (list)")
    internal_interactions: List[Any] = Field(..., description="Internal interactions data (list)")
    external_interactions: List[Any] = Field(..., description="External interactions data (list)")
    occlusions: List[Any] = Field(default=[], description="Occlusions/visibility data (list)")
    metadata: Dict[str, Any] = Field(..., description="Metadata")

    class Config:
        schema_extra = {
            "example": {
                "scene_id": "scene_001",
                "frame_id": "frame_001",
                "source": "nuScenes",
                "views": {"front": "/path/to/front.jpg"},
                "lidar": {"points": "/path/to/lidar.pcd"},
                "annotation": {"objects": []},
                "calibration": {"camera": {}},
                "spatial_relations": [],
                "internal_interactions": [],
                "external_interactions": [],
                "occlusions": [],
                "metadata": {"timestamp": "2024-01-01"}
            }
        }


class AVDataResponse(AVDataBase):
    """Response schema with additional fields"""
    id: int
    verified_metadata: int
    verified_spatial_relations: int
    verified_internal_interactions: int
    verified_external_interactions: int
    updated_metadata: Any
    updated_spatial_relations: Any
    updated_internal_interactions: Any
    updated_external_interactions: Any
    updated_occlusions: Optional[Any] = None
    verify_by: str
    created_at: str
    updated_at: str


class AVDataUpdate(BaseModel):
    """Schema for updating AV data"""
    scene_id: Optional[str] = None
    frame_id: Optional[str] = None
    source: Optional[str] = None
    views: Optional[Dict[str, Any]] = None
    lidar: Optional[Dict[str, Any]] = None
    annotation: Optional[Dict[str, Any]] = None
    calibration: Optional[Dict[str, Any]] = None
    spatial_relations: Optional[List[Any]] = None
    internal_interactions: Optional[List[Any]] = None
    external_interactions: Optional[List[Any]] = None
    occlusions: Optional[List[Any]] = None
    metadata: Optional[Dict[str, Any]] = None
    verified_metadata: Optional[int] = Field(None, ge=0, le=1)
    verified_spatial_relations: Optional[int] = Field(None, ge=0, le=1)
    verified_internal_interactions: Optional[int] = Field(None, ge=0, le=1)
    verified_external_interactions: Optional[int] = Field(None, ge=0, le=1)
    updated_metadata: Optional[Any] = None
    updated_spatial_relations: Optional[Any] = None
    updated_internal_interactions: Optional[Any] = None
    updated_external_interactions: Optional[Any] = None
    updated_occlusions: Optional[Any] = None
    verify_by: Optional[str] = None


class QueryResponse(BaseModel):
    """Response schema for query endpoint"""
    total: int = Field(..., description="Total number of records matching filter")
    count: int = Field(..., description="Number of records returned")
    data: List[AVDataResponse]


class UpdateInternalInteractionsBody(BaseModel):
    """Request body for PUT /api/v1/data/{record_id}/internal-interactions"""
    internal_interactions: List[Dict[str, Any]] = Field(..., description="Complete internal interactions array to save")
    verify_by: Optional[str] = Field(None, description="Verifier identifier; when set, marks frame as verified")


class UpdateExternalInteractionsBody(BaseModel):
    """Request body for PUT /api/v1/data/{record_id}/external-interactions
    
    Note: Each interaction has structure:
    {
      "source_object": { "track_id": str, "label": str },
      "target_object": { "track_id": str, "label": str },
      "interaction_type": str,  # Single string value
      "details": { ... }  # Optional metadata
    }
    """
    external_interactions: List[Dict[str, Any]] = Field(..., description="Complete external interactions array to save")
    verify_by: Optional[str] = Field(None, description="Verifier identifier; when set, marks frame as verified")


class LogTimeSpentBody(BaseModel):
    """Request body for PUT /api/v1/data/{record_id}/log-time"""
    mode: str = Field(..., description="Annotation mode: 'metadata', 'spatial', 'internal', 'external', or 'all'")
    time_spent: float = Field(..., description="Time spent in seconds")


# Database helper functions
def get_db_connection() -> sqlite3.Connection:
    """
    Create and return a database connection.
    
    Returns:
        sqlite3.Connection: Database connection
    """
    try:
        conn = sqlite3.connect(DATABASE_PATH)
        conn.row_factory = sqlite3.Row
        logger.info("Database connection established")
        return conn
    except Exception as e:
        logger.error(f"Failed to connect to database: {str(e)}")
        logger.error(traceback.format_exc())
        raise


def row_to_dict(row: sqlite3.Row) -> Dict[str, Any]:
    """
    Convert SQLite row to dictionary with JSON parsing.
    
    Args:
        row: SQLite row object
        
    Returns:
        Dict containing row data with parsed JSON fields
    """
    data = dict(row)
    
    # Parse JSON fields
    json_fields = [
        'views', 'lidar', 'annotation', 'calibration',
        'spatial_relations', 'internal_interactions',
        'external_interactions', 'occlusions', 'metadata',
        'updated_metadata', 'updated_spatial_relations',
        'updated_internal_interactions', 'updated_external_interactions',
        'updated_occlusions'
    ]
    
    interaction_field_names = [
        'spatial_relations', 'internal_interactions', 'external_interactions', 'occlusions',
        'updated_spatial_relations', 'updated_internal_interactions', 'updated_external_interactions', 'updated_occlusions'
    ]
    
    for field in json_fields:
        if field in data:
            if data[field]:
                try:
                    data[field] = json.loads(data[field])
                    # After parsing, ensure interaction fields are lists (not empty dicts)
                    if field in interaction_field_names:
                        if not isinstance(data[field], list):
                            # logger.warning(f"Record {data.get('id')}: Parsed {field} is not a list (type: {type(data[field]).__name__}), converting to empty list")
                            data[field] = []
                except json.JSONDecodeError:
                    logger.warning(f"Failed to parse JSON field: {field}")
                    # Use appropriate default based on field type
                    if field in interaction_field_names:
                        data[field] = []
                    else:
                        data[field] = {}
            else:
                # Handle empty/None values - set default
                if field in interaction_field_names:
                    data[field] = []
                else:
                    data[field] = {} if field != 'annotation' else None
    
    # Sanitize invalid float values (NaN, Inf) that are not JSON compliant
    def sanitize_floats(obj):
        """Replace NaN and Inf with None recursively"""
        if isinstance(obj, float):
            if obj != obj or obj == float('inf') or obj == float('-inf'):  # NaN or Inf
                return None
            return obj
        elif isinstance(obj, dict):
            return {k: sanitize_floats(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [sanitize_floats(item) for item in obj]
        return obj
    
    # Apply sanitization to all parsed JSON fields
    for field in json_fields:
        if field in data and data[field]:
            data[field] = sanitize_floats(data[field])
    
    # Prioritize updated fields over original fields
    if data.get('updated_metadata'):
        logger.debug(f"Record {data.get('id')}: Using updated_metadata")
        data['metadata'] = data['updated_metadata']
    
    if data.get('updated_spatial_relations'):
        logger.debug(f"Record {data.get('id')}: Using updated_spatial_relations")
        data['spatial_relations'] = data['updated_spatial_relations']
    else:
        logger.debug(f"Record {data.get('id')}: Using original spatial_relations")
    
    if data.get('updated_internal_interactions'):
        data['internal_interactions'] = data['updated_internal_interactions']
    if data.get('updated_external_interactions'):
        data['external_interactions'] = data['updated_external_interactions']
        logger.debug(f"Record {data.get('id')}: Using updated_external_interactions, type: {type(data['external_interactions'])}, sample: {data['external_interactions'][:1] if isinstance(data['external_interactions'], list) else 'not a list'}")
    elif data.get('external_interactions'):
        logger.debug(f"Record {data.get('id')}: Using original external_interactions, type: {type(data['external_interactions'])}, sample: {data['external_interactions'][:1] if isinstance(data['external_interactions'], list) else 'not a list'}")
    
    if data.get('updated_occlusions'):
        data['occlusions'] = data['updated_occlusions']
    
    # Final safety check: ensure all interaction fields are lists after prioritization
    final_interaction_fields = ['spatial_relations', 'internal_interactions', 'external_interactions', 'occlusions']
    for field in final_interaction_fields:
        if field not in data:
            # Field missing from database - set default
            data[field] = []
        elif not isinstance(data[field], list):
            logger.warning(f"Record {data.get('id')}: Final {field} is not a list (type: {type(data[field]).__name__}), converting to empty list")
            data[field] = []
    
    return data


# API Endpoints
@app.get("/", tags=["Health"])
async def root() -> Dict[str, str]:
    """Health check endpoint"""
    return {
        "status": "healthy",
        "message": "DriveSpatial AV Data API is running",
        "timestamp": datetime.now().isoformat()
    }


@app.get("/api/v1/data", response_model=QueryResponse, tags=["Data"])
async def get_data(
    top_k: int = Query(10, ge=1, le=1000, description="Number of records to return"),
    scene_id: Optional[str] = Query(None, description="Filter by scene ID"),
    frame_id: Optional[str] = Query(None, description="Filter by frame ID"),
    source: List[str] = Query(default=[], description="Filter by source(s) - can specify multiple"),
    verified_metadata: Optional[int] = Query(None, ge=0, le=1, description="Filter by metadata verification status"),
    verified_spatial_relations: Optional[int] = Query(None, ge=0, le=1, description="Filter by spatial relations verification status"),
    verified_internal_interactions: Optional[int] = Query(None, ge=0, le=1, description="Filter by internal interactions verification status"),
    verified_external_interactions: Optional[int] = Query(None, ge=0, le=1, description="Filter by external interactions verification status"),
    verify_by: Optional[str] = Query(None, description="Filter by verifier"),
    offset: int = Query(0, ge=0, description="Offset for pagination"),
    record_id: Optional[int] = Query(None, description="Get specific record by ID (instant)"),
    next_id: Optional[int] = Query(None, description="Get next record after this ID (instant)"),
    prev_id: Optional[int] = Query(None, description="Get previous record before this ID (instant)")
) -> QueryResponse:
    """
    Get top K rows from the database with optional filtering.
    OPTIMIZED: Use record_id/next_id/prev_id parameters for instant navigation.
    
    Args:
        top_k: Maximum number of records to return
        scene_id: Optional scene ID filter
        frame_id: Optional frame ID filter
        source: Optional source filter
        verified_metadata: Optional metadata verification filter
        verified_spatial_relations: Optional spatial relations verification filter
        verified_internal_interactions: Optional internal interactions verification filter
        verified_external_interactions: Optional external interactions verification filter
        verify_by: Optional verifier filter
        offset: Pagination offset (slow for large values)
        record_id: Get specific record by ID (instant)
        next_id: Get next record after this ID (instant)
        prev_id: Get previous record before this ID (instant)
        
    Returns:
        QueryResponse containing matching records
    """
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Build WHERE clause for filters (used by all query types)
        filter_clauses = []
        filter_params = []
        
        if scene_id:
            filter_clauses.append("scene_id = ?")
            filter_params.append(scene_id)
        
        if frame_id:
            filter_clauses.append("frame_id = ?")
            filter_params.append(frame_id)
        print("source", source)
        if source and len(source) > 0:
            source_placeholders = " OR ".join(["source = ?"] * len(source))
            filter_clauses.append(f"({source_placeholders})")
            filter_params.extend(source)
            logger.info(f"Source filter: sources={source}, placeholders={source_placeholders}")
        
        if verified_metadata is not None:
            filter_clauses.append("verified_metadata = ?")
            filter_params.append(verified_metadata)
        
        if verified_spatial_relations is not None:
            filter_clauses.append("verified_spatial_relations = ?")
            filter_params.append(verified_spatial_relations)
        
        if verified_internal_interactions is not None:
            filter_clauses.append("verified_internal_interactions = ?")
            filter_params.append(verified_internal_interactions)
        
        if verified_external_interactions is not None:
            filter_clauses.append("verified_external_interactions = ?")
            filter_params.append(verified_external_interactions)
        
        if verify_by:
            filter_clauses.append("verify_by = ?")
            filter_params.append(verify_by)
        
        filter_sql = " AND ".join(filter_clauses) if filter_clauses else "1=1"
        logger.info(f"Filter SQL: {filter_sql}")
        logger.info(f"Filter params: {filter_params}")
        
        # Get true total count for proper UI display (respects filters)
        count_query = f"SELECT COUNT(*) as total FROM av_data WHERE {filter_sql}"
        cursor.execute(count_query, filter_params)
        total_row = cursor.fetchone()
        total_count = total_row['total']
        logger.info(f"Total count with filters: {total_count}")
        
        # Fast path: Direct ID lookup
        if record_id is not None:
            logger.info(f"Direct ID query: record_id={record_id}, filters={filter_sql}")
            
            # Direct PRIMARY KEY lookup with filters
            query = f"SELECT * FROM av_data WHERE id = ? AND {filter_sql}"
            cursor.execute(query, [record_id] + filter_params)
            row = cursor.fetchone()
            
            if not row:
                conn.close()
                logger.warning(f"Record not found: {record_id} with filters")
                return QueryResponse(total=total_count, count=0, data=[])
            
            data = row_to_dict(row)
            conn.close()
            
            logger.info(f"Direct ID query successful: found record {record_id}, total: {total_count}")
            return QueryResponse(total=total_count, count=1, data=[data])
        
        # Fast path: Next record by ID (with filters)
        if next_id is not None:
            logger.info(f"Next record query: after ID={next_id}, filters={filter_sql}")
            
            # Get next record with filters
            query = f"""
                SELECT * FROM av_data 
                WHERE id > ? AND {filter_sql}
                ORDER BY id ASC 
                LIMIT 1
            """
            cursor.execute(query, [next_id] + filter_params)
            row = cursor.fetchone()
            
            if not row:
                conn.close()
                logger.warning(f"No next record after ID: {next_id} with filters")
                return QueryResponse(total=total_count, count=0, data=[])
            
            data = row_to_dict(row)
            conn.close()
            
            logger.info(f"Next record query successful: found record {data['id']} after {next_id}")
            return QueryResponse(total=total_count, count=1, data=[data])
        
        # Fast path: Previous record by ID (with filters)
        if prev_id is not None:
            logger.info(f"Previous record query: before ID={prev_id}, filters={filter_sql}")
            
            # Get previous record with filters
            query = f"""
                SELECT * FROM av_data 
                WHERE id < ? AND {filter_sql}
                ORDER BY id DESC 
                LIMIT 1
            """
            cursor.execute(query, [prev_id] + filter_params)
            row = cursor.fetchone()
            
            if not row:
                conn.close()
                logger.warning(f"No previous record before ID: {prev_id} with filters")
                return QueryResponse(total=total_count, count=0, data=[])
            
            data = row_to_dict(row)
            conn.close()
            
            logger.info(f"Previous record query successful: found record {data['id']} before {prev_id}")
            return QueryResponse(total=total_count, count=1, data=[data])
        
        # Original query path for filtering and pagination
        logger.info(f"Query request: top_k={top_k}, scene_id={scene_id}, frame_id={frame_id}, source={source}, offset={offset}")
        
        # Filters are already built above, just use them
        where_sql = filter_sql
        params = filter_params
        
        # Fetch data with LIMIT and OFFSET
        data_query = f"""
            SELECT * FROM av_data 
            WHERE {where_sql}
            ORDER BY id ASC
            LIMIT ? OFFSET ?
        """
        cursor.execute(data_query, params + [top_k, offset])
        rows = cursor.fetchall()
        
        # Convert rows to dictionaries
        data = [row_to_dict(row) for row in rows]
        
        conn.close()
        logger.info(f"Query successful: returned {len(data)} records out of {total_count} total")
        
        return QueryResponse(
            total=total_count,
            count=len(data),
            data=data
        )
        
    except Exception as e:
        logger.error(f"Error querying data: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail=f"Failed to query data: {str(e)}"
        )


@app.put("/api/v1/data/{record_id}", tags=["Data"])
async def update_data(
    record_id: int,
    update_data: AVDataUpdate
) -> Dict[str, Any]:
    """
    Update a specific record in the database.
    
    Args:
        record_id: ID of the record to update
        update_data: Data to update
        
    Returns:
        Updated record data
    """
    try:
        logger.info(f"Update request for record_id={record_id}")
        
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Check if record exists
        cursor.execute("SELECT * FROM av_data WHERE id = ?", (record_id,))
        existing_record = cursor.fetchone()
        
        if not existing_record:
            conn.close()
            logger.warning(f"Record not found: {record_id}")
            raise HTTPException(
                status_code=404,
                detail=f"Record with id {record_id} not found"
            )
        
        # Build UPDATE query dynamically
        update_fields = []
        params = []
        
        update_dict = update_data.dict(exclude_unset=True)
        
        for field, value in update_dict.items():
            if value is not None:
                update_fields.append(f"{field} = ?")
                
                # Serialize dicts to JSON strings
                if isinstance(value, dict):
                    params.append(json.dumps(value))
                else:
                    params.append(value)
        
        if not update_fields:
            conn.close()
            logger.warning("No fields to update")
            raise HTTPException(
                status_code=400,
                detail="No fields to update"
            )
        
        # Add updated_at timestamp
        update_fields.append("updated_at = ?")
        params.append(datetime.now().isoformat())
        
        # Execute update
        update_query = f"""
            UPDATE av_data 
            SET {', '.join(update_fields)}
            WHERE id = ?
        """
        params.append(record_id)
        
        cursor.execute(update_query, params)
        conn.commit()
        
        # Fetch updated record
        cursor.execute("SELECT * FROM av_data WHERE id = ?", (record_id,))
        updated_record = cursor.fetchone()
        
        if not updated_record:
            conn.close()
            logger.error(f"Record {record_id} not found after update")
            raise HTTPException(
                status_code=500,
                detail=f"Record {record_id} was updated but could not be retrieved"
            )
        
        result = row_to_dict(updated_record)
        
        conn.close()
        logger.info(f"Record {record_id} updated successfully")
        
        return {
            "status": "success",
            "message": f"Record {record_id} updated successfully",
            "data": result
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating data: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail=f"Failed to update data: {str(e)}"
        )


@app.get("/api/v1/data/navigate")
async def navigate_frame(
    scene_id: str = Query(..., description="Current scene ID"),
    current_offset: int = Query(..., description="Current offset within scene"),
    direction: str = Query(..., description="Navigation direction: 'next' or 'previous'")
):
    """
    Navigate to the next or previous frame within a scene, or move to adjacent scene if at boundary.
    
    Args:
        scene_id: Current scene ID
        current_offset: Current offset within the scene
        direction: 'next' or 'previous'
    
    Returns:
        The target frame data with scene context
    """
    try:
        if direction not in ["next", "previous"]:
            raise HTTPException(status_code=400, detail="Direction must be 'next' or 'previous'")
        
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Get all frames in current scene, sorted by timestamp
        cursor.execute("""
            SELECT * FROM av_data
            WHERE scene_id = ?
            ORDER BY CAST(json_extract(calibration, '$.img_timestamp[0]') AS REAL) ASC
        """, (scene_id,))
        
        current_scene_frames = cursor.fetchall()
        total_in_scene = len(current_scene_frames)
        
        if direction == "next":
            target_offset = current_offset + 1
            
            if target_offset < total_in_scene:
                # Stay in current scene
                row = current_scene_frames[target_offset]
                data = row_to_dict(row)
                conn.close()
                
                return {
                    "scene_id": scene_id,
                    "offset": target_offset,
                    "total_in_scene": total_in_scene,
                    "scene_changed": False,
                    "data": data
                }
            else:
                # Move to next scene
                cursor.execute("""
                    SELECT DISTINCT scene_id
                    FROM av_data
                    ORDER BY scene_id ASC
                """)
                scenes = [row['scene_id'] for row in cursor.fetchall()]
                
                try:
                    current_scene_index = scenes.index(scene_id)
                except ValueError:
                    raise HTTPException(status_code=404, detail="Current scene not found")
                
                if current_scene_index >= len(scenes) - 1:
                    conn.close()
                    raise HTTPException(status_code=404, detail="Already at last frame of last scene")
                
                next_scene_id = scenes[current_scene_index + 1]
                
                # Get first frame of next scene
                cursor.execute("""
                    SELECT * FROM av_data
                    WHERE scene_id = ?
                    ORDER BY CAST(json_extract(calibration, '$.img_timestamp[0]') AS REAL) ASC
                    LIMIT 1
                """, (next_scene_id,))
                
                row = cursor.fetchone()
                
                # Get total frames in new scene
                cursor.execute("""
                    SELECT COUNT(*) as count FROM av_data
                    WHERE scene_id = ?
                """, (next_scene_id,))
                total_in_new_scene = cursor.fetchone()['count']
                
                conn.close()
                
                data = row_to_dict(row)
                
                return {
                    "scene_id": next_scene_id,
                    "offset": 0,
                    "total_in_scene": total_in_new_scene,
                    "scene_changed": True,
                    "data": data
                }
        
        else:  # previous
            target_offset = current_offset - 1
            
            if target_offset >= 0:
                # Stay in current scene
                row = current_scene_frames[target_offset]
                data = row_to_dict(row)
                conn.close()
                
                return {
                    "scene_id": scene_id,
                    "offset": target_offset,
                    "total_in_scene": total_in_scene,
                    "scene_changed": False,
                    "data": data
                }
            else:
                # Move to previous scene
                cursor.execute("""
                    SELECT DISTINCT scene_id
                    FROM av_data
                    ORDER BY scene_id ASC
                """)
                scenes = [row['scene_id'] for row in cursor.fetchall()]
                
                try:
                    current_scene_index = scenes.index(scene_id)
                except ValueError:
                    raise HTTPException(status_code=404, detail="Current scene not found")
                
                if current_scene_index <= 0:
                    conn.close()
                    raise HTTPException(status_code=404, detail="Already at first frame of first scene")
                
                prev_scene_id = scenes[current_scene_index - 1]
                
                # Get all frames in previous scene to get the last one
                cursor.execute("""
                    SELECT * FROM av_data
                    WHERE scene_id = ?
                    ORDER BY CAST(json_extract(calibration, '$.img_timestamp[0]') AS REAL) ASC
                """, (prev_scene_id,))
                
                prev_scene_frames = cursor.fetchall()
                total_in_prev_scene = len(prev_scene_frames)
                
                # Get last frame of previous scene
                row = prev_scene_frames[-1]
                
                conn.close()
                
                data = row_to_dict(row)
                
                return {
                    "scene_id": prev_scene_id,
                    "offset": total_in_prev_scene - 1,
                    "total_in_scene": total_in_prev_scene,
                    "scene_changed": True,
                    "data": data
                }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error navigating frame: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail=f"Failed to navigate frame: {str(e)}"
        )


@app.get("/api/v1/data/{record_id}", response_model=AVDataResponse, tags=["Data"])
async def get_data_by_id(record_id: int) -> AVDataResponse:
    """
    Get a specific record by ID.
    
    Args:
        record_id: ID of the record to retrieve
        
    Returns:
        AVDataResponse containing the record data
    """
    try:
        logger.info(f"Get record by id: {record_id}")
        
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM av_data WHERE id = ?", (record_id,))
        row = cursor.fetchone()
        
        if not row:
            conn.close()
            logger.warning(f"Record not found: {record_id}")
            raise HTTPException(
                status_code=404,
                detail=f"Record with id {record_id} not found"
            )
        
        result = row_to_dict(row)
        conn.close()
        
        logger.info(f"Record {record_id} retrieved successfully")
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting data: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get data: {str(e)}"
        )


@app.get("/api/v1/scenes")
async def get_scenes():
    """
    Get list of all unique scene IDs in order.
    
    Returns:
        List of scene IDs sorted by their first frame's timestamp
    """
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Get unique scene IDs ordered by the first frame's timestamp in each scene
        query = """
            SELECT DISTINCT scene_id
            FROM av_data
            ORDER BY scene_id ASC
        """
        
        cursor.execute(query)
        rows = cursor.fetchall()
        conn.close()
        
        scenes = [row['scene_id'] for row in rows]
        
        logger.info(f"Retrieved {len(scenes)} unique scenes")
        return {"scenes": scenes, "count": len(scenes)}
        
    except Exception as e:
        logger.error(f"Error getting scenes: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get scenes: {str(e)}"
        )


@app.get("/api/v1/scenes/jump")
async def jump_to_scene(
    current_scene_id: str = Query(..., description="Current scene ID"),
    direction: str = Query(..., description="Navigation direction: 'next' or 'previous'"),
    source: List[str] = Query(default=[], description="Filter by source(s)")
):
    """
    Get the target scene info for navigation.
    OPTIMIZED: Returns both offset and record ID for fast ID-based pagination.
    Uses ID-based ordering for chronological scene navigation.
    
    Args:
        current_scene_id: The current scene ID
        direction: 'next' or 'previous'
        source: Optional list of data sources to filter by
    
    Returns:
        Target scene ID, global offset, and record ID
    """
    try:
        import time
        start = time.time()
        if direction not in ["next", "previous"]:
            raise HTTPException(status_code=400, detail="Direction must be 'next' or 'previous'")
        
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Build filter clause for source
        filter_clauses = []
        filter_params = []
        
        if source and len(source) > 0:
            placeholders = ",".join("?" * len(source))
            filter_clauses.append(f"source IN ({placeholders})")
            filter_params.extend(source)
        
        filter_sql = " AND ".join(filter_clauses) if filter_clauses else "1=1"
        
        # Get the minimum ID (first frame) of the current scene
        query = f"""
            SELECT MIN(id) as min_id
            FROM av_data
            WHERE scene_id = ? AND {filter_sql}
        """
        cursor.execute(query, [current_scene_id] + filter_params)
        
        row = cursor.fetchone()
        if not row or row['min_id'] is None:
            conn.close()
            raise HTTPException(status_code=404, detail=f"Scene ID '{current_scene_id}' not found in filtered data")
        
        current_scene_min_id = row['min_id']
        
        # Find the target scene based on ID ordering with source filter
        if direction == "next":
            # Get the scene whose first frame ID is right after current scene's last frame
            query = f"""
                SELECT scene_id, MIN(id) as min_id
                FROM av_data
                WHERE scene_id != ? AND {filter_sql}
                GROUP BY scene_id
                HAVING min_id > ?
                ORDER BY min_id ASC
                LIMIT 1
            """
            cursor.execute(query, [current_scene_id] + filter_params + [current_scene_min_id])
        else:  # previous
            # Get the scene whose first frame ID is right before current scene's first frame
            query = f"""
                SELECT scene_id, MIN(id) as min_id
                FROM av_data
                WHERE scene_id != ? AND {filter_sql}
                GROUP BY scene_id
                HAVING min_id < ?
                ORDER BY min_id DESC
                LIMIT 1
            """
            cursor.execute(query, [current_scene_id] + filter_params + [current_scene_min_id])
        
        row = cursor.fetchone()
        if not row:
            conn.close()
            raise HTTPException(status_code=404, detail=f"No {direction} scene available in filtered data")
        
        target_scene_id = row['scene_id']
        target_scene_min_id = row['min_id']
        
        # Get the FILTERED offset by counting records with ID less than target scene's first ID
        count_query = f"""
            SELECT COUNT(*) as offset
            FROM av_data
            WHERE id < ? AND {filter_sql}
        """
        cursor.execute(count_query, [target_scene_min_id] + filter_params)
        
        result = cursor.fetchone()
        global_offset = result['offset']
        
        conn.close()
        
        logger.info(f"Scene jump (filtered by {source or 'all'}): '{current_scene_id}' (ID {current_scene_min_id}) -> '{target_scene_id}' (ID {target_scene_min_id}, offset: {global_offset})")
        end = time.time()
        logger.info(f"Time taken: {end - start} seconds")
        return {
            "target_scene_id": target_scene_id,
            "global_offset": global_offset,
            "target_id": target_scene_min_id,  # Return the actual ID for ID-based pagination
            "direction": direction,
            "message": "Scene navigation successful"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error jumping scene: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail=f"Failed to jump scene: {str(e)}"
        )


@app.get("/api/v1/scenes/navigate")
async def navigate_scene(
    current_scene_id: str = Query(..., description="Current scene ID"),
    direction: str = Query(..., description="Navigation direction: 'next' or 'previous'")
):
    """
    Navigate to the next or previous scene.
    
    Args:
        current_scene_id: The current scene ID
        direction: 'next' or 'previous'
    
    Returns:
        The target scene ID and the first frame's data
    """
    try:
        if direction not in ["next", "previous"]:
            raise HTTPException(status_code=400, detail="Direction must be 'next' or 'previous'")
        
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Get all unique scene IDs in order
        cursor.execute("""
            SELECT DISTINCT scene_id
            FROM av_data
            ORDER BY scene_id ASC
        """)
        scenes = [row['scene_id'] for row in cursor.fetchall()]
        
        # Find current scene index
        try:
            current_index = scenes.index(current_scene_id)
        except ValueError:
            raise HTTPException(status_code=404, detail=f"Scene ID '{current_scene_id}' not found")
        
        # Calculate target index
        if direction == "next":
            target_index = current_index + 1
            if target_index >= len(scenes):
                raise HTTPException(status_code=404, detail="No next scene available")
        else:  # previous
            target_index = current_index - 1
            if target_index < 0:
                raise HTTPException(status_code=404, detail="No previous scene available")
        
        target_scene_id = scenes[target_index]
        
        # Get the first frame of the target scene
        cursor.execute("""
            SELECT * FROM av_data
            WHERE scene_id = ?
            ORDER BY CAST(json_extract(calibration, '$.img_timestamp[0]') AS REAL) ASC
            LIMIT 1
        """, (target_scene_id,))
        
        row = cursor.fetchone()
        conn.close()
        
        if not row:
            raise HTTPException(status_code=404, detail=f"No frames found for scene '{target_scene_id}'")
        
        # Convert row to dictionary
        data = row_to_dict(row)
        
        logger.info(f"Navigated from scene '{current_scene_id}' to '{target_scene_id}' ({direction})")
        
        return {
            "scene_id": target_scene_id,
            "direction": direction,
            "data": data
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error navigating scene: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail=f"Failed to navigate scene: {str(e)}"
        )


@app.get("/api/v1/data/{record_id}/scene-offset")
async def get_scene_offset(record_id: int):
    """
    Get the scene-relative offset for a specific record.
    
    Args:
        record_id: The record ID
    
    Returns:
        Scene ID, offset within scene, and total frames in scene
    """
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Get the record's scene_id
        cursor.execute("SELECT scene_id FROM av_data WHERE id = ?", (record_id,))
        row = cursor.fetchone()
        
        if not row:
            raise HTTPException(status_code=404, detail="Record not found")
        
        scene_id = row['scene_id']
        
        # Get all frames in the scene, sorted by timestamp
        cursor.execute("""
            SELECT id FROM av_data
            WHERE scene_id = ?
            ORDER BY CAST(json_extract(calibration, '$.img_timestamp[0]') AS REAL) ASC
        """, (scene_id,))
        
        scene_frame_ids = [r['id'] for r in cursor.fetchall()]
        total_in_scene = len(scene_frame_ids)
        
        # Find the offset of this record within the scene
        try:
            offset = scene_frame_ids.index(record_id)
        except ValueError:
            offset = 0
        
        conn.close()
        
        return {
            "scene_id": scene_id,
            "offset": offset,
            "total_in_scene": total_in_scene
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting scene offset: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get scene offset: {str(e)}"
        )


@app.get("/api/v1/scenes/next-uncompleted-metadata", tags=["Scenes"])
async def get_next_uncompleted_metadata_scene(
    current_scene_id: Optional[str] = Query(None, description="Current scene ID to start from"),
    source: List[str] = Query(default=[], description="Filter by source(s)")
):
    """
    Find the next scene that has uncompleted metadata (verified_metadata = 0).
    Returns the first frame of that scene.
    
    Args:
        current_scene_id: Optional current scene ID. If provided, search starts after this scene.
        source: Optional list of data sources to filter by
    
    Returns:
        The first frame of the next uncompleted scene, along with target_id and global_offset
    """
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Build filter clause for source
        filter_clauses = []
        filter_params = []
        
        if source and len(source) > 0:
            placeholders = ",".join("?" * len(source))
            filter_clauses.append(f"source IN ({placeholders})")
            filter_params.extend(source)
        
        filter_sql = " AND ".join(filter_clauses) if filter_clauses else "1=1"
        
        # Get all unique scenes ordered by their minimum ID (chronological)
        query = f"""
            SELECT scene_id, MIN(id) as min_id
            FROM av_data
            WHERE {filter_sql}
            GROUP BY scene_id
            ORDER BY min_id ASC
        """
        cursor.execute(query, filter_params)
        all_scenes = cursor.fetchall()
        
        if not all_scenes:
            raise HTTPException(status_code=404, detail="No scenes found")
        
        # Find starting point
        start_index = 0
        if current_scene_id:
            for i, scene in enumerate(all_scenes):
                if scene['scene_id'] == current_scene_id:
                    start_index = i + 1
                    break
        
        # Search for next uncompleted scene
        target_scene = None
        for i in range(start_index, len(all_scenes)):
            scene_id = all_scenes[i]['scene_id']
            
            # Check if this scene has any unverified metadata frames
            check_query = f"""
                SELECT COUNT(*) as unverified_count
                FROM av_data
                WHERE scene_id = ? AND verified_metadata = 0 AND {filter_sql}
            """
            cursor.execute(check_query, [scene_id] + filter_params)
            result = cursor.fetchone()
            
            if result['unverified_count'] > 0:
                target_scene = all_scenes[i]
                break
        
        # If no uncompleted scene found after current, wrap around to beginning
        if not target_scene:
            for i in range(0, start_index):
                scene_id = all_scenes[i]['scene_id']
                
                # Check if this scene has any unverified metadata frames
                check_query = f"""
                    SELECT COUNT(*) as unverified_count
                    FROM av_data
                    WHERE scene_id = ? AND verified_metadata = 0 AND {filter_sql}
                """
                cursor.execute(check_query, [scene_id] + filter_params)
                result = cursor.fetchone()
                
                if result['unverified_count'] > 0:
                    target_scene = all_scenes[i]
                    break
        
        if not target_scene:
            raise HTTPException(
                status_code=404,
                detail="No uncompleted metadata scenes found. All scenes are verified!"
            )
        
        target_scene_id = target_scene['scene_id']
        
        # Get the first UNVERIFIED frame of the target scene (not just first frame)
        query = f"""
            SELECT * FROM av_data
            WHERE scene_id = ? AND verified_metadata = 0 AND {filter_sql}
            ORDER BY id ASC
            LIMIT 1
        """
        cursor.execute(query, [target_scene_id] + filter_params)
        row = cursor.fetchone()
        
        if not row:
            logger.error(f"No unverified frame found for scene {target_scene_id}, but count showed unverified frames exist")
            raise HTTPException(status_code=404, detail="Target scene unverified frame not found")
        
        target_min_id = row['id']
        
        logger.info(f"Found uncompleted frame: scene={target_scene_id}, record_id={target_min_id}, verified_metadata={row['verified_metadata']}")
        
        # Calculate global offset (total records before this one with filters)
        count_query = f"""
            SELECT COUNT(*) as offset
            FROM av_data
            WHERE id < ? AND {filter_sql}
        """
        cursor.execute(count_query, [target_min_id] + filter_params)
        offset_result = cursor.fetchone()
        global_offset = offset_result['offset'] if offset_result else 0
        
        conn.close()
        
        record = row_to_dict(row)
        
        return {
            "target_id": target_min_id,
            "global_offset": global_offset,
            "scene_id": target_scene_id,
            "record": record,
            "message": f"Found uncompleted scene: {target_scene_id}"
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error finding next uncompleted metadata scene: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail=f"Failed to find next uncompleted metadata scene: {str(e)}"
        )


@app.get("/api/v1/scenes/frame-ids", tags=["Scenes"])
async def get_scene_frame_ids(
    scene_id: str = Query(..., description="Scene ID to get frame IDs for"),
    source: List[str] = Query(default=[], description="Filter by source(s)")
):
    """
    Get only the frame IDs for a specific scene (lightweight, fast query).
    Returns only IDs without loading full record data.
    
    Args:
        scene_id: The scene ID to get frame IDs for (as query parameter to handle paths with slashes)
        source: Optional list of data sources to filter by
    
    Returns:
        List of frame IDs and count
    """
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Build filter clause for source
        filter_clauses = ["scene_id = ?"]
        filter_params = [scene_id]
        
        if source and len(source) > 0:
            placeholders = ",".join("?" * len(source))
            filter_clauses.append(f"source IN ({placeholders})")
            filter_params.extend(source)
        
        filter_sql = " AND ".join(filter_clauses)
        
        # Only select IDs - much faster than selecting all columns
        query = f"""
            SELECT id FROM av_data
            WHERE {filter_sql}
            ORDER BY id ASC
        """
        
        logger.info(f"Fetching frame IDs for scene: {scene_id}")
        cursor.execute(query, filter_params)
        rows = cursor.fetchall()
        
        frame_ids = [row['id'] for row in rows]
        
        conn.close()
        
        logger.info(f"Found {len(frame_ids)} frames for scene {scene_id}")
        
        return {
            "scene_id": scene_id,
            "count": len(frame_ids),
            "frame_ids": frame_ids
        }
    
    except Exception as e:
        logger.error(f"Error fetching frame IDs for scene {scene_id}: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch frame IDs: {str(e)}"
        )


@app.get("/api/v1/lidar/convert-feather", tags=["LiDAR"])
async def convert_feather_to_pcd(
    path: str = Query(..., description="Absolute path to the .feather lidar file")
):
    """
    Convert AV2 feather lidar file to PCD format (ASCII) for frontend visualization.
    Supports both local filesystem and HPC remote files.
    
    Args:
        path: Absolute path to the .feather file
    
    Returns:
        PCD format string (ASCII) with x, y, z, intensity fields
    """
    try:
        if not FEATHER_SUPPORT:
            raise HTTPException(
                status_code=501,
                detail="Feather file support not available. Please install pandas and pyarrow."
            )
        
        # Check if path looks like an HPC path (starts with /home/ typically)
        # and if HPC credentials are configured
        hpc_username = os.getenv("HPC_USERNAME", "")
        hpc_password = os.getenv("HPC_PASSWORD", "")
        hpc_base_url = os.getenv("HPC_BASE_URL", "")
        
        use_hpc = False
        if hpc_username and hpc_password and REQUESTS_AVAILABLE:
            # If path starts with /home/, treat as HPC path
            if path.startswith("/home/"):
                use_hpc = True
                logger.info(f"Detected HPC path, will fetch from HPC: {path}")
        
        # Fetch or read the feather file
        if use_hpc:
            # Fetch from HPC
            try:
                base_url_with_slash = hpc_base_url if hpc_base_url.endswith('/') else f"{hpc_base_url}/"
                full_url = f"{base_url_with_slash}{path}"
                
                logger.info(f"Fetching feather file from HPC: {full_url}")
                
                # Create Basic Auth
                auth = (hpc_username, hpc_password)
                
                # Fetch the file
                response = requests.get(full_url, auth=auth, timeout=30)
                
                if response.status_code != 200:
                    raise HTTPException(
                        status_code=response.status_code,
                        detail=f"Failed to fetch from HPC: {response.status_code} {response.reason}"
                    )
                
                # Read feather from bytes
                logger.info(f"Downloaded {len(response.content)} bytes from HPC")
                file_handle = io.BytesIO(response.content)
                df = feather.read_feather(file_handle, memory_map=False)
                
            except requests.RequestException as e:
                logger.error(f"HPC fetch error: {str(e)}")
                raise HTTPException(
                    status_code=502,
                    detail=f"Failed to fetch from HPC: {str(e)}"
                )
        else:
            # Read from local filesystem
            feather_path = Path(path)
            
            if not feather_path.exists():
                raise HTTPException(
                    status_code=404,
                    detail=f"Feather file not found: {path}"
                )
            
            if not feather_path.suffix == ".feather":
                raise HTTPException(
                    status_code=400,
                    detail="File must have .feather extension"
                )
            
            # Read feather file
            logger.info(f"Reading feather file from local filesystem: {feather_path}")
            with feather_path.open("rb") as file_handle:
                df = feather.read_feather(file_handle, memory_map=True)
        
        # Validate required columns
        required_columns = ["x", "y", "z"]
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise HTTPException(
                status_code=400,
                detail=f"Missing required columns: {missing_columns}"
            )
        
        # Extract point cloud data
        xyz = df.loc[:, ["x", "y", "z"]].to_numpy().astype(float)
        
        # AV2/Argoverse2 coordinate system transformation
        # Rotate by 90 degrees to align with nuScenes coordinate system
        # Original AV2: X forward, Y left, Z up
        # Target (nuScenes): X forward, Y left, Z up
        # Apply 90-degree rotation around Z-axis: (x, y, z) -> (-y, x, z)
        logger.info(f"Applying 90-degree rotation to {len(xyz)} points")
        xyz_rotated = np.zeros_like(xyz)
        xyz_rotated[:, 0] = -xyz[:, 1]  # new_x = -old_y
        xyz_rotated[:, 1] = xyz[:, 0]   # new_y = old_x
        xyz_rotated[:, 2] = xyz[:, 2]   # new_z = old_z (no change in height)
        
        xyz = xyz_rotated  # Use rotated coordinates
        
        # Get intensity if available, otherwise use default value
        if "intensity" in df.columns:
            intensity = df.loc[:, "intensity"].to_numpy().astype(float)
        else:
            intensity = np.full(len(xyz), 100.0, dtype=float)
        
        num_points = len(xyz)
        
        # Generate PCD header (ASCII format)
        pcd_header = [
            "# .PCD v0.7 - Point Cloud Data file format",
            "VERSION 0.7",
            "FIELDS x y z intensity",
            "SIZE 4 4 4 4",
            "TYPE F F F F",
            "COUNT 1 1 1 1",
            f"WIDTH {num_points}",
            "HEIGHT 1",
            "VIEWPOINT 0 0 0 1 0 0 0",
            f"POINTS {num_points}",
            "DATA ascii"
        ]
        
        # Build PCD content
        pcd_lines = pcd_header.copy()
        
        # Add point data
        for i in range(num_points):
            x, y, z = xyz[i]
            intens = intensity[i]
            pcd_lines.append(f"{x} {y} {z} {intens}")
        
        pcd_content = "\n".join(pcd_lines)
        
        logger.info(f"Successfully converted feather to PCD: {num_points} points from {'HPC' if use_hpc else 'local filesystem'}")
        
        return JSONResponse(
            content={
                "pcd_content": pcd_content,
                "num_points": num_points,
                "source": "argoverse2",
                "fetched_from": "hpc" if use_hpc else "local"
            }
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error converting feather to PCD: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail=f"Failed to convert feather file: {str(e)}"
        )


class SpatialRelationUpdate(BaseModel):
    """Schema for updating a specific spatial relation"""
    source_track_id: str
    target_track_id: str
    relations: List[str]  # List of relation strings from the Relation enum


class UpdateSpatialRelationsBody(BaseModel):
    """Request body for PUT /api/v1/data/{record_id}/spatial-relations"""
    updates: List[SpatialRelationUpdate] = Field(..., description="List of relation updates")
    verify_by: Optional[str] = Field(None, description="Verifier identifier; when set, marks frame as verified")


class VerifySpatialRelationsBody(BaseModel):
    """Request body for PUT /api/v1/data/{record_id}/verify-spatial"""
    spatial_relations: List[Dict[str, Any]] = Field(..., description="Complete spatial relations array to save")
    verify_by: Optional[str] = Field(None, description="Verifier identifier")


@app.put("/api/v1/data/{record_id}/spatial-relations")
async def update_spatial_relations(
    record_id: int,
    body: UpdateSpatialRelationsBody
):
    """
    Update spatial relations for a specific record.
    Saves to updated_spatial_relations and marks as verified.
    
    Args:
        record_id: The record ID
        body: Request body containing updates list and optional verify_by
    
    Returns:
        Success message and updated count
    """
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Get current spatial relations
        cursor.execute("""
            SELECT spatial_relations, updated_spatial_relations 
            FROM av_data 
            WHERE id = ?
        """, (record_id,))
        
        row = cursor.fetchone()
        if not row:
            conn.close()
            raise HTTPException(status_code=404, detail="Record not found")
        
        # Start with current relations (prefer updated if exists, otherwise copy from original)
        if row['updated_spatial_relations']:
            current_relations = json.loads(row['updated_spatial_relations'])
            logger.info(f"Using existing updated_spatial_relations for record {record_id}")
        elif row['spatial_relations']:
            current_relations = json.loads(row['spatial_relations'])
            logger.info(f"Copying from spatial_relations to updated_spatial_relations for record {record_id}")
        else:
            current_relations = []
            logger.warning(f"No spatial relations found for record {record_id}")
        
        logger.info(f"Current relations count: {len(current_relations)}")
        
        # Update the relations based on the updates list
        update_map = {
            (u.source_track_id, u.target_track_id): u.relations 
            for u in body.updates
        }
        
        logger.info(f"Update map: {update_map}")
        
        updated_count = 0
        for relation in current_relations:
            key = (
                relation['source_object']['track_id'],
                relation['target_object']['track_id']
            )
            if key in update_map:
                old_relations = relation['relations']
                relation['relations'] = update_map[key]
                updated_count += 1
                logger.info(f"Updated relation {key}: {old_relations} -> {update_map[key]}")
        
        # Save to updated_spatial_relations with optional verification
        updated_json = json.dumps(current_relations)
        logger.info(f"Saving updated_spatial_relations with {len(current_relations)} relations")
        
        # Build UPDATE query based on whether verify_by is provided
        if body.verify_by:
            cursor.execute("""
                UPDATE av_data 
                SET updated_spatial_relations = ?,
                    verified_spatial_relations = 1,
                    verify_by = ?,
                    updated_at = ?
                WHERE id = ?
            """, (
                updated_json,
                body.verify_by,
                datetime.now().isoformat(),
                record_id
            ))
        else:
            cursor.execute("""
                UPDATE av_data 
                SET updated_spatial_relations = ?,
                    updated_at = ?
                WHERE id = ?
            """, (
                updated_json,
                datetime.now().isoformat(),
                record_id
            ))
        
        conn.commit()
        
        # Verify the save
        cursor.execute("SELECT updated_spatial_relations, verified_spatial_relations, verify_by FROM av_data WHERE id = ?", (record_id,))
        verify_row = cursor.fetchone()
        if verify_row and verify_row['updated_spatial_relations']:
            logger.info(f"Verified: updated_spatial_relations saved successfully for record {record_id}")
        else:
            logger.error(f"Failed to save updated_spatial_relations for record {record_id}")
        
        conn.close()
        
        logger.info(f"Updated {updated_count} spatial relations for record {record_id}")
        
        return {
            "success": True,
            "record_id": record_id,
            "updated_count": updated_count,
            "message": f"Successfully updated {updated_count} spatial relations",
            "verified": verify_row['verified_spatial_relations'] == 1 if verify_row else False,
            "verify_by": verify_row['verify_by'] if verify_row else None
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating spatial relations for record {record_id}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to update spatial relations: {str(e)}")

@app.put("/api/v1/data/{record_id}/internal-interactions")
async def update_internal_interactions(
    record_id: int,
    body: UpdateInternalInteractionsBody,
):
    """
    Overwrite internal interactions for a specific record.
    Saves the entire internal_interactions array to updated_internal_interactions
    and marks the frame as verified.
    
    Args:
        record_id: The record ID
        body: { internal_interactions: [...], verify_by?: str }
    
    Returns:
        Success message
    """
    try:
        internal_interactions = body.internal_interactions
        verify_by = body.verify_by or ""

        conn = get_db_connection()
        cursor = conn.cursor()

        # Check if record exists
        cursor.execute("SELECT id FROM av_data WHERE id = ?", (record_id,))
        row = cursor.fetchone()
        if not row:
            conn.close()
            raise HTTPException(status_code=404, detail="Record not found")

        # Directly save the entire internal_interactions array to updated_internal_interactions
        interactions_json = json.dumps(internal_interactions)
        logger.info("Overwriting updated_internal_interactions with %s interactions for record %s", len(internal_interactions), record_id)

        cursor.execute("""
            UPDATE av_data
            SET updated_internal_interactions = ?,
                verified_internal_interactions = 1,
                verify_by = ?,
                updated_at = ?
            WHERE id = ?
        """, (
            interactions_json,
            verify_by,
            datetime.now().isoformat(),
            record_id,
        ))

        conn.commit()

        # Verify the save
        cursor.execute(
            "SELECT updated_internal_interactions, verified_internal_interactions, verify_by FROM av_data WHERE id = ?",
            (record_id,),
        )
        verify_row = cursor.fetchone()
        if verify_row and verify_row["updated_internal_interactions"]:
            logger.info(
                "Saved internal interactions for record %s: count=%s, verified=%s, verify_by=%s",
                record_id,
                len(internal_interactions),
                verify_row["verified_internal_interactions"],
                verify_row["verify_by"],
            )
        else:
            logger.error("Failed to save updated_internal_interactions for record %s", record_id)

        conn.close()

        logger.info("Successfully saved %s internal interactions for record %s", len(internal_interactions), record_id)

        return {
            "success": True,
            "record_id": record_id,
            "interactions_count": len(internal_interactions),
            "verified_internal_interactions": 1,
            "verify_by": verify_by,
            "message": f"Successfully saved {len(internal_interactions)} internal interactions and marked as verified",
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error updating internal interactions: %s", str(e))
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail=f"Failed to update internal interactions: {str(e)}",
        )


@app.put("/api/v1/data/{record_id}/external-interactions")
async def update_external_interactions(
    record_id: int,
    body: UpdateExternalInteractionsBody,
):
    """
    Overwrite external interactions for a specific record.
    Saves the entire external_interactions array to updated_external_interactions
    and marks the frame as verified.
    
    Args:
        record_id: The record ID
        body: { external_interactions: [...], verify_by?: str }
    
    Returns:
        Success message
    """
    try:
        external_interactions = body.external_interactions
        verify_by = body.verify_by or ""

        conn = get_db_connection()
        cursor = conn.cursor()

        # Check if record exists
        cursor.execute("SELECT id FROM av_data WHERE id = ?", (record_id,))
        row = cursor.fetchone()
        if not row:
            conn.close()
            raise HTTPException(status_code=404, detail="Record not found")

        # Directly save the entire external_interactions array to updated_external_interactions
        interactions_json = json.dumps(external_interactions)
        logger.info("Overwriting updated_external_interactions with %s interactions for record %s", len(external_interactions), record_id)

        cursor.execute("""
            UPDATE av_data
            SET updated_external_interactions = ?,
                verified_external_interactions = 1,
                verify_by = ?,
                updated_at = ?
            WHERE id = ?
        """, (
            interactions_json,
            verify_by,
            datetime.now().isoformat(),
            record_id,
        ))

        conn.commit()

        # Verify the save
        cursor.execute(
            "SELECT updated_external_interactions, verified_external_interactions, verify_by FROM av_data WHERE id = ?",
            (record_id,),
        )
        verify_row = cursor.fetchone()
        if verify_row and verify_row["updated_external_interactions"]:
            logger.info(
                "Saved external interactions for record %s: count=%s, verified=%s, verify_by=%s",
                record_id,
                len(external_interactions),
                verify_row["verified_external_interactions"],
                verify_row["verify_by"],
            )
        else:
            logger.error("Failed to save updated_external_interactions for record %s", record_id)

        conn.close()

        logger.info("Successfully saved %s external interactions for record %s", len(external_interactions), record_id)

        return {
            "success": True,
            "record_id": record_id,
            "interactions_count": len(external_interactions),
            "verified_external_interactions": 1,
            "verify_by": verify_by,
            "message": f"Successfully saved {len(external_interactions)} external interactions and marked as verified",
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error updating external interactions: %s", str(e))
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail=f"Failed to update external interactions: {str(e)}",
        )


@app.put("/api/v1/data/{record_id}/log-time")
async def log_time_spent(
    record_id: int,
    body: LogTimeSpentBody,
):
    """
    Log time spent on a specific annotation mode for a record.
    Accumulates time spent in the appropriate column.
    
    Args:
        record_id: The record ID
        body: { mode: str, time_spent: float }
    
    Returns:
        Success message
    """
    try:
        mode = body.mode
        time_spent = body.time_spent
        
        # Map mode to column name
        column_map = {
            "metadata": "time_spent_metadata",
            "spatial": "time_spent_spatial",
            "internal": "time_spent_internal",
            "external": "time_spent_external",
            "all": "time_spent_all",
        }
        
        if mode not in column_map:
            raise HTTPException(status_code=400, detail=f"Invalid mode: {mode}")
        
        column_name = column_map[mode]
        
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Check if record exists
        cursor.execute("SELECT id FROM av_data WHERE id = ?", (record_id,))
        row = cursor.fetchone()
        if not row:
            conn.close()
            raise HTTPException(status_code=404, detail="Record not found")
        
        # Accumulate time spent (add to existing time)
        cursor.execute(f"""
            UPDATE av_data
            SET {column_name} = {column_name} + ?,
                updated_at = ?
            WHERE id = ?
        """, (
            time_spent,
            datetime.now().isoformat(),
            record_id,
        ))
        
        conn.commit()
        
        # Get updated time
        cursor.execute(f"SELECT {column_name} FROM av_data WHERE id = ?", (record_id,))
        verify_row = cursor.fetchone()
        total_time = verify_row[column_name] if verify_row else 0
        
        conn.close()
        
        logger.info(f"Logged {time_spent:.2f}s for mode '{mode}' on record {record_id}. Total: {total_time:.2f}s")
        
        return {
            "success": True,
            "record_id": record_id,
            "mode": mode,
            "time_logged": time_spent,
            "total_time": total_time,
            "message": f"Logged {time_spent:.2f}s for {mode} mode",
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error logging time spent: %s", str(e))
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail=f"Failed to log time spent: {str(e)}",
        )


@app.put("/api/v1/data/{record_id}/verify-spatial")
async def verify_spatial_relations(
    record_id: int,
    body: VerifySpatialRelationsBody
):
    """
    Save the entire spatial relations array to updated_spatial_relations and mark as verified.
    This overwrites updated_spatial_relations with the current state from frontend.
    
    Args:
        record_id: The record ID
        body: { spatial_relations: [...], verify_by?: str }
    
    Returns:
        Success message
    """
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Build update fields
        update_fields = [
            "updated_spatial_relations = ?",
            "verified_spatial_relations = 1",
            "updated_at = ?"
        ]
        params = [
            json.dumps(body.spatial_relations),
            datetime.now().isoformat()
        ]
        
        # Add verify_by if provided
        if body.verify_by:
            update_fields.append("verify_by = ?")
            params.append(body.verify_by)
        
        params.append(record_id)
        
        # Save the entire spatial relations array to updated_spatial_relations
        update_query = f"""
            UPDATE av_data 
            SET {', '.join(update_fields)}
            WHERE id = ?
        """
        cursor.execute(update_query, params)
        
        conn.commit()
        
        # Verify the update
        cursor.execute("""
            SELECT 
                length(spatial_relations) as orig_len,
                length(updated_spatial_relations) as updated_len,
                verified_spatial_relations
            FROM av_data 
            WHERE id = ?
        """, (record_id,))
        
        verify_row = cursor.fetchone()
        if verify_row:
            logger.info(f"Record {record_id} saved: orig_len={verify_row['orig_len']}, updated_len={verify_row['updated_len']}, verified={verify_row['verified_spatial_relations']}")
        
        conn.close()
        
        logger.info(f"Successfully saved and verified {len(body.spatial_relations)} spatial relations for record {record_id}")
        
        return {
            "success": True,
            "record_id": record_id,
            "relations_count": len(body.spatial_relations),
            "message": f"Successfully saved {len(body.spatial_relations)} spatial relations"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error saving spatial relations: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail=f"Failed to save spatial relations: {str(e)}"
        )


# ============================================================================
# Question Database Endpoints
# ============================================================================

class QuestionResponse(BaseModel):
    """Response schema for question data"""
    id: int
    question_id: str
    question_category: str
    question_type: str
    question_text: str
    options: Optional[List[str]] = None
    correct_answer: float  # Float to support absolute answer types (distance_absolute, scene_counting_absolute)
    source: Optional[str] = None
    scene_id: Optional[str] = None
    media_type: str
    images: Optional[List[str]] = None
    videos: Optional[List[str]] = None
    frame_paths: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None
    status: str = Field(default="pending", description="Review status: pending, accepted, rejected, ambiguous")
    time_spent_review: float = Field(default=0.0, description="Total seconds spent reviewing")
    created_at: str


class QuestionStatusUpdate(BaseModel):
    """Request schema for updating question status"""
    status: str = Field(..., description="New status: accepted, rejected, or ambiguous")
    reviewed_by: Optional[str] = Field(default=None, description="Username of reviewer")
    time_spent_review: Optional[float] = Field(default=None, description="Seconds spent reviewing this question")


class QuestionQueryResponse(BaseModel):
    """Response schema for question query endpoint"""
    total: int = Field(..., description="Total number of questions matching filter")
    count: int = Field(..., description="Number of questions returned")
    data: List[QuestionResponse]


def get_question_db_connection() -> sqlite3.Connection:
    """
    Create and return a connection to the question database.
    
    Returns:
        sqlite3.Connection: Database connection
    """
    try:
        conn = sqlite3.connect(QUESTION_DATABASE_PATH)
        conn.row_factory = sqlite3.Row
        logger.info("Question database connection established")
        return conn
    except Exception as e:
        logger.error(f"Failed to connect to question database: {str(e)}")
        logger.error(traceback.format_exc())
        raise


def question_row_to_dict(row: sqlite3.Row) -> Dict[str, Any]:
    """
    Convert question SQLite row to dictionary with JSON parsing.
    
    Args:
        row: SQLite row object
        
    Returns:
        Dict containing row data with parsed JSON fields
    """
    data = dict(row)
    
    # Parse JSON fields
    json_fields = ['options', 'images', 'videos', 'frame_paths', 'metadata']
    
    for field in json_fields:
        if field in data and data[field]:
            try:
                data[field] = json.loads(data[field])
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse JSON field: {field}")
                data[field] = None
        else:
            data[field] = None
    
    # Ensure status always has a value
    if "status" not in data or data["status"] is None:
        data["status"] = "pending"
    
    return data


@app.get("/api/v1/questions", tags=["Questions"])
async def get_questions(
    top_k: int = Query(default=10, description="Number of questions to return"),
    offset: int = Query(default=0, description="Offset for pagination"),
    question_category: Optional[str] = Query(default=None, description="Filter by category"),
    question_type: Optional[str] = Query(default=None, description="Filter by type"),
    source: Optional[str] = Query(default=None, description="Filter by source"),
    media_type: Optional[str] = Query(default=None, description="Filter by media type"),
    status: Optional[str] = Query(default=None, description="Filter by status: pending, accepted, rejected, ambiguous")
) -> QuestionQueryResponse:
    """
    Query questions with filtering and pagination.
    
    Args:
        top_k: Number of questions to return
        offset: Offset for pagination
        question_category: Filter by category ('construct', 'relation', 'causality')
        question_type: Filter by specific type
        status: Filter by review status (pending, accepted, rejected, ambiguous)
        source: Filter by data source
        media_type: Filter by media type ('image' or 'video')
        
    Returns:
        QuestionQueryResponse containing matching questions
    """
    try:
        conn = get_question_db_connection()
        cursor = conn.cursor()
        
        # Build filter conditions
        filter_clauses = []
        filter_params = []
        
        if question_category:
            filter_clauses.append("question_category = ?")
            filter_params.append(question_category)
        
        if question_type:
            filter_clauses.append("question_type = ?")
            filter_params.append(question_type)
        
        if source:
            filter_clauses.append("source = ?")
            filter_params.append(source)
        
        if media_type:
            filter_clauses.append("media_type = ?")
            filter_params.append(media_type)

        if status:
            filter_clauses.append("status = ?")
            filter_params.append(status)

        where_sql = " AND ".join(filter_clauses) if filter_clauses else "1=1"
        
        # Get total count
        count_query = f"SELECT COUNT(*) as count FROM questions WHERE {where_sql}"
        cursor.execute(count_query, filter_params)
        total_count = cursor.fetchone()['count']
        
        # Fetch data with LIMIT and OFFSET
        data_query = f"""
            SELECT * FROM questions 
            WHERE {where_sql}
            ORDER BY id ASC
            LIMIT ? OFFSET ?
        """
        cursor.execute(data_query, filter_params + [top_k, offset])
        rows = cursor.fetchall()
        
        # Convert rows to dictionaries
        data = [question_row_to_dict(row) for row in rows]
        
        conn.close()
        logger.info(f"Question query successful: returned {len(data)} questions out of {total_count} total")
        
        return QuestionQueryResponse(
            total=total_count,
            count=len(data),
            data=data
        )
        
    except Exception as e:
        logger.error(f"Error querying questions: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail=f"Failed to query questions: {str(e)}"
        )


@app.get("/api/v1/questions/{question_id}", tags=["Questions"])
async def get_question_by_id(question_id: str) -> QuestionResponse:
    """
    Get a specific question by its question_id.
    
    Args:
        question_id: UUID of the question
        
    Returns:
        QuestionResponse containing the question data
    """
    try:
        conn = get_question_db_connection()
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM questions WHERE question_id = ?", (question_id,))
        row = cursor.fetchone()
        
        if not row:
            conn.close()
            raise HTTPException(status_code=404, detail=f"Question {question_id} not found")
        
        result = question_row_to_dict(row)
        conn.close()
        
        logger.info(f"Question {question_id} retrieved successfully")
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting question: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get question: {str(e)}"
        )


@app.patch("/api/v1/questions/{question_id}/status", tags=["Questions"])
async def update_question_status(
    question_id: str,
    body: QuestionStatusUpdate
) -> QuestionResponse:
    """
    Update the review status of a question.

    Args:
        question_id: UUID of the question
        body: QuestionStatusUpdate with 'status' (accepted | rejected | ambiguous) and optional 'reviewed_by'

    Returns:
        Updated QuestionResponse
    """
    valid_statuses = {"accepted", "rejected", "ambiguous", "pending"}
    if body.status not in valid_statuses:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid status '{body.status}'. Must be one of: {valid_statuses}"
        )

    try:
        conn = get_question_db_connection()
        cursor = conn.cursor()

        # Verify question exists
        cursor.execute("SELECT id FROM questions WHERE question_id = ?", (question_id,))
        if not cursor.fetchone():
            conn.close()
            raise HTTPException(status_code=404, detail=f"Question {question_id} not found")

        # Apply update — accumulate time_spent_review rather than overwrite
        if body.time_spent_review is not None and body.time_spent_review > 0:
            cursor.execute(
                "UPDATE questions SET status = ?, time_spent_review = time_spent_review + ? WHERE question_id = ?",
                (body.status, body.time_spent_review, question_id)
            )
        else:
            cursor.execute(
                "UPDATE questions SET status = ? WHERE question_id = ?",
                (body.status, question_id)
            )
        conn.commit()

        # Return updated record
        cursor.execute("SELECT * FROM questions WHERE question_id = ?", (question_id,))
        row = cursor.fetchone()
        result = question_row_to_dict(row)
        conn.close()

        logger.info(
            f"Question {question_id} status updated to '{body.status}'"
            + (f" by {body.reviewed_by}" if body.reviewed_by else "")
        )
        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating question status: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail=f"Failed to update question status: {str(e)}"
        )


@app.get("/api/v1/questions/categories/list", tags=["Questions"])
async def list_question_categories():
    """
    Get list of all question categories and types.
    
    Returns:
        Dict containing categories and their types
    """
    try:
        conn = get_question_db_connection()
        cursor = conn.cursor()
        
        # Get all unique categories
        cursor.execute("""
            SELECT DISTINCT question_category, question_type 
            FROM questions 
            ORDER BY question_category, question_type
        """)
        rows = cursor.fetchall()
        
        # Organize by category
        categories = {}
        for row in rows:
            category = row['question_category']
            qtype = row['question_type']
            
            if category not in categories:
                categories[category] = []
            categories[category].append(qtype)
        
        conn.close()
        
        return {
            "categories": categories,
            "total_categories": len(categories)
        }
        
    except Exception as e:
        logger.error(f"Error listing categories: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail=f"Failed to list categories: {str(e)}"
        )


@app.get("/api/v1/questions/sources/list", tags=["Questions"])
async def list_question_sources():
    """
    Get list of all distinct question sources.

    Returns:
        Dict containing list of sources and counts per source
    """
    try:
        conn = get_question_db_connection()
        cursor = conn.cursor()

        cursor.execute("""
            SELECT source, COUNT(*) as count
            FROM questions
            GROUP BY source
            ORDER BY count DESC
        """)
        rows = cursor.fetchall()
        conn.close()

        sources = [
            {"source": row["source"], "count": row["count"]}
            for row in rows
            if row["source"]
        ]

        logger.info(f"Listed {len(sources)} question sources")
        return {"sources": sources, "total": len(sources)}

    except Exception as e:
        logger.error(f"Error listing sources: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail=f"Failed to list sources: {str(e)}"
        )


# ============================================================================
# Human Evaluation Database Endpoints
# ============================================================================

class EvaluationResponse(BaseModel):
    """Response schema for evaluation question data"""
    id: int
    question_id: str
    question_category: str
    question_type: str
    question_text: str
    options: Optional[List[str]] = None
    correct_answer: float  # Float to support absolute answer types (distance_absolute, scene_counting_absolute)
    source: Optional[str] = None
    scene_id: Optional[str] = None
    media_type: str
    images: Optional[List[str]] = None
    videos: Optional[List[str]] = None
    frame_paths: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None
    status: str = Field(default="pending", description="Review status")
    time_spent_review: float = Field(default=0.0, description="Total seconds spent reviewing")
    human_answer: Optional[float] = Field(default=None, description="Human answer - float for absolute types, index for multiple choice")
    user_time_spent: Optional[float] = Field(default=None, description="Time spent by current user on this question")
    created_at: str


class EvaluationAnswerUpdate(BaseModel):
    """Request schema for updating human answer"""
    human_answer: float = Field(..., description="Human answer - float value for absolute types, index for multiple choice")
    username: str = Field(..., description="Username of person answering")
    time_spent: float = Field(default=0.0, description="Seconds spent on this question")


class EvaluationQueryResponse(BaseModel):
    """Response schema for evaluation query endpoint"""
    total: int = Field(..., description="Total number of questions matching filter")
    count: int = Field(..., description="Number of questions returned")
    data: List[EvaluationResponse]


def get_evaluation_db_connection() -> sqlite3.Connection:
    """
    Create and return a connection to the evaluation database.
    
    Returns:
        sqlite3.Connection: Database connection
    """
    try:
        conn = sqlite3.connect(EVALUATION_DATABASE_PATH)
        conn.row_factory = sqlite3.Row
        logger.info("Evaluation database connection established")
        return conn
    except Exception as e:
        logger.error(f"Failed to connect to evaluation database: {str(e)}")
        logger.error(traceback.format_exc())
        raise


def evaluation_row_to_dict(row: sqlite3.Row, user_answer_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Convert evaluation SQLite row to dictionary with JSON parsing.
    
    Args:
        row: SQLite row object
        user_answer_data: Optional dict with user's answer data (human_answer, time_spent)
        
    Returns:
        Dict containing row data with parsed JSON fields
    """
    data = dict(row)
    
    # Parse JSON fields
    json_fields = ['options', 'images', 'videos', 'frame_paths', 'metadata']
    
    for field in json_fields:
        if field in data and data[field]:
            try:
                data[field] = json.loads(data[field])
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse JSON field: {field}")
                data[field] = None
        else:
            data[field] = None
    
    # Ensure status always has a value
    if "status" not in data or data["status"] is None:
        data["status"] = "pending"
    
    # Set user-specific answer data
    if user_answer_data:
        data["human_answer"] = user_answer_data.get("human_answer")
        data["user_time_spent"] = user_answer_data.get("time_spent", 0.0)
    else:
        data["human_answer"] = None
        data["user_time_spent"] = None
    
    return data


@app.get("/api/v1/evaluation", tags=["Evaluation"])
async def get_evaluation_questions(
    top_k: int = Query(default=10, description="Number of questions to return"),
    offset: int = Query(default=0, description="Offset for pagination"),
    question_category: Optional[str] = Query(default=None, description="Filter by category"),
    question_type: Optional[str] = Query(default=None, description="Filter by type"),
    source: Optional[str] = Query(default=None, description="Filter by source"),
    media_type: Optional[str] = Query(default=None, description="Filter by media type"),
    answered: Optional[bool] = Query(default=None, description="Filter by answered status for this user: true=answered, false=unanswered"),
    username: Optional[str] = Query(default=None, description="Username to check answered status for")
) -> EvaluationQueryResponse:
    """
    Get evaluation questions with optional filtering and pagination.
    Answered status is per-user based on the user_answers table.
    
    Args:
        top_k: Number of questions to return (default 10)
        offset: Offset for pagination (default 0)
        question_category: Filter by category (optional)
        question_type: Filter by type (optional)
        source: Filter by source (optional)
        media_type: Filter by media type (optional)
        answered: Filter by answered status for this user (optional)
        username: Username to filter answered status (required if answered is set)
        
    Returns:
        EvaluationQueryResponse containing total count and list of questions
    """
    try:
        conn = get_evaluation_db_connection()
        cursor = conn.cursor()
        
        # Build WHERE clause for questions table
        conditions = []
        params: List[Any] = []
        
        if question_category:
            conditions.append("q.question_category = ?")
            params.append(question_category)
        if question_type:
            conditions.append("q.question_type = ?")
            params.append(question_type)
        if source:
            conditions.append("q.source = ?")
            params.append(source)
        if media_type:
            conditions.append("q.media_type = ?")
            params.append(media_type)
        
        # Handle per-user answered filter
        join_clause = ""
        if answered is not None and username:
            if answered:
                # Only questions this user has answered
                join_clause = "INNER JOIN user_answers ua ON q.question_id = ua.question_id AND ua.username = ?"
                params.insert(0, username)  # Join param comes first
            else:
                # Only questions this user has NOT answered
                join_clause = f"""
                    LEFT JOIN user_answers ua ON q.question_id = ua.question_id AND ua.username = ?
                """
                conditions.append("ua.id IS NULL")
                params.insert(0, username)
        
        where_clause = " AND ".join(conditions) if conditions else "1=1"
        
        # Get total count
        count_query = f"""
            SELECT COUNT(*) as total 
            FROM questions q
            {join_clause}
            WHERE {where_clause}
        """
        cursor.execute(count_query, params)
        total = cursor.fetchone()["total"]
        
        # Get paginated data
        data_query = f"""
            SELECT q.* FROM questions q
            {join_clause}
            WHERE {where_clause}
            ORDER BY q.id ASC
            LIMIT ? OFFSET ?
        """
        cursor.execute(data_query, params + [top_k, offset])
        rows = cursor.fetchall()
        
        # Get user answers for these questions if username provided
        user_answers_map: Dict[str, Dict[str, Any]] = {}
        if username and rows:
            question_ids = [dict(row)["question_id"] for row in rows]
            placeholders = ",".join(["?" for _ in question_ids])
            cursor.execute(f"""
                SELECT question_id, human_answer, time_spent
                FROM user_answers
                WHERE username = ? AND question_id IN ({placeholders})
            """, [username] + question_ids)
            for ua_row in cursor.fetchall():
                user_answers_map[ua_row["question_id"]] = {
                    "human_answer": ua_row["human_answer"],
                    "time_spent": ua_row["time_spent"]
                }
        
        conn.close()
        
        questions = [
            EvaluationResponse(**evaluation_row_to_dict(row, user_answers_map.get(dict(row)["question_id"])))
            for row in rows
        ]
        
        logger.info(f"Retrieved {len(questions)} evaluation questions (total: {total}) for user: {username}")
        return EvaluationQueryResponse(total=total, count=len(questions), data=questions)
        
    except Exception as e:
        logger.error(f"Error fetching evaluation questions: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch evaluation questions: {str(e)}"
        )


@app.get("/api/v1/evaluation/{question_id}", tags=["Evaluation"])
async def get_evaluation_question_by_id(question_id: str) -> EvaluationResponse:
    """
    Get a specific evaluation question by its question_id.
    
    Args:
        question_id: The unique question identifier
        
    Returns:
        EvaluationResponse containing the question data
    """
    try:
        conn = get_evaluation_db_connection()
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM questions WHERE question_id = ?", (question_id,))
        row = cursor.fetchone()
        conn.close()
        
        if not row:
            raise HTTPException(status_code=404, detail=f"Question not found: {question_id}")
        
        return EvaluationResponse(**evaluation_row_to_dict(row))
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching evaluation question {question_id}: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch evaluation question: {str(e)}"
        )


@app.patch("/api/v1/evaluation/{question_id}/answer", tags=["Evaluation"])
async def update_evaluation_answer(
    question_id: str,
    body: EvaluationAnswerUpdate
) -> EvaluationResponse:
    """
    Update the human answer for an evaluation question (per-user).
    Stores the answer in user_answers table, allowing multiple users to answer the same question.
    
    Args:
        question_id: The unique question identifier
        body: EvaluationAnswerUpdate containing human_answer, username, and time_spent
        
    Returns:
        Updated EvaluationResponse with user's answer
    """
    try:
        conn = get_evaluation_db_connection()
        cursor = conn.cursor()
        
        # Check if question exists
        cursor.execute("SELECT * FROM questions WHERE question_id = ?", (question_id,))
        row = cursor.fetchone()
        
        if not row:
            conn.close()
            raise HTTPException(status_code=404, detail=f"Question not found: {question_id}")
        
        # Insert or update user answer in user_answers table
        cursor.execute("""
            INSERT INTO user_answers (question_id, username, human_answer, time_spent, updated_at)
            VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(question_id, username) DO UPDATE SET
                human_answer = excluded.human_answer,
                time_spent = user_answers.time_spent + excluded.time_spent,
                updated_at = CURRENT_TIMESTAMP
        """, (question_id, body.username, body.human_answer, body.time_spent))
        
        conn.commit()
        
        # Fetch user's answer data
        cursor.execute("""
            SELECT human_answer, time_spent
            FROM user_answers
            WHERE question_id = ? AND username = ?
        """, (question_id, body.username))
        user_answer_row = cursor.fetchone()
        user_answer_data = {
            "human_answer": user_answer_row["human_answer"],
            "time_spent": user_answer_row["time_spent"]
        } if user_answer_row else None
        
        conn.close()
        
        logger.info(f"Updated human_answer for question {question_id} by user {body.username} to {body.human_answer}")
        return EvaluationResponse(**evaluation_row_to_dict(row, user_answer_data))
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating evaluation answer for {question_id}: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail=f"Failed to update evaluation answer: {str(e)}"
        )


@app.delete("/api/v1/evaluation/{question_id}", tags=["Evaluation"])
async def delete_evaluation_question(question_id: str) -> dict:
    """
    Delete an evaluation question from the database.
    This also removes any associated user answers.
    
    Args:
        question_id: The unique question identifier
        
    Returns:
        Dict with success message
    """
    try:
        conn = get_evaluation_db_connection()
        cursor = conn.cursor()
        
        # Check if question exists
        cursor.execute("SELECT question_id FROM questions WHERE question_id = ?", (question_id,))
        row = cursor.fetchone()
        
        if not row:
            conn.close()
            raise HTTPException(status_code=404, detail=f"Question not found: {question_id}")
        
        # Delete user answers first (foreign key constraint)
        cursor.execute("DELETE FROM user_answers WHERE question_id = ?", (question_id,))
        deleted_answers = cursor.rowcount
        
        # Delete the question
        cursor.execute("DELETE FROM questions WHERE question_id = ?", (question_id,))
        
        conn.commit()
        conn.close()
        
        logger.info(f"Deleted evaluation question {question_id} and {deleted_answers} associated user answers")
        return {
            "success": True,
            "message": f"Question {question_id} deleted successfully",
            "deleted_answers": deleted_answers
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting evaluation question {question_id}: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail=f"Failed to delete evaluation question: {str(e)}"
        )


@app.get("/api/v1/evaluation/sources/list", tags=["Evaluation"])
async def list_evaluation_sources():
    """
    Get list of all distinct evaluation question sources.
    
    Returns:
        Dict containing list of sources and counts per source
    """
    try:
        conn = get_evaluation_db_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT source, COUNT(*) as count
            FROM questions
            GROUP BY source
            ORDER BY count DESC
        """)
        rows = cursor.fetchall()
        conn.close()
        
        sources = [
            {"source": row["source"], "count": row["count"]}
            for row in rows
            if row["source"]
        ]
        
        logger.info(f"Listed {len(sources)} evaluation sources")
        return {"sources": sources, "total": len(sources)}
        
    except Exception as e:
        logger.error(f"Error listing evaluation sources: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail=f"Failed to list evaluation sources: {str(e)}"
        )


@app.get("/api/v1/evaluation/stats", tags=["Evaluation"])
async def get_evaluation_stats(
    username: Optional[str] = Query(default=None, description="Username to get stats for")
):
    """
    Get statistics for evaluation progress (per-user if username provided).
    
    Args:
        username: Username to get stats for (optional, if not provided returns global stats)
    
    Returns:
        Dict containing total, answered, unanswered counts, breakdown by category, and time stats
    """
    try:
        conn = get_evaluation_db_connection()
        cursor = conn.cursor()
        
        # Total questions
        cursor.execute("SELECT COUNT(*) as total FROM questions")
        total = cursor.fetchone()["total"]
        
        if username:
            # Per-user stats
            cursor.execute("""
                SELECT COUNT(*) as answered
                FROM user_answers
                WHERE username = ?
            """, (username,))
            answered = cursor.fetchone()["answered"]
            
            # Total time spent by this user
            cursor.execute("""
                SELECT COALESCE(SUM(time_spent), 0) as total_time
                FROM user_answers
                WHERE username = ?
            """, (username,))
            total_time = cursor.fetchone()["total_time"]
            
            # By category for this user
            cursor.execute("""
                SELECT 
                    q.question_category,
                    COUNT(DISTINCT q.question_id) as total,
                    COUNT(DISTINCT ua.question_id) as answered
                FROM questions q
                LEFT JOIN user_answers ua ON q.question_id = ua.question_id AND ua.username = ?
                GROUP BY q.question_category
            """, (username,))
            by_category = [
                {
                    "category": row["question_category"],
                    "total": row["total"],
                    "answered": row["answered"],
                    "unanswered": row["total"] - row["answered"]
                }
                for row in cursor.fetchall()
            ]
            
            # Average time per question for this user
            avg_time = total_time / answered if answered > 0 else 0
            
        else:
            # Global stats (count unique question_id answered by any user)
            cursor.execute("""
                SELECT COUNT(DISTINCT question_id) as answered
                FROM user_answers
            """)
            answered = cursor.fetchone()["answered"]
            
            # Total time across all users
            cursor.execute("SELECT COALESCE(SUM(time_spent), 0) as total_time FROM user_answers")
            total_time = cursor.fetchone()["total_time"]
            
            # By category (global)
            cursor.execute("""
                SELECT 
                    q.question_category,
                    COUNT(DISTINCT q.question_id) as total,
                    COUNT(DISTINCT ua.question_id) as answered
                FROM questions q
                LEFT JOIN user_answers ua ON q.question_id = ua.question_id
                GROUP BY q.question_category
            """)
            by_category = [
                {
                    "category": row["question_category"],
                    "total": row["total"],
                    "answered": row["answered"],
                    "unanswered": row["total"] - row["answered"]
                }
                for row in cursor.fetchall()
            ]
            
            # Total answers (including duplicates from different users)
            cursor.execute("SELECT COUNT(*) as total_answers FROM user_answers")
            total_answers = cursor.fetchone()["total_answers"]
            avg_time = total_time / total_answers if total_answers > 0 else 0
        
        conn.close()
        
        return {
            "total": total,
            "answered": answered,
            "unanswered": total - answered,
            "by_category": by_category,
            "total_time_spent": round(total_time, 2),
            "avg_time_per_question": round(avg_time, 2),
            "username": username
        }
        
    except Exception as e:
        logger.error(f"Error getting evaluation stats: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get evaluation stats: {str(e)}"
        )


# Main entry point for testing
if __name__ == "__main__":
    import uvicorn
    
    logger.info("Starting FastAPI application in development mode")
    
    # Test database connection
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) as count FROM av_data")
        count = cursor.fetchone()['count']
        logger.info(f"Database connected successfully. Total records: {count}")
        conn.close()
    except Exception as e:
        logger.error(f"Database connection test failed: {str(e)}")
        logger.error(traceback.format_exc())
    
    # Test question database connection
    try:
        conn = get_question_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) as count FROM questions")
        count = cursor.fetchone()['count']
        logger.info(f"Question database connected successfully. Total questions: {count}")
        conn.close()
    except Exception as e:
        logger.error(f"Question database connection test failed: {str(e)}")
        logger.error(traceback.format_exc())
    
    # Run the application
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
