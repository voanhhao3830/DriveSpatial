import cv2
import logging
import os
import math
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import Dict, Any, List, Optional, Tuple
# import FigureCanvas
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from sources.data_collection.utils.constant import STATIC_LABELS
from sources.data_collection.enum import (
    Relation, 
    IntrinsicInteraction,
    ExtrinsicInteraction
)

logger = logging.getLogger(__name__)

from sources.data_collection.enum.data_sources import DataSource

class InteractionExtractor:
    @staticmethod
    def _angle_diff(angle1: float, angle2: float) -> float:
        """
        Calculate the smallest difference between two angles in radians.
        
        Args:
            angle1: First angle in radians
            angle2: Second angle in radians
            
        Returns:
            Difference in radians, range [-pi, pi]
        """
        diff = angle1 - angle2
        while diff > np.pi:
            diff -= 2 * np.pi
        while diff < -np.pi:
            diff += 2 * np.pi
        return diff

    @staticmethod
    def _transform_bbox_to_current_frame(
        prev_bbox: np.ndarray,
        prev_ego_pose: np.ndarray,
        curr_ego_pose_inv: np.ndarray
    ) -> np.ndarray:
        """
        Transform a 3D bounding box from previous frame's ego coordinate to current frame's ego coordinate.
        
        Args:
            prev_bbox: [x, y, z, length, width, height, yaw] in previous ego frame
            prev_ego_pose: Previous frame's ego pose (lidar→global, 4x4)
            curr_ego_pose_inv: Current frame's inverse ego pose (global→lidar, 4x4)
            
        Returns:
            Transformed bbox [x, y, z, length, width, height, yaw] in current ego frame
        """
        # Extract position and dimensions
        x, y, z = float(prev_bbox[0]), float(prev_bbox[1]), float(prev_bbox[2])
        length, width, height = float(prev_bbox[3]), float(prev_bbox[4]), float(prev_bbox[5])
        yaw = float(prev_bbox[6])
        
        # Transform position: prev_ego → global → curr_ego
        prev_pos_homo = np.array([x, y, z, 1.0], dtype=float)
        global_pos = prev_ego_pose @ prev_pos_homo
        curr_pos_homo = curr_ego_pose_inv @ global_pos
        curr_x, curr_y, curr_z = curr_pos_homo[:3]
        
        # Transform heading: need to account for ego rotation
        # Extract rotation angles from transformation matrices
        prev_ego_rot = prev_ego_pose[:3, :3]
        curr_ego_rot_inv = curr_ego_pose_inv[:3, :3]
        
        # Transform heading vector
        heading_vec = np.array([np.cos(yaw), np.sin(yaw), 0.0], dtype=float)
        # Transform: prev_ego → global → curr_ego
        global_heading = prev_ego_rot @ heading_vec
        curr_heading = curr_ego_rot_inv @ global_heading
        
        # Calculate new yaw in current frame
        curr_yaw = np.arctan2(curr_heading[1], curr_heading[0])
        
        # Dimensions stay the same
        return np.array([curr_x, curr_y, curr_z, length, width, height, curr_yaw], dtype=float)

    @staticmethod
    def _calculate_ego_velocity(
        prev_ego_pose: np.ndarray,
        curr_ego_pose: np.ndarray,
        time_delta: float
    ) -> np.ndarray:
        """
        Calculate ego vehicle velocity from pose changes in the ego's current coordinate frame.
        
        Args:
            prev_ego_pose: Previous frame's ego pose (lidar→global, 4x4)
            curr_ego_pose: Current frame's ego pose (lidar→global, 4x4)
            time_delta: Time difference between frames (seconds)
            
        Returns:
            Velocity [vx, vy] in current ego frame (m/s)
        """
        if time_delta <= 0:
            return np.array([0.0, 0.0], dtype=np.float32)
        
        # Extract positions in global frame
        prev_pos_global = prev_ego_pose[:3, 3]
        curr_pos_global = curr_ego_pose[:3, 3]
        
        # Calculate displacement in global frame
        displacement_global = curr_pos_global - prev_pos_global
        
        # Transform displacement to current ego frame
        curr_ego_rot_inv = np.linalg.inv(curr_ego_pose[:3, :3])
        displacement_ego = curr_ego_rot_inv @ displacement_global
        
        # Calculate velocity in ego frame (only x, y components)
        vx = float(displacement_ego[0] / time_delta)
        vy = float(displacement_ego[1] / time_delta)
        
        return np.array([vx, vy], dtype=np.float32)

    @staticmethod
    def _extract_external_interactions(
        frame_idx: int,
        track_ids: List[str],
        labels: List[str],
        bboxes: List[List[float]],
        velocities: Optional[np.ndarray],
        track_history: Dict[str, List[Dict[str, Any]]],
        curr_ego_pose_inv: np.ndarray,
        internal_interactions: List[Dict[str, Any]],  # NEW: Internal interactions for lane change detection
        velocity_threshold: float,
        lead_follow_distance_threshold: float,
        same_direction_angle_threshold: float,
        distance_threshold: float = 8.0, # meters
        opposite_direction_angle_threshold: float = 150.0,  # degrees for opposite direction
        perpendicular_angle_threshold: float = 20.0,  # degrees tolerance for perpendicular
        yielding_distance_threshold: float = 10.0,  # meters for yielding detection
        same_lane_lateral_threshold: float = 2.0,  # meters - max lateral offset for same lane
        approaching_distance_threshold: float = 20.0,  # meters - max distance for approaching detection
        crossing_distance_threshold: float = 20.0,  # meters - max distance for crossing detection
        passing_distance_threshold: float = 20.0,  # meters - max distance for passing detection
    ) -> List[Dict[str, Any]]:
        """
        Extract external interactions (between objects).
        
        Important: This function performs ego-motion compensation when checking
        for overtaking behavior by transforming previous frame coordinates to the
        current frame's ego coordinate system.
        
        Args:
            frame_idx: Current frame index
            track_ids: List of track IDs
            labels: List of object labels
            bboxes: List of 3D bounding boxes [x, y, z, length, width, height, yaw]
            velocities: Optional array of velocities [vx, vy]
            track_history: Dictionary mapping track_id to history
            curr_ego_pose_inv: Current frame's inverse ego pose
            internal_interactions: List of internal interactions (for lane change detection)
            velocity_threshold: Speed below which object is considered stopped
            lead_follow_distance_threshold: Max distance for interaction detection
            same_direction_angle_threshold: Max angle difference for same direction
            opposite_direction_angle_threshold: Min angle difference for opposite direction
            perpendicular_angle_threshold: Tolerance for perpendicular detection
            yielding_distance_threshold: Max distance for yielding detection
            same_lane_lateral_threshold: Max lateral displacement for same lane detection
            approaching_distance_threshold: Max distance for approaching interaction detection
            crossing_distance_threshold: Max distance for crossing interaction detection
        
        Returns:
            List of interaction dictionaries containing:
                - source_object: {track_id, label}
                - target_object: {track_id, label}
                - interaction_type: ExtrinsicInteraction enum value
                - details: additional numeric details
        """
        interactions = []
        
        # Create lookup for internal interactions (for lane change detection)
        internal_interaction_map = {}
        for interaction in internal_interactions:
            track_id = interaction['track_id']
            actions = interaction.get('actions', [])
            internal_interaction_map[track_id] = actions
        
        # Analyze pairs of objects
        for i in range(len(track_ids)):
            for j in range(len(track_ids)):
                if i == j:
                    continue
                
                track_i = track_ids[i]
                track_j = track_ids[j]
                bbox_i = bboxes[i]
                bbox_j = bboxes[j]
                label_i = str(labels[i])
                label_j = str(labels[j])
                
                # Get positions
                pos_i = np.array([float(bbox_i[0]), float(bbox_i[1])], dtype=float)
                pos_j = np.array([float(bbox_j[0]), float(bbox_j[1])], dtype=float)
                yaw_i = float(bbox_i[6])
                yaw_j = float(bbox_j[6])
                
                # Calculate distance
                distance = np.linalg.norm(pos_j - pos_i)
                
                if distance > lead_follow_distance_threshold:
                    continue
                
                # Get velocities
                if velocities is not None:
                    vel_i = np.array([float(velocities[i][0]), float(velocities[i][1])], dtype=float)
                    vel_j = np.array([float(velocities[j][0]), float(velocities[j][1])], dtype=float)
                    speed_i = np.linalg.norm(vel_i)
                    speed_j = np.linalg.norm(vel_j)
                else:
                    speed_i = None
                    speed_j = None
                
                # Check if moving in similar direction
                heading_diff = abs(np.degrees(InteractionExtractor._angle_diff(yaw_i, yaw_j)))
                same_direction = heading_diff < same_direction_angle_threshold
                similar_direction = heading_diff < 45.0  # Relaxed for overtake/passing (allows lane change angles)
                opposite_direction = heading_diff > opposite_direction_angle_threshold
                perpendicular = abs(heading_diff - 90.0) < perpendicular_angle_threshold or abs(heading_diff - 270.0) < perpendicular_angle_threshold
                
                # Check if i is ahead of j in the direction of travel
                heading_vec_j = np.array([np.cos(yaw_j), np.sin(yaw_j)], dtype=float)
                heading_vec_i = np.array([np.cos(yaw_i), np.sin(yaw_i)], dtype=float)
                displacement_vec = pos_i - pos_j
                forward_distance = float(np.dot(displacement_vec, heading_vec_j))
                
                # Calculate lateral displacement (perpendicular distance for lane detection)
                # Right vector perpendicular to j's heading
                right_vec_j = np.array([np.sin(yaw_j), -np.cos(yaw_j)], dtype=float)
                lateral_distance = float(abs(np.dot(displacement_vec, right_vec_j)))
                
                # Check if vehicles are in the same lane
                same_lane = lateral_distance < same_lane_lateral_threshold
                
                euclidean_distance = np.linalg.norm(pos_i - pos_j)
                if euclidean_distance > distance_threshold:
                    continue
                
                # Determine interaction type
                interaction_type = None
                details = {
                    'distance': distance,
                    'heading_difference': heading_diff,
                    'forward_distance': forward_distance,
                    'lateral_distance': lateral_distance,
                    'same_lane': same_lane,
                    'same_direction': same_direction,
                    'similar_direction': similar_direction,
                }
                
                if speed_i is not None and speed_j is not None:
                    details['speed_i'] = speed_i
                    details['speed_j'] = speed_j
                    
                    # Both stopped - skip
                    if speed_i < velocity_threshold and speed_j < velocity_threshold:
                        continue
                    
                    # ==================== PREPARE LANE CHANGE DETECTION ====================
                    # Check for lane change of SOURCE object i using IntrinsicInteraction
                    is_i_lane_changing = False
                    is_i_turning = False
                    
                    # Get internal interactions for source object i
                    i_actions = internal_interaction_map.get(track_i, [])
                    
                    # Check if i is performing lane change
                    is_i_lane_changing = (
                        IntrinsicInteraction.LANE_CHANGE_LEFT in i_actions or 
                        IntrinsicInteraction.LANE_CHANGE_RIGHT in i_actions
                    )
                    
                    # Check if i is turning
                    is_i_turning = (
                        IntrinsicInteraction.TURN_LEFT in i_actions or 
                        IntrinsicInteraction.TURN_RIGHT in i_actions or
                        IntrinsicInteraction.U_TURN in i_actions
                    )
                    
                    # Calculate lateral change for debugging/details
                    prev_lateral_distance = None
                    lateral_change = None
                    if len(track_history.get(track_i, [])) >= 2 and len(track_history.get(track_j, [])) >= 2:
                        full_history_i = track_history.get(track_i, [])
                        full_history_j = track_history.get(track_j, [])
                        history_i = [h for h in full_history_i if h['frame_idx'] <= frame_idx]
                        history_j = [h for h in full_history_j if h['frame_idx'] <= frame_idx]
                        
                        if len(history_i) >= 2 and len(history_j) >= 2:
                            prev_i = history_i[-2]
                            prev_j = history_j[-2]
                            
                            # Transform to current frame
                            prev_bbox_i_transformed = InteractionExtractor._transform_bbox_to_current_frame(
                                prev_bbox=prev_i['bbox_3d'],
                                prev_ego_pose=prev_i['ego_pose'],
                                curr_ego_pose_inv=curr_ego_pose_inv
                            )
                            prev_bbox_j_transformed = InteractionExtractor._transform_bbox_to_current_frame(
                                prev_bbox=prev_j['bbox_3d'],
                                prev_ego_pose=prev_j['ego_pose'],
                                curr_ego_pose_inv=curr_ego_pose_inv
                            )
                            
                            prev_pos_i = np.array([float(prev_bbox_i_transformed[0]), float(prev_bbox_i_transformed[1])], dtype=float)
                            prev_pos_j = np.array([float(prev_bbox_j_transformed[0]), float(prev_bbox_j_transformed[1])], dtype=float)
                            prev_yaw_j = float(prev_bbox_j_transformed[6])
                            
                            # Calculate previous lateral distance (for debugging/details only)
                            prev_right_vec_j = np.array([np.sin(prev_yaw_j), -np.cos(prev_yaw_j)], dtype=float)
                            prev_displacement_vec = prev_pos_i - prev_pos_j
                            prev_lateral_distance = float(abs(np.dot(prev_displacement_vec, prev_right_vec_j)))
                            lateral_change = abs(lateral_distance - prev_lateral_distance)
                    
                    # Add to details
                    if prev_lateral_distance is not None:
                        details['prev_lateral_distance'] = prev_lateral_distance
                    if lateral_change is not None:
                        details['lateral_change'] = lateral_change
                    details['is_i_lane_changing'] = is_i_lane_changing
                    details['is_i_turning'] = is_i_turning
                    details['i_actions'] = [str(action) for action in i_actions]
                    
                    # ==================== INTERACTIONS (NO STRICT DIRECTION REQUIREMENT) ====================
                    
                    # i is ahead of j (OVERTAKE/PASSING happens here - source has overtaken target)
                    if forward_distance > 1.0 :  # i is ahead
                        if speed_i >= velocity_threshold:
                            # i is moving
                            
                            # LEAD: Same lane, both moving, i ahead of j (requires same_direction)
                            if same_lane and same_direction and speed_j >= velocity_threshold:
                                interaction_type = ExtrinsicInteraction.LEAD
                            
                            # OVERTAKE or PASSING:
                            else:
                                # Check if i is meaningfully faster (or j is stopped)
                                is_faster = (speed_j < velocity_threshold) or (speed_i > speed_j + 0.5)
                                
                                if is_faster and lateral_distance <= passing_distance_threshold:
                                    # OVERTAKE: i (source) has overtaken j (target) - i is now ahead with lane change and not same lane
                                    if (is_i_lane_changing or is_i_turning or perpendicular):
                                        interaction_type = ExtrinsicInteraction.OVERTAKE
                                        if speed_j < velocity_threshold:
                                            details['overtake_type'] = 'overtaken_stopped_vehicle'
                                        else:
                                            details['overtake_type'] = 'overtaken_moving_vehicle'
                                        
                                        # YIELDING: If i overtakes j, then j yields to i
                                        # Create reciprocal yielding interaction: j yields to i
                                        yielding_details = details.copy()
                                        yielding_details['yielding_reason'] = 'overtaken_by_vehicle'
                                        yielding_details['overtaking_vehicle'] = track_i
                                        interactions.append({
                                            'source_object': {'track_id': track_j, 'label': label_j},
                                            'target_object': {'track_id': track_i, 'label': label_i},
                                            'interaction_type': ExtrinsicInteraction.YIELDING,
                                            'details': yielding_details
                                        })
                                    # PASSING: i is ahead in parallel lane (requires similar direction)
                                    elif not same_lane and similar_direction:
                                        interaction_type = ExtrinsicInteraction.PASSING
                                        if speed_j < velocity_threshold:
                                            details['passing_type'] = 'parallel_stopped_vehicle'
                                        else:
                                            details['passing_type'] = 'parallel_lanes'
                    
                    # j is ahead of i (i is behind j)
                    elif forward_distance < -1.0:  # j is ahead, i is behind
                        if speed_i >= velocity_threshold:
                            # i is moving
                            
                            # FOLLOW: Same lane, both moving, i behind j (requires same_direction)
                            if same_lane and same_direction and speed_j >= velocity_threshold:
                                interaction_type = ExtrinsicInteraction.FOLLOW
                    
                    # Side by side (CO_MOVING): small forward distance, different lanes, similar speeds
                    elif abs(forward_distance) <= 1.0:  # Side by side (neither clearly ahead)
                        if speed_i >= velocity_threshold and speed_j >= velocity_threshold:
                            # Both moving
                            speed_diff = abs(speed_i - speed_j)
                            
                            # CO_MOVING: Different lanes, similar speeds (implies similar direction)
                            if not same_lane and speed_diff < 10 and similar_direction:
                                interaction_type = ExtrinsicInteraction.CO_MOVING
                                details['speed_difference'] = speed_diff
                                details['co_moving_type'] = 'side_by_side'
                    
                    # ==================== DIFFERENT DIRECTION INTERACTIONS ====================
                    # Only check if no interaction found yet
                    if not interaction_type and (opposite_direction or perpendicular):
                        # Skip if both stopped
                        if speed_i < velocity_threshold and speed_j < velocity_threshold:
                            continue
                        
                        # Calculate if vehicles are approaching (distance decreasing)
                        is_approaching = False
                        if len(track_history.get(track_i, [])) >= 2 and len(track_history.get(track_j, [])) >= 2:
                            full_history_i = track_history.get(track_i, [])
                            full_history_j = track_history.get(track_j, [])
                            history_i = [h for h in full_history_i if h['frame_idx'] <= frame_idx]
                            history_j = [h for h in full_history_j if h['frame_idx'] <= frame_idx]
                            
                            if len(history_i) >= 2 and len(history_j) >= 2:
                                prev_i = history_i[-2]
                                prev_j = history_j[-2]
                                
                                # Transform to current frame
                                prev_bbox_i_transformed = InteractionExtractor._transform_bbox_to_current_frame(
                                    prev_bbox=prev_i['bbox_3d'],
                                    prev_ego_pose=prev_i['ego_pose'],
                                    curr_ego_pose_inv=curr_ego_pose_inv
                                )
                                prev_bbox_j_transformed = InteractionExtractor._transform_bbox_to_current_frame(
                                    prev_bbox=prev_j['bbox_3d'],
                                    prev_ego_pose=prev_j['ego_pose'],
                                    curr_ego_pose_inv=curr_ego_pose_inv
                                )
                                
                                prev_pos_i = np.array([float(prev_bbox_i_transformed[0]), float(prev_bbox_i_transformed[1])], dtype=float)
                                prev_pos_j = np.array([float(prev_bbox_j_transformed[0]), float(prev_bbox_j_transformed[1])], dtype=float)
                                prev_distance = np.linalg.norm(prev_pos_i - prev_pos_j)
                                
                                is_approaching = distance < prev_distance
                                details['previous_distance'] = prev_distance
                                details['distance_change'] = distance - prev_distance
                    
                        # APPROACHING: opposite direction, both moving, distance decreasing, within threshold
                        if (opposite_direction and 
                            speed_i >= velocity_threshold and 
                            speed_j >= velocity_threshold and 
                            is_approaching and 
                            distance < approaching_distance_threshold):
                            interaction_type = ExtrinsicInteraction.APPROACHING
                            details['approaching_threshold'] = approaching_distance_threshold
                        
                        # VEHICLE_CROSSING: perpendicular paths, both moving, close proximity, within threshold
                        elif ( opposite_direction and
                                speed_i >= velocity_threshold and 
                                speed_j >= velocity_threshold and 
                                distance < crossing_distance_threshold):
                            interaction_type = ExtrinsicInteraction.CROSSING
                            details['crossing_threshold'] = crossing_distance_threshold
                    
                    # ==================== WAITING & YIELDING INTERACTIONS ====================
                    # Only check if no interaction found yet
                    if not interaction_type:
                        # WAITING: i is stopped near pedestrians (need pedestrian in label_j)
                        if speed_i < velocity_threshold and 'pedestrian' in label_j.lower():
                            if distance < yielding_distance_threshold:
                                interaction_type = ExtrinsicInteraction.WAITING
                
                if interaction_type:
                    interactions.append({
                        'source_object': {'track_id': track_i, 'label': label_i},
                        'target_object': {'track_id': track_j, 'label': label_j},
                        'interaction_type': interaction_type,
                        'details': details
                    })
        

        return interactions

    @staticmethod
    def _extract_internal_interactions(
        frame_idx: int,
        track_ids: List[str],
        labels: List[str],
        bboxes: List[List[float]],
        velocities: Optional[np.ndarray],
        track_history: Dict[str, List[Dict[str, Any]]],
        curr_ego_pose_inv: np.ndarray,
        velocity_threshold: float,
        acceleration_threshold: float,
        turn_angle_threshold: float,
        u_turn_angle_threshold: float,
        lane_change_lateral_threshold: float,
        moving_cos_angle_threshold: float = 60,
    ) -> List[Dict[str, Any]]:
        """
        Extract internal interactions (individual object actions).
        
        Important: This function performs ego-motion compensation. When comparing
        positions across frames, previous frame coordinates are transformed to the
        current frame's ego coordinate system using:
            prev_in_curr = curr_ego_inv @ prev_ego @ prev_pos
        
        This ensures that displacement and motion calculations account for the
        ego vehicle's own movement.
        
        Returns:
            List of interaction dictionaries containing:
                - track_id: object track ID
                - label: object label
                - actions: list of action strings
                - details: additional numeric details
        """
        interactions = []
        
        for i, track_id in enumerate(track_ids):
            actions = []
            details = {}
            
            bbox = bboxes[i]
            label = str(labels[i])
            # Filter history to only include frames up to and including current frame
            full_history = track_history.get(track_id, [])
            history = [h for h in full_history if h['frame_idx'] <= frame_idx]
            
            # Get current position and heading
            x, y, z = float(bbox[0]), float(bbox[1]), float(bbox[2])
            yaw = float(bbox[6])
            
            # Calculate velocity magnitude
            if velocities is not None:
                vx, vy = float(velocities[i][0]), float(velocities[i][1])
                velocity_mag = np.sqrt(vx**2 + vy**2)
                details['velocity_magnitude'] = velocity_mag
                details['velocity_x'] = vx
                details['velocity_y'] = vy
            else:
                velocity_mag = None
            
            # Need at least 2 frames of history for motion analysis
            if len(history) >= 2:
                prev_data = history[-2]
                curr_data = history[-1]
                # Ensure we're looking at the current frame
                if curr_data['frame_idx'] == frame_idx:
                    # Transform previous frame's bbox to current frame's coordinate system
                    prev_bbox_orig = prev_data['bbox_3d']
                    prev_ego_pose = prev_data['ego_pose']
                    # Transform coordinates to account for ego vehicle motion
                    prev_bbox_in_curr_frame = InteractionExtractor._transform_bbox_to_current_frame(
                        prev_bbox=prev_bbox_orig,
                        prev_ego_pose=prev_ego_pose,
                        curr_ego_pose_inv=curr_ego_pose_inv
                    )
                    
                    prev_x, prev_y = float(prev_bbox_in_curr_frame[0]), float(prev_bbox_in_curr_frame[1])
                    prev_yaw = float(prev_bbox_in_curr_frame[6])
                    
                    # Calculate displacement (now ego-compensated)
                    dx = x - prev_x
                    dy = y - prev_y
                    displacement = np.sqrt(dx**2 + dy**2)
                    details['displacement'] = displacement
                    
                    # Heading change (in degrees) - ego-compensated
                    heading_change = np.degrees(InteractionExtractor._angle_diff(yaw, prev_yaw))
                    details['heading_change'] = heading_change
                    # Check if stopped
                    if velocity_mag is not None:
                        if velocity_mag < velocity_threshold or label in STATIC_LABELS:
                            actions.append(IntrinsicInteraction.STOPPED) # => OK
                        else:
                            # Check movement direction
                            heading_vec = np.array([np.cos(yaw), np.sin(yaw)])
                            velocity_vec = np.array([vx, vy])
                            dot_product = np.dot(heading_vec, velocity_vec) # cos(a) = h.v / |h|.|v|; |h| = 1
                            cos_angle = dot_product / velocity_mag
                            degree_angle = np.degrees(np.arccos(cos_angle))
                            details['cos_angle'] = degree_angle
                            if degree_angle < moving_cos_angle_threshold: # 60 degrees is a turn
                                actions.append(IntrinsicInteraction.MOVING_FORWARD)
                            elif degree_angle > moving_cos_angle_threshold * 2: # 120 degrees is a u-turn
                                actions.append(IntrinsicInteraction.MOVING_BACKWARD)
                            
                            # Detect turns
                            if abs(heading_change) > u_turn_angle_threshold: # TODO: we need to chekc this 
                                actions.append(IntrinsicInteraction.U_TURN)
                            elif heading_change > turn_angle_threshold:
                                actions.append(IntrinsicInteraction.TURN_LEFT)
                            elif heading_change < -turn_angle_threshold:
                                actions.append(IntrinsicInteraction.TURN_RIGHT)
                    
                            # Detect lane change (lateral displacement perpendicular to heading)
                            if displacement > 0.1:  # Some movement occurred
                                # Calculate forward and lateral components relative to previous heading
                                heading_vec = np.array([np.cos(prev_yaw), np.sin(prev_yaw)])
                                displacement_vec = np.array([dx, dy])
                                
                                # Forward displacement along previous heading
                                forward_displacement = np.dot(displacement_vec, heading_vec)
                                
                                # Right vector perpendicular to heading
                                right_vec = np.array([np.sin(prev_yaw), -np.cos(prev_yaw)])
                                lateral_displacement = np.dot(displacement_vec, right_vec)
                                
                                # Adjust lateral displacement for curved motion
                                # When turning, expected lateral deviation = forward_displacement * sin(heading_change_rad)
                                # Subtract this to get true lane change component
                                heading_change_rad = np.radians(heading_change)
                                expected_lateral_from_turn = forward_displacement * np.sin(heading_change_rad)
                                adjusted_lateral = lateral_displacement - expected_lateral_from_turn
                                
                                details['lateral_displacement'] = lateral_displacement
                                details['adjusted_lateral'] = adjusted_lateral
                                
                                # Use adjusted lateral for lane change detection
                                if abs(adjusted_lateral) > lane_change_lateral_threshold:
                                    if adjusted_lateral > 0:
                                        actions.append(IntrinsicInteraction.LANE_CHANGE_RIGHT)
                                    else:
                                        actions.append(IntrinsicInteraction.LANE_CHANGE_LEFT)
                                    
                            # Detect acceleration/deceleration (need 3 frames)
                            if len(history) >= 3 and velocity_mag is not None:
                                prev_prev_data = history[-3]
                                if 'velocity' in prev_data:
                                    prev_vx, prev_vy = float(prev_data['velocity'][0]), float(prev_data['velocity'][1])
                                    prev_velocity_mag = np.sqrt(prev_vx**2 + prev_vy**2)
                                    
                                    acceleration = velocity_mag - prev_velocity_mag
                                    details['acceleration'] = acceleration
                                    
                                    if acceleration > acceleration_threshold:
                                        actions.append(IntrinsicInteraction.ACCELERATE)
                                    elif acceleration < -acceleration_threshold:
                                        actions.append(IntrinsicInteraction.DECELERATE)
            # if actions:
            #     interactions.append({
            #         'track_id': track_id,
            #         'label': label,
            #         'actions': actions,
            #         'details': details
            #     })
            interactions.append({
                    'track_id': track_id,
                    'label': label,
                    'actions': actions,
                    'details': details
                })
        return interactions

    @staticmethod
    def _calculate_velocity_from_history(
        curr_bbox: np.ndarray,
        prev_bbox_transformed: np.ndarray,
        time_delta: float = 0.5
    ) -> np.ndarray:
        """
        Calculate velocity from position change between frames.
        
        Args:
            curr_bbox: Current frame bbox [x, y, z, length, width, height, yaw]
            prev_bbox_transformed: Previous frame bbox transformed to current frame
            time_delta: Time between frames in seconds (default 0.5s = 2Hz)
            
        Returns:
            Velocity [vx, vy] in m/s
        """
        dx = float(curr_bbox[0]) - float(prev_bbox_transformed[0])
        dy = float(curr_bbox[1]) - float(prev_bbox_transformed[1])
        
        vx = dx / time_delta
        vy = dy / time_delta
        
        return np.array([vx, vy], dtype=np.float32)

    @staticmethod
    def _build_track_history(video_data: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """
        Build a history of each track across frames.
        
        Args:
            video_data: List of frame data dictionaries
            
        Returns:
            Dictionary mapping track_id to list of frame data containing:
                - frame_idx: frame index
                - bbox_3d: [x, y, z, length, width, height, yaw]
                - velocity: [vx, vy] if available
                - label: object label
                - ego_pose: 4x4 transformation matrix (lidar→global)
                - ego_pose_inv: 4x4 transformation matrix (global→lidar)
        """
        track_history: Dict[str, List[Dict[str, Any]]] = {}
        
        for frame_idx, frame in enumerate(video_data):
            bbox_3d = frame['annotation']['bbox_3d']
            label_3d = frame['annotation']['label_3d']
            track_3d = frame['annotation']['track_3d']
            velocity = frame['annotation'].get('velocity', None)
            ego_pose = frame['calibration']['ego_pose']
            ego_pose_inv = frame['calibration']['ego_pose_inv']
            
            timestamp = None
            if 'img_timestamp' in frame['calibration']:
                timestamp = frame['calibration']['img_timestamp'][-1]
            
            # Calculate ego yaw from ego_pose rotation matrix
            # ego_yaw = float(np.arctan2(ego_pose[1, 0], ego_pose[0, 0]))
            
            # Calculate ego velocity if we have history
            ego_velocity = np.array([0.0, 0.0], dtype=np.float32)
            if '00000' in track_history and len(track_history['00000']) > 0:
                prev_frame_data = track_history['00000'][-1]
                prev_ego_pose = prev_frame_data['ego_pose']
                if timestamp is not None and prev_frame_data['timestamp'] is not None:
                    time_delta = timestamp - prev_frame_data['timestamp']
                    ego_velocity = InteractionExtractor._calculate_ego_velocity(
                        prev_ego_pose=prev_ego_pose,
                        curr_ego_pose=ego_pose,
                        time_delta=time_delta
                    )
            
            # Add ego vehicle with ID 00000 at origin
            bbox_3d = [[0.0, 0.0, 0.0, 4.5, 2.0, 1.8, math.pi / 2]] + list(bbox_3d)
            label_3d = ['ego'] + list(label_3d)
            track_3d = ['00000'] + list(track_3d)
            if velocity is not None:
                velocity = np.vstack([ego_velocity, velocity])
                
            for i, track_id in enumerate(track_3d):
                if track_id not in track_history:
                    track_history[track_id] = []
                
                track_data = {
                    'frame_idx': frame_idx,
                    'bbox_3d': bbox_3d[i],
                    'label': str(label_3d[i]),
                    'ego_pose': ego_pose,
                    'ego_pose_inv': ego_pose_inv,
                    'timestamp': timestamp,
                }
                
                # import ipdb; ipdb.set_trace()
                if velocity is not None:
                    track_data['velocity'] = velocity[i]
                
                track_history[track_id].append(track_data)
        
        return track_history

    @staticmethod
    def visualize_internal_actions(
        frame: Dict[str, Any],
        internal_interactions: List[Dict[str, Any]],
        view_idx: str = None,
        save_path: str = None,
        show_velocity: bool = True,
        show_details: bool = True
    ):
        """
        Visualize internal actions (individual object behaviors) on 2D bounding boxes.
        
        Args:
            frame: Frame data containing views, annotations
            internal_interactions: List of internal interaction dictionaries
            view_idx: Camera view name (e.g., 'CAM_FRONT')
            save_path: Optional path to save the visualization
            show_velocity: Whether to show velocity information
            show_details: Whether to show detailed metrics
        """
        view_names = list(frame['views'].keys())
        if view_idx is None:
            view_idx = view_names[-1]
        if view_idx not in view_names:
            logger.warning(f"View {view_idx} not found in frame. Available: {view_names}")
            return
        
        idx = view_names.index(view_idx)
        
        bbox_2d = frame['annotation']['bbox_2d'][idx]
        track_2d = frame['annotation']['track_2d'][idx]
        track_3d = frame['annotation']['track_3d']
        label_3d = frame['annotation']['label_3d']
        
        # Load image
        image = plt.imread(frame['views'][view_idx])
        
        # Ego position as a point at bottom center
        img_h, img_w = image.shape[:2]
        ego_point = (img_w / 2, img_h - 30)  # Bottom center point
        
        # Map 3D tracks to 2D bboxes (track_3d to track_2d mapping)
        track_3d_to_2d = {}
        for i, t3d in enumerate(track_3d):
            for j, t2d in enumerate(track_2d):
                # Find corresponding 2D track
                if t3d == t2d:  # They should match in nuScenes
                    track_3d_to_2d[t3d] = bbox_2d[j]
                    break
        # Create interaction lookup by track_id
        interaction_map = {inter['track_id']: inter for inter in internal_interactions}
        
        # Create figure
        fig, ax = plt.subplots(1, figsize=(20, 14))
        ax.imshow(image)
        
        # Color scheme for different action types
        action_colors = {
            'stopped': 'red',
            'moving_forward': 'green',
            'moving_backward': 'orange',
            'left_turn': 'cyan',
            'right_turn': 'magenta',
            'u_turn': 'purple',
            'lane_change_left': 'blue',
            'lane_change_right': 'yellow',
            'accelerating': 'lime',
            'decelerating': 'pink'
        }
        
        # Get ego interaction for color
        ego_interaction = interaction_map.get('00000')
        
        # Draw bounding boxes and annotations for non-ego objects
        for track_id_3d, bbox in track_3d_to_2d.items():
            x1, y1, x2, y2 = bbox
            
            # Get interaction data if available
            interaction = interaction_map.get(track_id_3d)
            
            if interaction:
                # Use color based on primary action
                primary_action = interaction['actions'][0] if interaction['actions'] else 'unknown'
                color = action_colors.get(primary_action, 'white')
                linewidth = 3
            else:
                # No interaction detected
                color = 'gray'
                linewidth = 2
            
            # Draw bounding box
            rect = patches.Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                linewidth=linewidth,
                edgecolor=color,
                facecolor='none'
            )
            ax.add_patch(rect)
            
            # Prepare annotation text
            annotations = []
            
            # Track ID and label
            label = interaction['label'] if interaction else 'No label'
            annotations.append(f"Label: {label}")
            # Actions
            actions_str = ', '.join(interaction['actions']) if interaction else 'No actions'
            annotations.append(f"Actions: {actions_str}")
            
            # Velocity info
            if interaction and show_velocity and 'velocity_magnitude' in interaction['details']:
                vel_mag = interaction['details']['velocity_magnitude']
                annotations.append(f"Speed: {vel_mag:.2f} m/s")
            
            # Detailed metrics
            if interaction and show_details:
                details = interaction['details']
                if 'heading_change' in details:
                    annotations.append(f"Δθ: {details['heading_change']:.1f}°")
                if 'acceleration' in details:
                    annotations.append(f"Accel: {details['acceleration']:.2f} m/s²")
                if 'displacement' in details:
                    annotations.append(f"Disp: {details['displacement']:.2f}m")
                if 'lateral_displacement' in details:
                    lat = details['lateral_displacement']
                    if 'adjusted_lateral' in details:
                        adj_lat = details['adjusted_lateral']
                        annotations.append(f"Lateral: {lat:.2f}m (adj: {adj_lat:.2f}m)")
                    else:
                        annotations.append(f"Lateral: {lat:.2f}m")
                if 'cos_angle' in details:
                    annotations.append(f"Cos: {details['cos_angle']:.2f}")
            
            
            # Draw text box with annotations
            text_str = '\n'.join(annotations)
            ax.text(
                x1, y1 - 10,
                text_str,
                color='white',
                fontsize=9,
                weight='bold',
                verticalalignment='bottom',
                bbox=dict(
                    boxstyle='round,pad=0.5',
                    facecolor=color,
                    alpha=0.8,
                    edgecolor='white',
                    linewidth=1
                )
            )
        
        # Draw ego vehicle as a point
        if ego_interaction:
            primary_action = ego_interaction['actions'][0] if ego_interaction['actions'] else 'unknown'
            ego_color = action_colors.get(primary_action, 'green')
        else:
            ego_color = 'green'
        
        # Draw ego as a circle
        ego_circle = patches.Circle(
            ego_point, 
            radius=15, 
            facecolor=ego_color, 
            edgecolor='white', 
            linewidth=3,
            alpha=0.9
        )
        ax.add_patch(ego_circle)
        
        # Draw ego annotations
        ego_annotations = ["Label: EGO"]
        if ego_interaction:
            actions_str = ', '.join(ego_interaction['actions']) if ego_interaction['actions'] else 'No actions'
            ego_annotations.append(f"Actions: {actions_str}")
            
            if show_velocity and 'velocity_magnitude' in ego_interaction['details']:
                vel_mag = ego_interaction['details']['velocity_magnitude']
                ego_annotations.append(f"Speed: {vel_mag:.2f} m/s")
            
            if show_details:
                details = ego_interaction['details']
                if 'heading_change' in details:
                    ego_annotations.append(f"Δθ: {details['heading_change']:.1f}°")
                if 'acceleration' in details:
                    ego_annotations.append(f"Accel: {details['acceleration']:.2f} m/s²")
                if 'displacement' in details:
                    ego_annotations.append(f"Disp: {details['displacement']:.2f}m")
                if 'lateral_displacement' in details:
                    lat = details['lateral_displacement']
                    if 'adjusted_lateral' in details:
                        adj_lat = details['adjusted_lateral']
                        ego_annotations.append(f"Lateral: {lat:.2f}m (adj: {adj_lat:.2f}m)")
                    else:
                        ego_annotations.append(f"Lateral: {lat:.2f}m")
                if 'cos_angle' in details:
                    ego_annotations.append(f"Cos: {details['cos_angle']:.2f}")
        
        ego_text_str = '\n'.join(ego_annotations)
        ax.text(
            ego_point[0], ego_point[1] - 20,
            ego_text_str,
            color='white',
            fontsize=10,
            weight='bold',
            verticalalignment='bottom',
            horizontalalignment='center',
            bbox=dict(
                boxstyle='round,pad=0.5',
                facecolor=ego_color,
                alpha=0.8,
                edgecolor='white',
                linewidth=2
            )
        )
        
        # Add legend
        legend_elements = []
        for action, color in action_colors.items():
            legend_elements.append(
                patches.Patch(facecolor=color, edgecolor='white', label=action)
            )
        ax.legend(
            handles=legend_elements,
            loc='upper right',
            fontsize=10,
            framealpha=0.9
        )
        
        # Title
        frame_token = frame['metadata'].get('frame_token', 'unknown')
        ax.set_title(
            f"Internal Actions ",
            fontsize=14,
            weight='bold'
        )
        ax.axis('off')
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=150)
            logger.info(f"Saved visualization to {save_path}")
        canvas = FigureCanvas(fig)
        canvas.draw()
        w, h = canvas.get_width_height()
        argb = np.frombuffer(canvas.tostring_argb(), dtype=np.uint8).reshape((h, w, 4))
        rgba = argb[:, :, [1, 2, 3, 0]]
        # drop alpha, RGBA -> RGB
        rgb = rgba[:, :, :3]
        # RGB -> BGR (OpenCV)
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        plt.close(fig)
        return bgr

    @staticmethod
    def extract_interactions(
        video_data: List[Dict[str, Any]], 
        velocity_threshold: float = 0.2,  # m/s - below this is considered stopped
        acceleration_threshold: float = 0.3,  # m/s^2
        turn_angle_threshold: float = 10.0,  # degrees
        u_turn_angle_threshold: float = 60.0,  # degrees
        lane_change_lateral_threshold: float = 2.0,  # meters
        lead_follow_distance_threshold: float = 30.0,  # meters
        same_direction_angle_threshold: float = 30.0,  # degrees
        moving_cos_angle_threshold: float = 60,
        distance_threshold: float = 20, # meters
        passing_distance_threshold: float = 5.0,  # meters - max distance for passing detection
        approaching_distance_threshold: float = 10.0,  # meters - max distance for approaching detection
        crossing_distance_threshold: float = 10.0,  # meters - max distance for crossing detection
    ) -> List[Dict[str, Any]]:
        """
        Extract interaction between objects in a scene.
        
        Args:
            video_data: List of frame data dictionaries
            velocity_threshold: Speed below which object is considered stopped (m/s)
            acceleration_threshold: Acceleration magnitude for accel/decel detection (m/s^2)
            turn_angle_threshold: Heading change for turn detection (degrees)
            u_turn_angle_threshold: Heading change for U-turn detection (degrees)
            lane_change_lateral_threshold: Lateral movement for lane change (meters)
            lead_follow_distance_threshold: Max distance for lead/follow relationship (meters)
            same_direction_angle_threshold: Max angle difference for same direction (degrees)
            distance_threshold: Max distance for interaction detection (meters)
            passing_distance_threshold: Max distance for passing detection (meters)
        
        Returns:
            video_data with added 'internal_interactions' and 'external_interactions' fields
            
        Interactions:
            Internal (individual object actions):
                - Moving forward/backward
                - Stopped
                - Left/Right turn
                - U Turn
                - Lane Change
                - Accelerate/Decelerate
            
            External (between objects):
                - Lead/Follow: Same lane, both moving, one ahead/behind (requires same direction)
                - Overtake: Source ahead after lane change/turn (NO direction requirement)
                - Passing: Different lanes, source ahead (requires similar direction)
                - Co-Moving: Different lanes, side by side, similar speeds (requires similar direction)
                - Approaching: Opposite directions, getting closer
                - Vehicle Crossing: Perpendicular paths
                - Waiting: Stopped near pedestrians
                - Yielding: Stopped, yielding to moving vehicle
        """
        if len(video_data) < 2:
            logger.warning("Video data has less than 2 frames, skipping interaction extraction")
            return video_data
        # check if source is ONCE return
        if video_data[0]['metadata']['source'] == DataSource.ONCE.value:
            return video_data
        if video_data[0]['metadata']['source'] == DataSource.WAYMO.value:
            velocity_threshold = 1
        # Build track history across frames
        track_history = InteractionExtractor._build_track_history(video_data)
        
        os.makedirs("visualization", exist_ok=True)
        video_path_internal = os.path.join("visualization", "internal", "debug_internal_interactions.mp4")
        video_path_external = os.path.join("visualization", "external", "debug_external_interactions.mp4")
        fps = 1
        target_size = (1920, 1080)  # (width, height)
        # video_writer_internal = cv2.VideoWriter(
        #     video_path_internal,
        #     cv2.VideoWriter_fourcc(*"mp4v"),
        #     fps,
        #     target_size,
        # )
        # video_writer_external = cv2.VideoWriter(
        #     video_path_external,
        #     cv2.VideoWriter_fourcc(*"mp4v"),
        #     fps,
        #     target_size,
        # )
        results = []
        for frame_idx, frame in enumerate(video_data):
            bbox_3d = frame['annotation']['bbox_3d']
            label_3d = frame['annotation']['label_3d']
            track_3d = frame['annotation']['track_3d']
            velocity = frame['annotation'].get('velocity', None)
            curr_time_stamp = frame['calibration']['img_timestamp'][-1]
            
            # Calculate ego yaw from ego_pose rotation matrix
            ego_pose = frame['calibration']['ego_pose']
            ego_yaw = float(np.arctan2(ego_pose[1, 0], ego_pose[0, 0]))
            
            # Calculate ego velocity from ego_pose changes
            ego_velocity = np.array([0.0, 0.0], dtype=np.float32)
            ego_history = track_history.get('00000', [])
            
            if len(ego_history) > 0 and velocity is not None:
                ego_velocity = ego_history[frame_idx]['velocity']
            
            # Add ego vehicle with ID 00000 at origin
            bbox_3d = [[0.0, 0.0, 0.0, 4.5, 2.0, 1.8, math.pi / 2]] + list(bbox_3d)
            label_3d = ['ego'] + list(label_3d)
            track_3d = ['00000'] + list(track_3d)
            if velocity is not None:
                velocity = np.vstack([ego_velocity, velocity])
            
            # Calculate velocity from position changes if not provided
            if velocity is None:
                ego_pose_inv = frame['calibration']['ego_pose_inv']
                calculated_velocities = []
                
                for i, track_id in enumerate(track_3d):
                    history = track_history.get(track_id, [])
                    history = [h for h in history if h['frame_idx'] <= frame_idx]
                    
                    if len(history) >= 2:
                        prev_data = history[-2]
                        curr_bbox = bbox_3d[i]
                        
                        # Transform previous bbox to current frame
                        prev_bbox_transformed = InteractionExtractor._transform_bbox_to_current_frame(
                            prev_bbox=prev_data['bbox_3d'],
                            prev_ego_pose=prev_data['ego_pose'],
                            curr_ego_pose_inv=ego_pose_inv
                        )
                        # Calculate velocity
                        vel = InteractionExtractor._calculate_velocity_from_history(
                            curr_bbox=curr_bbox,
                            prev_bbox_transformed=prev_bbox_transformed,
                            time_delta=(curr_time_stamp - prev_data['timestamp'])
                        )
                        calculated_velocities.append(vel)
                    else:
                        # No history, assume zero velocity
                        calculated_velocities.append(np.array([0.0, 0.0], dtype=np.float32))
                
                velocity = np.array(calculated_velocities, dtype=np.float32)
            
            # ================ Internal Interaction Extraction ================
            ego_pose_inv = frame['calibration']['ego_pose_inv']
            
            internal_interactions = InteractionExtractor._extract_internal_interactions(
                frame_idx=frame_idx,
                track_ids=track_3d,
                labels=label_3d,
                bboxes=bbox_3d,
                velocities=velocity,
                track_history=track_history,
                curr_ego_pose_inv=ego_pose_inv,
                velocity_threshold=velocity_threshold,
                acceleration_threshold=acceleration_threshold,
                turn_angle_threshold=turn_angle_threshold,
                u_turn_angle_threshold=u_turn_angle_threshold,
                lane_change_lateral_threshold=lane_change_lateral_threshold,
                moving_cos_angle_threshold=moving_cos_angle_threshold,
            )

            # # # Matplotlib -> OpenCV: get BGR uint8 frame, resize to writer resolution, then write.
            # vis_bgr = InteractionExtractor.visualize_internal_actions(
            #     frame,
            #     internal_interactions,
            #     view_idx="CAM_FRONT",
            #     save_path=f"visualization/internal/debug_internal_interactions_{frame_idx}.png",
            # )
            # if vis_bgr is not None:
            #     if (vis_bgr.shape[1], vis_bgr.shape[0]) != target_size:
            #         vis_bgr = cv2.resize(vis_bgr, dsize=target_size, interpolation=cv2.INTER_AREA)
            #     video_writer_internal.write(vis_bgr)
            
            # # ================ External Interaction Extraction ================
            external_interactions = InteractionExtractor._extract_external_interactions(
                frame_idx=frame_idx,
                track_ids=track_3d,
                labels=label_3d,
                bboxes=bbox_3d,
                velocities=velocity,
                track_history=track_history,
                curr_ego_pose_inv=ego_pose_inv,
                internal_interactions=internal_interactions,  # Pass internal interactions
                velocity_threshold=velocity_threshold,
                lead_follow_distance_threshold=lead_follow_distance_threshold,
                same_direction_angle_threshold=same_direction_angle_threshold,
                passing_distance_threshold=passing_distance_threshold,
                approaching_distance_threshold=approaching_distance_threshold,
                crossing_distance_threshold=crossing_distance_threshold,
                distance_threshold=distance_threshold,
            )
            # vis_bgr = InteractionExtractor.visualize_external_interactions(
            #     frame,
            #     external_interactions,
            #     view_idx="CAM_FRONT",
            #     save_path=f"visualization/external/debug_external_interactions_{frame_idx}.png",
            #     show_all_pairs=False,  # Show all pair metrics for debugging
            #     # target_track="59541a06def340299c23b7b3e5519d86",
            #     is_show_details=True,
            # )
            # if vis_bgr is not None:
            #     if (vis_bgr.shape[1], vis_bgr.shape[0]) != target_size:
            #         vis_bgr = cv2.resize(vis_bgr, dsize=target_size, interpolation=cv2.INTER_AREA)
            #     video_writer_external.write(vis_bgr)
            results.append({
                **frame,
                'internal_interactions': internal_interactions,
                'external_interactions': external_interactions
            })

        # video_writer_internal.release()
        # video_writer_external.release()
        # import ipdb; ipdb.set_trace()
        # from sources.data_collection.models.data.waymo import WaymoDataset
        # WaymoDataset.render_topdown_map(
        #     lidar_path=frame['lidar']['path'],
        #     ego_pose=None,
        #     bboxes_3d=frame['annotation']['bbox_3d'],
        #     labels_3d=frame['annotation']['label_3d'],
        #     out_path="test_waymo_bev.png"
        # )
        return results

    @staticmethod
    def visualize_external_interactions(
        frame: Dict[str, Any],
        external_interactions: List[Dict[str, Any]],
        view_idx: str = None,
        save_path: str = None,
        target_track: str = None,
        show_all_pairs: bool = True,  # New parameter to show all pair metrics
        distance_threshold: float = 15.0, # Distance threshold for showing pair info
        is_show_details: bool = True,
    ):
        """
        Visualize external interactions (relationships between objects) on 2D bounding boxes.
        
        Args:
            frame: Frame data containing views, annotations
            external_interactions: List of external interaction dictionaries
            view_idx: Camera view name (e.g., 'CAM_FRONT')
            save_path: Optional path to save the visualization
            target_track: Optional track ID to focus on (shows only its interactions)
            show_all_pairs: If True, shows metrics for all pairs even without interactions
            distance_threshold: Max distance to show pair information
        """
        view_names = list(frame['views'].keys())
        if view_idx is None:
            view_idx = view_names[-1]
        if view_idx not in view_names:
            logger.warning(f"View {view_idx} not found in frame. Available: {view_names}")
            return
        
        idx = view_names.index(view_idx)
        
        bbox_2d = frame['annotation']['bbox_2d'][idx]
        track_2d = frame['annotation']['track_2d'][idx]
        track_3d = frame['annotation']['track_3d']
        bbox_3d = frame['annotation']['bbox_3d']
        velocity = frame['annotation'].get('velocity', None)
        
        # Load image
        image = plt.imread(frame['views'][view_idx])
        
        # Ego position as a point at bottom center
        img_h, img_w = image.shape[:2]
        ego_point = (img_w / 2, img_h - 30)  # Bottom center point
        
        # Map 3D tracks to 2D bboxes
        track_3d_to_2d = {}
        for i, t3d in enumerate(track_3d):
            for j, t2d in enumerate(track_2d):
                if t3d == t2d:
                    track_3d_to_2d[t3d] = bbox_2d[j]
                    break
        
        # Create figure
        fig, ax = plt.subplots(1, figsize=(20, 14))
        ax.imshow(image)
        
        # Color scheme for interaction types
        interaction_colors = {
            ExtrinsicInteraction.LEAD: 'green',
            ExtrinsicInteraction.FOLLOW: 'blue',
            ExtrinsicInteraction.OVERTAKE: 'orange',
            ExtrinsicInteraction.PASSING: 'purple',
            ExtrinsicInteraction.CO_MOVING: 'lime',
            ExtrinsicInteraction.APPROACHING: 'red',
            ExtrinsicInteraction.CROSSING: 'cyan',
            ExtrinsicInteraction.WAITING: 'yellow',
            ExtrinsicInteraction.YIELDING: 'magenta',
            # Fallback for string values (backward compatibility)
            'lead': 'green',
            'follow': 'blue',
            'overtake': 'orange',
            'pass': 'purple',
            'co_moving': 'lime',
            'approaching': 'red',
            'crossing': 'cyan',
            'waiting': 'yellow',
            'yielding': 'magenta',
        }
        
        # Calculate bbox centers and draw bounding boxes for non-ego objects
        bbox_centers = {}
        # Add ego center
        bbox_centers['00000'] = ego_point
        
        for track_id, bbox in track_3d_to_2d.items():
            x1, y1, x2, y2 = bbox
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            bbox_centers[track_id] = (center_x, center_y)
            
            # Draw bounding box
            if track_id == target_track:
                color = 'red'
                linewidth = 3
            else:
                color = 'white'
                linewidth = 2
            rect = patches.Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                linewidth=linewidth,
                edgecolor=color,
                facecolor='none'
            )
            ax.add_patch(rect)
            
            # Label
            ax.text(
                x1, y1 - 5,
                f"{track_id[:8]}",
                color=color,
                fontsize=9,
                weight='bold'
            )
        
        # Draw ego vehicle as a circle
        ego_circle = patches.Circle(
            ego_point, 
            radius=15, 
            facecolor='green', 
            edgecolor='white', 
            linewidth=3,
            alpha=0.9
        )
        ax.add_patch(ego_circle)
        ax.text(
            ego_point[0], ego_point[1] - 20,
            'EGO',
            color='white',
            fontsize=10,
            weight='bold',
            horizontalalignment='center',
            verticalalignment='bottom',
            bbox=dict(
                boxstyle='round,pad=0.3',
                facecolor='green',
                alpha=0.8,
                edgecolor='white',
                linewidth=2
            )
        )
        
        # Calculate all pair relationships if requested
        all_pair_metrics = {}
        if show_all_pairs:
            # Calculate metrics for all pairs
            for i, track_i in enumerate(track_3d):
                if track_i not in track_3d_to_2d:
                    continue
                for j, track_j in enumerate(track_3d):
                    if i >= j or track_j not in track_3d_to_2d:  # Only calculate once per pair
                        continue
                    
                    # Get 3D positions
                    bbox_i = bbox_3d[i]
                    bbox_j = bbox_3d[j]
                    pos_i = np.array([float(bbox_i[0]), float(bbox_i[1])], dtype=float)
                    pos_j = np.array([float(bbox_j[0]), float(bbox_j[1])], dtype=float)
                    yaw_i = float(bbox_i[6])
                    yaw_j = float(bbox_j[6])
                    
                    # Calculate distance
                    distance = np.linalg.norm(pos_j - pos_i)
                    
                    if distance > distance_threshold:
                        continue
                    
                    # Calculate heading difference
                    heading_diff = abs(np.degrees(InteractionExtractor._angle_diff(yaw_i, yaw_j)))
                    
                    # Calculate lateral and forward distances
                    heading_vec_j = np.array([np.cos(yaw_j), np.sin(yaw_j)], dtype=float)
                    right_vec_j = np.array([np.sin(yaw_j), -np.cos(yaw_j)], dtype=float)
                    displacement_vec = pos_i - pos_j
                    forward_distance = float(np.dot(displacement_vec, heading_vec_j))
                    lateral_distance = float(abs(np.dot(displacement_vec, right_vec_j)))
                    
                    same_lane = lateral_distance < 2.0  # Use default threshold
                    same_direction = heading_diff < 30.0  # Use default threshold
                    similar_direction = heading_diff < 45.0  # Relaxed for overtake/passing
                    opposite_direction = heading_diff > 150.0  # Use default threshold
                    perpendicular = abs(heading_diff - 90.0) < 30.0 or abs(heading_diff - 270.0) < 30.0
                    
                    # Get speeds
                    speed_i = None
                    speed_j = None
                    if velocity is not None:
                        vel_i = velocity[i]
                        vel_j = velocity[j]
                        speed_i = np.linalg.norm(vel_i)
                        speed_j = np.linalg.norm(vel_j)
                    
                    # Determine why no interaction (for debugging)
                    reasons = []
                    if opposite_direction and distance >= distance_threshold:
                        reasons.append(f"Too far for approaching (>{distance_threshold:.0f}m)")
                    if perpendicular and distance >= distance_threshold:
                        reasons.append(f"Too far for crossing (>{distance_threshold:.0f}m)")
                    if same_direction and not same_lane:
                        reasons.append("Not same lane")
                    
                    all_pair_metrics[(track_i, track_j)] = {
                        'distance': distance,
                        'heading_diff': heading_diff,
                        'lateral_distance': lateral_distance,
                        'forward_distance': forward_distance,
                        'same_lane': same_lane,
                        'same_direction': same_direction,
                        'similar_direction': similar_direction,
                        'opposite_direction': opposite_direction,
                        'perpendicular': perpendicular,
                        'speed_i': speed_i,
                        'speed_j': speed_j,
                        'no_interaction_reasons': reasons,
                    }
        
        # Draw interaction arrows and metrics
        drawn_pairs = set()
        
        # First, draw interactions that were detected
        for interaction in external_interactions:
            src_track = interaction['source_object']['track_id']
            tgt_track = interaction['target_object']['track_id']
            
            # Filter by target track if specified
            if target_track and src_track != target_track:
                continue
            
            if src_track not in bbox_centers or tgt_track not in bbox_centers:
                continue
            
            drawn_pairs.add((src_track, tgt_track))
            drawn_pairs.add((tgt_track, src_track))
            
            x1, y1 = bbox_centers[src_track]
            x2, y2 = bbox_centers[tgt_track]
            
            interaction_type = interaction['interaction_type']
            color = interaction_colors.get(interaction_type, 'yellow')
            
            # Draw arrow from source to target
            ax.annotate(
                '',
                xy=(x2, y2),
                xytext=(x1, y1),
                arrowprops=dict(
                    arrowstyle='->',
                    color=color,
                    lw=2.5,
                    alpha=0.7
                )
            )
            
            # Annotation text at midpoint
            mid_x = (x1 + x2) / 2
            mid_y = (y1 + y2) / 2
            
            details = interaction['details']
            # Handle both enum and string interaction types
            interaction_display = interaction_type.value if hasattr(interaction_type, 'value') else str(interaction_type)
            text_lines = [interaction_display.upper().replace('_', ' ')]
            if is_show_details:
                # Show type-specific details
                if 'overtake_type' in details:
                    text_lines.append(f"Type: {details['overtake_type']}")
                if 'passing_type' in details:
                    text_lines.append(f"Type: {details['passing_type']}")
                if 'co_moving_type' in details:
                    text_lines.append(f"Type: {details['co_moving_type']}")
                
                text_lines.append(f"Dist: {details['distance']:.1f}m")
                text_lines.append(f"Lateral: {details['lateral_distance']:.2f}m")
                if 'forward_distance' in details:
                    fwd = details['forward_distance']
                    text_lines.append(f"Forward: {fwd:.2f}m {'(src ahead)' if fwd > 0 else '(src behind)'}")
                text_lines.append(f"SameLane: {details['same_lane']}")
                
                # Show lane changing info for overtake/passing
                if 'is_i_lane_changing' in details:
                    text_lines.append(f"SrcLaneChange: {details['is_i_lane_changing']}")
                if 'is_i_turning' in details and details['is_i_turning']:
                    text_lines.append(f"SrcTurning: {details['is_i_turning']}")
                if 'i_actions' in details and details['i_actions']:
                    actions_str = ', '.join([a.split('.')[-1] if '.' in a else a for a in details['i_actions'][:2]])  # Show first 2
                    text_lines.append(f"SrcActions: {actions_str}")
                if 'lateral_change' in details:
                    text_lines.append(f"ΔLateral: {details['lateral_change']:.2f}m")
                
                if 'speed_i' in details and 'speed_j' in details:
                    text_lines.append(f"Speeds: {details['speed_i']:.1f}, {details['speed_j']:.1f} m/s")
                if 'speed_difference' in details:
                    text_lines.append(f"ΔSpeed: {details['speed_difference']:.2f} m/s")
                if 'heading_difference' in details:
                    text_lines.append(f"ΔHeading: {details['heading_difference']:.1f}°")
                if 'distance_change' in details:
                    text_lines.append(f"ΔDist: {details['distance_change']:.2f}m")
                if 'approach_angle' in details:
                    text_lines.append(f"Approach∠: {details['approach_angle']:.1f}°")
                
            ax.text(
                mid_x, mid_y,
                '\n'.join(text_lines),
                color='white',
                fontsize=7,
                weight='bold',
                ha='center',
                va='center',
                bbox=dict(
                    boxstyle='round,pad=0.4',
                    facecolor=color,
                    alpha=0.9,
                    edgecolor='white',
                    linewidth=1.5
                )
            )
        
        # Draw metrics for pairs without interactions (in gray/debug mode)
        if show_all_pairs:
            for (track_i, track_j), metrics in all_pair_metrics.items():
                # Skip if already drawn as an interaction
                if (track_i, track_j) in drawn_pairs or (track_j, track_i) in drawn_pairs:
                    continue
                
                # Filter by target track if specified
                if target_track and track_i != target_track and track_j != target_track:
                    continue
                
                if track_i not in bbox_centers or track_j not in bbox_centers:
                    continue
                
                x1, y1 = bbox_centers[track_i]
                x2, y2 = bbox_centers[track_j]
                
                # Draw dashed line (not arrow) for non-interactions
                ax.plot([x1, x2], [y1, y2], 'gray', linestyle='--', linewidth=1, alpha=0.4)
                
                # Annotation text at midpoint (smaller, gray)
                mid_x = (x1 + x2) / 2
                mid_y = (y1 + y2) / 2
                
                text_lines = ["NO INTERACTION"]
                text_lines.append(f"Dist: {metrics['distance']:.1f}m")
                text_lines.append(f"Lateral: {metrics['lateral_distance']:.2f}m")
                if 'forward_distance' in metrics:
                    fwd = metrics['forward_distance']
                    text_lines.append(f"Forward: {fwd:.2f}m")
                text_lines.append(f"SameLane: {metrics['same_lane']}")
                text_lines.append(f"SameDir: {metrics['same_direction']}")
                if metrics.get('similar_direction') and not metrics['same_direction']:
                    text_lines.append(f"SimilarDir: {metrics['similar_direction']}")
                if metrics.get('opposite_direction'):
                    text_lines.append("OppositeDir: True")
                if metrics.get('perpendicular'):
                    text_lines.append("Perpendicular: True")
                if metrics['speed_i'] is not None and metrics['speed_j'] is not None:
                    text_lines.append(f"Speeds: {metrics['speed_i']:.1f}, {metrics['speed_j']:.1f}")
                text_lines.append(f"ΔHeading: {metrics['heading_diff']:.1f}°")
                # Show reasons why no interaction
                if metrics.get('no_interaction_reasons'):
                    text_lines.append("---")
                    for reason in metrics['no_interaction_reasons']:
                        text_lines.append(reason)
                
                ax.text(
                    mid_x, mid_y,
                    '\n'.join(text_lines),
                    color='black',
                    fontsize=6,
                    weight='normal',
                    ha='center',
                    va='center',
                    bbox=dict(
                        boxstyle='round,pad=0.3',
                        facecolor='lightgray',
                        alpha=0.6,
                        edgecolor='gray',
                        linewidth=0.5
                    )
                )
        
        # Add legend (only show enum values, not string duplicates)
        legend_elements = []
        added_types = set()
        for itype, color in interaction_colors.items():
            # Only add enum types (not string duplicates)
            if isinstance(itype, ExtrinsicInteraction):
                label = itype.value.replace('_', ' ').title()
                if color not in added_types:
                    legend_elements.append(
                        patches.Patch(facecolor=color, edgecolor='white', label=label)
                    )
                    added_types.add(color)
        ax.legend(
            handles=legend_elements,
            loc='upper right',
            fontsize=12,
            framealpha=0.9
        )
        
        # Title
        frame_token = frame['metadata'].get('frame_token', 'unknown')
        title = f"External Interactions - {view_idx}\nFrame: {frame_token[:16]}..."
        if target_track:
            title += f"\nFocusing on: {target_track[:8]}"
        ax.set_title(title, fontsize=14, weight='bold')
        ax.axis('off')
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=150)
            logger.info(f"Saved visualization to {save_path}")
        
        # convert that plt figure to opencv image
        canvas = FigureCanvas(fig)
        canvas.draw()
        w, h = canvas.get_width_height()
        argb = np.frombuffer(canvas.tostring_argb(), dtype=np.uint8).reshape((h, w, 4))
        rgba = argb[:, :, [1, 2, 3, 0]]
        # drop alpha, RGBA -> RGB
        rgb = rgba[:, :, :3]
        # RGB -> BGR (OpenCV)
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        plt.close(fig)
        return bgr
        
