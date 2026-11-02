import os
import json
import random
import logging
import math
import numpy as np
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional, Callable
from uuid import uuid4
from enum import Enum

from sources.question_generation.enum import RelationQuestionType
from sources.question_generation.utils import visualize_multiview
from sources.db.sql import SQLiteController
from sources.data_collection.enum.data_sources import DataSource
from sources.data_collection.enum.interaction_types import Relation, ExtrinsicInteraction, IntrinsicInteraction

logger = logging.getLogger(__name__)


class RotationAngle(Enum):
    """Rotation angles for ego actions."""
    U_TURN = 180
    TURN_LEFT = 90
    TURN_RIGHT = -90
    NO_ROTATION = 0


@dataclass
class EgoState:
    """Represents the current state of the ego vehicle."""
    position: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.0]))
    orientation: float = 0.0  # Degrees, 0 = forward (positive x)
    velocity: float = 0.0  # m/s
    
    def copy(self) -> 'EgoState':
        """Create a deep copy of the state."""
        return EgoState(
            position=self.position.copy(),
            orientation=self.orientation,
            velocity=self.velocity
        )


@dataclass
class ChainStep:
    """A single step in the movement chain."""
    action: str
    rotation_delta: float  # Degrees
    movement_delta: np.ndarray  # [dx, dy, dz]
    description: str


class InteractionOrientation(Enum):
    """
    Defines how interactions affect ego's final orientation relative to target.
    
    SAME_DIRECTION: After the interaction, ego faces the same direction as target.
        - LEAD, FOLLOW, PASSING, CO_MOVING, OVERTAKE
        - These interactions imply ego aligns with target's movement direction
    
    OPPOSITE_DIRECTION: After the interaction, ego faces opposite direction to target.
        - CROSSING, APPROACHING
        - These interactions imply ego and target are moving towards each other
    
    NEUTRAL: Interaction doesn't change relative orientation.
        - WAITING, YIELDING
        - Ego maintains its current orientation relative to target
    """
    SAME_DIRECTION = "same_direction"
    OPPOSITE_DIRECTION = "opposite_direction"
    NEUTRAL = "neutral"


class EgoMovementChain:
    """
    Chain-based approach for tracking ego movement and rotation.
    
    Instead of if-else logic, builds a chain of operations that:
    1. Start from current ego position/orientation
    2. Apply rotation based on action type (u-turn: 180°, turn left: +90°, turn right: -90°)
    3. Apply interaction-based orientation adjustment:
       - LEAD/FOLLOW/PASSING/CO_MOVING/OVERTAKE → align with target direction
       - CROSSING/APPROACHING → opposite to target direction
    4. Calculate new position after movement
    5. Track cumulative state changes
    """
    
    # Rotation mappings for actions
    ACTION_ROTATIONS: Dict[str, float] = {
        IntrinsicInteraction.U_TURN.value: RotationAngle.U_TURN.value,
        IntrinsicInteraction.TURN_LEFT.value: RotationAngle.TURN_LEFT.value,
        IntrinsicInteraction.TURN_RIGHT.value: RotationAngle.TURN_RIGHT.value,
        IntrinsicInteraction.MOVING_FORWARD.value: RotationAngle.NO_ROTATION.value,
        IntrinsicInteraction.MOVING_BACKWARD.value: RotationAngle.NO_ROTATION.value,
        IntrinsicInteraction.LANE_CHANGE_LEFT.value: RotationAngle.NO_ROTATION.value,
        IntrinsicInteraction.LANE_CHANGE_RIGHT.value: RotationAngle.NO_ROTATION.value,
        IntrinsicInteraction.ACCELERATE.value: RotationAngle.NO_ROTATION.value,
        IntrinsicInteraction.DECELERATE.value: RotationAngle.NO_ROTATION.value,
        IntrinsicInteraction.STOPPED.value: RotationAngle.NO_ROTATION.value,
    }
    
    # Interaction to orientation mapping
    # After performing these interactions, ego will be in this orientation relative to target
    INTERACTION_ORIENTATIONS: Dict[str, InteractionOrientation] = {
        # Same direction interactions - ego aligns with target's direction
        ExtrinsicInteraction.LEAD.value: InteractionOrientation.SAME_DIRECTION,
        ExtrinsicInteraction.FOLLOW.value: InteractionOrientation.SAME_DIRECTION,
        ExtrinsicInteraction.PASSING.value: InteractionOrientation.SAME_DIRECTION,
        ExtrinsicInteraction.CO_MOVING.value: InteractionOrientation.SAME_DIRECTION,
        ExtrinsicInteraction.OVERTAKE.value: InteractionOrientation.SAME_DIRECTION,
        # Opposite direction interactions - ego faces opposite to target
        ExtrinsicInteraction.CROSSING.value: InteractionOrientation.OPPOSITE_DIRECTION,
        ExtrinsicInteraction.APPROACHING.value: InteractionOrientation.OPPOSITE_DIRECTION,
        # Neutral interactions - no orientation change
        ExtrinsicInteraction.WAITING.value: InteractionOrientation.NEUTRAL,
        ExtrinsicInteraction.YIELDING.value: InteractionOrientation.NEUTRAL,
    }
    
    def __init__(self, initial_state: Optional[EgoState] = None):
        """
        Initialize the movement chain.
        
        Args:
            initial_state: Starting ego state. Defaults to origin facing forward.
        """
        self.initial_state = initial_state or EgoState()
        self.current_state = self.initial_state.copy()
        self.steps: List[ChainStep] = []
        self._total_rotation = 0.0
        
    def reset(self) -> 'EgoMovementChain':
        """Reset chain to initial state."""
        self.current_state = self.initial_state.copy()
        self.steps = []
        self._total_rotation = 0.0
        return self
    
    def apply_rotation(self, action: str) -> 'EgoMovementChain':
        """
        Apply rotation based on action type.
        
        Args:
            action: The ego action (e.g., 'u_turn', 'turn_left', 'turn_right')
            
        Returns:
            Self for chaining
        """
        action_lower = action.lower().replace(' ', '_').replace('-', '_')
        rotation = self._get_rotation_for_action(action_lower)
        
        self.current_state.orientation = (self.current_state.orientation + rotation) % 360
        self._total_rotation = (self._total_rotation + rotation) % 360
        
        step = ChainStep(
            action=action,
            rotation_delta=rotation,
            movement_delta=np.array([0.0, 0.0, 0.0]),
            description=f"Rotate {rotation}° ({action})"
        )
        self.steps.append(step)
        
        logger.debug(f"Applied rotation: {rotation}° for action '{action}', new orientation: {self.current_state.orientation}°")
        return self
    
    def _get_rotation_for_action(self, action: str) -> float:
        """
        Get rotation angle for a given action.
        
        Args:
            action: Normalized action string
            
        Returns:
            Rotation angle in degrees
        """
        # Direct lookup
        if action in self.ACTION_ROTATIONS:
            return self.ACTION_ROTATIONS[action]
        
        # Pattern matching for variations
        if 'u_turn' in action or 'u turn' in action:
            return RotationAngle.U_TURN.value
        elif 'turn_left' in action or 'turn left' in action:
            return RotationAngle.TURN_LEFT.value
        elif 'turn_right' in action or 'turn right' in action:
            return RotationAngle.TURN_RIGHT.value
        
        return RotationAngle.NO_ROTATION.value
    
    def apply_movement_to_target(
        self,
        target_position: np.ndarray,
        movement_fraction: float = 1.0
    ) -> 'EgoMovementChain':
        """
        Move ego towards a target position.
        
        Args:
            target_position: Target position [x, y, z]
            movement_fraction: Fraction of distance to move (0-1)
            
        Returns:
            Self for chaining
        """
        direction = target_position - self.current_state.position
        distance = np.linalg.norm(direction[:2])
        
        if distance > 0.01:
            movement = direction * movement_fraction
            self.current_state.position = self.current_state.position + movement
            
            step = ChainStep(
                action="move_to_target",
                rotation_delta=0.0,
                movement_delta=movement,
                description=f"Move {distance * movement_fraction:.2f}m towards target"
            )
            self.steps.append(step)
            
            logger.debug(f"Moved to position: {self.current_state.position}")
        
        return self
    
    def apply_interaction_orientation(
        self,
        interaction: str,
        initial_same_direction: bool
    ) -> 'EgoMovementChain':
        """
        Apply orientation adjustment based on interaction type.
        
        This is the KEY logic for determining final ego orientation:
        
        1. SAME_DIRECTION interactions (LEAD, FOLLOW, PASSING, CO_MOVING, OVERTAKE):
           - After turn + interaction, ego aligns with target's direction
           - Final orientation relative to target = 'same' (0°)
        
        2. OPPOSITE_DIRECTION interactions (CROSSING, APPROACHING):
           - After turn + interaction, ego faces opposite to target
           - Final orientation relative to target = 'opposite' (180°)
        
        3. NEUTRAL interactions (WAITING, YIELDING):
           - Interaction doesn't change orientation
           - Final orientation = result of action rotation only
        
        Args:
            interaction: The interaction type
            initial_same_direction: Whether ego initially faces same direction as target
            
        Returns:
            Self for chaining
        """
        interaction_lower = interaction.lower().replace(' ', '_').replace('-', '_')
        
        # Get interaction orientation type
        interaction_orientation = self._get_interaction_orientation(interaction_lower)
        
        if interaction_orientation == InteractionOrientation.SAME_DIRECTION:
            # After this interaction, ego should face same direction as target
            # Calculate required rotation to achieve this
            # Target is at 0° (our reference), we need to end up at 0°
            current_orientation = self.current_state.orientation % 360
            required_rotation = (0 - current_orientation) % 360
            if required_rotation > 180:
                required_rotation -= 360  # Take shorter path
            
            self.current_state.orientation = 0.0
            self._total_rotation = 0.0  # Reset to same direction
            
            step = ChainStep(
                action=f"align_with_target ({interaction})",
                rotation_delta=required_rotation,
                movement_delta=np.array([0.0, 0.0, 0.0]),
                description=f"Align with target direction via {interaction} (same direction)"
            )
            self.steps.append(step)
            
            logger.debug(f"Applied same-direction interaction '{interaction}': final orientation = same (0°)")
            
        elif interaction_orientation == InteractionOrientation.OPPOSITE_DIRECTION:
            # After this interaction, ego should face opposite to target
            # Calculate required rotation to achieve 180°
            current_orientation = self.current_state.orientation % 360
            required_rotation = (180 - current_orientation) % 360
            if required_rotation > 180:
                required_rotation -= 360  # Take shorter path
            
            self.current_state.orientation = 180.0
            self._total_rotation = 180.0  # Set to opposite direction
            
            step = ChainStep(
                action=f"face_opposite ({interaction})",
                rotation_delta=required_rotation,
                movement_delta=np.array([0.0, 0.0, 0.0]),
                description=f"Face opposite to target via {interaction} (opposite direction)"
            )
            self.steps.append(step)
            
            logger.debug(f"Applied opposite-direction interaction '{interaction}': final orientation = opposite (180°)")
            
        else:
            # NEUTRAL - interaction doesn't change orientation, keep current rotation
            step = ChainStep(
                action=f"neutral_interaction ({interaction})",
                rotation_delta=0.0,
                movement_delta=np.array([0.0, 0.0, 0.0]),
                description=f"Neutral interaction {interaction} - orientation unchanged"
            )
            self.steps.append(step)
            
            logger.debug(f"Applied neutral interaction '{interaction}': orientation unchanged at {self._total_rotation}°")
        
        return self
    
    def _get_interaction_orientation(self, interaction: str) -> InteractionOrientation:
        """
        Get the orientation type for a given interaction.
        
        Args:
            interaction: Normalized interaction string
            
        Returns:
            InteractionOrientation enum value
        """
        # Direct lookup
        if interaction in self.INTERACTION_ORIENTATIONS:
            return self.INTERACTION_ORIENTATIONS[interaction]
        
        # Pattern matching for variations
        same_direction_patterns = ['lead', 'follow', 'passing', 'co_moving', 'comoving', 'overtake']
        opposite_direction_patterns = ['crossing', 'approaching']
        
        for pattern in same_direction_patterns:
            if pattern in interaction:
                return InteractionOrientation.SAME_DIRECTION
        
        for pattern in opposite_direction_patterns:
            if pattern in interaction:
                return InteractionOrientation.OPPOSITE_DIRECTION
        
        return InteractionOrientation.NEUTRAL
    
    def apply_approaching_or_crossing(
        self,
        target_position: np.ndarray,
        ego_direction: np.ndarray,
        target_direction: np.ndarray,
        time_delta: float = 0.5
    ) -> 'EgoMovementChain':
        """
        Calculate new ego position for approaching/crossing when objects are in different directions.
        
        This operation only applies when:
        - The two objects are moving in different directions (angle > 45°)
        
        Args:
            target_position: Target object position [x, y, z]
            ego_direction: Ego's movement direction unit vector [dx, dy]
            target_direction: Target's movement direction unit vector [dx, dy]
            time_delta: Time to simulate movement (seconds)
            
        Returns:
            Self for chaining
        """
        # Normalize directions
        ego_dir_norm = ego_direction / (np.linalg.norm(ego_direction) + 1e-6)
        target_dir_norm = target_direction / (np.linalg.norm(target_direction) + 1e-6)
        
        # Calculate angle between directions
        dot_product = np.clip(np.dot(ego_dir_norm, target_dir_norm), -1.0, 1.0)
        angle_between = math.degrees(math.acos(dot_product))
        
        # Only apply if objects are moving in different directions (> 45°)
        if angle_between > 45:
            # Calculate movement towards target
            direction_to_target = target_position[:2] - self.current_state.position[:2]
            distance_to_target = np.linalg.norm(direction_to_target)
            
            if distance_to_target > 0.01:
                # Move based on velocity and time
                movement_distance = min(self.current_state.velocity * time_delta, distance_to_target)
                movement_unit = direction_to_target / distance_to_target
                movement = np.array([
                    movement_unit[0] * movement_distance,
                    movement_unit[1] * movement_distance,
                    0.0
                ])
                
                self.current_state.position = self.current_state.position + movement
                
                step = ChainStep(
                    action="approaching_crossing",
                    rotation_delta=0.0,
                    movement_delta=movement,
                    description=f"Move {movement_distance:.2f}m towards target (angle: {angle_between:.1f}°)"
                )
                self.steps.append(step)
                
                logger.debug(f"Applied approaching/crossing movement: {movement}, angle between: {angle_between:.1f}°")
        else:
            logger.debug(f"Skipped approaching/crossing: objects in similar direction (angle: {angle_between:.1f}°)")
        
        return self
    
    def get_total_rotation(self) -> float:
        """Get the total accumulated rotation in degrees."""
        return self._total_rotation
    
    def get_final_orientation_category(self) -> str:
        """
        Map total rotation to orientation category.
        
        Returns:
            'same' (0°), 'left' (90°), 'opposite' (180°), or 'right' (270°)
        """
        # Normalize rotation to 0-360
        rotation = self._total_rotation % 360
        
        # Map to nearest category
        if rotation < 45 or rotation >= 315:
            return 'same'
        elif 45 <= rotation < 135:
            return 'left'
        elif 135 <= rotation < 225:
            return 'opposite'
        else:  # 225 <= rotation < 315
            return 'right'
    
    def get_final_state(self) -> EgoState:
        """Get the final ego state after all chain operations."""
        return self.current_state.copy()
    
    def get_chain_summary(self) -> Dict[str, Any]:
        """Get a summary of the chain operations."""
        orientation_category = self.get_final_orientation_category()
        
        # Explain what the orientation means
        orientation_explanations = {
            'same': 'Ego faces same direction as target (relations unchanged)',
            'opposite': 'Ego faces opposite to target (front↔rear, left↔right swap)',
            'left': 'Ego turned left relative to target (90° CCW rotation of relations)',
            'right': 'Ego turned right relative to target (90° CW rotation of relations)'
        }
        
        return {
            'initial_position': self.initial_state.position.tolist(),
            'final_position': self.current_state.position.tolist(),
            'initial_orientation': self.initial_state.orientation,
            'final_orientation': self.current_state.orientation,
            'total_rotation': self._total_rotation,
            'orientation_category': orientation_category,
            'orientation_explanation': orientation_explanations.get(orientation_category, ''),
            'num_steps': len(self.steps),
            'steps': [
                {
                    'action': step.action,
                    'rotation_delta': step.rotation_delta,
                    'movement_delta': step.movement_delta.tolist(),
                    'description': step.description
                }
                for step in self.steps
            ]
        }


class RelationTransformer:
    """
    Transforms spatial relations based on ego orientation changes.
    
    Uses rotation math to transform relations:
    - 0° (same): No change
    - 90° (left): Rotate relations counter-clockwise
    - 180° (opposite): Flip front↔rear, left↔right
    - 270° (right): Rotate relations clockwise
    """
    
    # Transformation maps for each orientation
    TRANSFORM_MAPS: Dict[str, Dict[str, str]] = {
        'same': {},  # No transformation
        'opposite': {
            'front': 'rear',
            'rear': 'front',
            'ahead': 'behind',
            'behind': 'ahead',
            'left': 'right',
            'right': 'left'
        },
        'left': {
            'front': 'right',
            'ahead': 'right',
            'right': 'rear',
            'rear': 'left',
            'behind': 'left',
            'left': 'front'
        },
        'right': {
            'front': 'left',
            'ahead': 'left',
            'left': 'rear',
            'rear': 'right',
            'behind': 'right',
            'right': 'front'
        }
    }
    
    @classmethod
    def transform(cls, relation: str, orientation: str) -> str:
        """
        Transform a relation based on ego's final orientation.
        
        Args:
            relation: Original relation string (e.g., 'rear_left_of')
            orientation: Ego's final orientation ('same', 'opposite', 'left', 'right')
            
        Returns:
            Transformed relation string
        """
        if orientation == 'same':
            return relation
        
        transform_map = cls.TRANSFORM_MAPS.get(orientation, {})
        if not transform_map:
            return relation
        
        # Split relation into parts to transform each component separately
        parts = relation.lower().replace('_', ' ').split()
        
        # Transform each part
        transformed_parts = [transform_map.get(part, part) for part in parts]
        
        # Rejoin with underscores to maintain original format
        return '_'.join(transformed_parts)
    
    @classmethod
    def transform_from_rotation(cls, relation: str, rotation_degrees: float) -> str:
        """
        Transform a relation based on rotation angle.
        
        Args:
            relation: Original relation string
            rotation_degrees: Total rotation in degrees
            
        Returns:
            Transformed relation string
        """
        # Normalize rotation to 0-360
        rotation = rotation_degrees % 360
        
        # Map to orientation category
        if rotation < 45 or rotation >= 315:
            orientation = 'same'
        elif 45 <= rotation < 135:
            orientation = 'left'
        elif 135 <= rotation < 225:
            orientation = 'opposite'
        else:
            orientation = 'right'
        
        return cls.transform(relation, orientation)


class MultiStepReasoningGenerator:
    QUESTION_TYPE = RelationQuestionType.MULTI_STEPS_REASONING
    QUESTION_TEMPLATE = """
What if the ego vehicle is {ego_action} and {interaction} with object #{target_id}.
What is the object {relation} of yourself?
"""
    
    # Distance threshold for near object preference (in meters)
    NEAR_OBJECT_THRESHOLD = 20.0
    # Probability to select from near objects (80%)
    NEAR_OBJECT_PROBABILITY = 0.8
    
    # Map camera keys to dataset-specific names
    CAMERA_ORDER = {
        'nuscenes': ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
                     'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT'],
        'once': ['cam01', 'cam03', 'cam05', 'cam06', 'cam07', 'cam08', 'cam09'],
        'waymo': ['FRONT_LEFT', 'FRONT', 'FRONT_RIGHT', 'SIDE_LEFT', 'SIDE_RIGHT'],
        'truckscenes': ['CAMERA_LEFT_FRONT', 'CAMERA_RIGHT_FRONT',
                        'CAMERA_LEFT_BACK', 'CAMERA_RIGHT_BACK'],
        'argoverse2': ['ring_front_left', 'ring_front_center', 'ring_front_right',
                       'ring_side_left', 'ring_side_right',
                       'ring_rear_left', 'ring_rear_right']
    }
    
    @staticmethod
    def generate_question(
        sampled_data: Dict[str, Any],
        db_path: str,
        save_path: str,
        num_options: int = 4
    ) -> Optional[Dict[str, Any]]:
        """
        Generate a multi-step reasoning question.
        
        Args:
            sampled_data: Sampled data record from database
            db_path: Path to the database
            save_path: Path to save questions and images
            num_options: Number of options to generate (default: 4)
            
        Returns:
            question: Question dictionary or None if generation fails
        """
        try:
            # Extract metadata
            metadata = sampled_data.get("metadata", {})
            source = metadata.get("source")
            scene_id = sampled_data.get("scene_id")
            frame_id = sampled_data.get("frame_id")
            
            # Extract interaction and spatial data
            external_interactions = sampled_data.get("external_interactions", [])
            internal_interactions = sampled_data.get("internal_interactions", [])
            spatial_relations = sampled_data.get("spatial_relations", {})
            annotation = sampled_data.get("annotation", {})
            
            # Debug: Print data structure
            logger.info(f"Debug - External interactions: {len(external_interactions)}")
            logger.info(f"Debug - Internal interactions: {len(internal_interactions)}")
            logger.info(f"Debug - Spatial relations: {len(spatial_relations)}")
            
            if external_interactions and len(external_interactions) > 0:
                logger.info(f"Debug - Sample external interaction: {external_interactions[0]}")
            
            # Step 1: Find a valid reasoning chain using the chain-based approach
            reasoning_chain = MultiStepReasoningGenerator._find_reasoning_chain(
                external_interactions,
                spatial_relations,
                annotation,
                internal_interactions
            )
            
            if not reasoning_chain:
                logger.warning("No valid reasoning chain found")
                return None
            
            # Create save directories
            question_folder = Path(save_path) / "relation" / MultiStepReasoningGenerator.QUESTION_TYPE.value
            images_folder = question_folder / "images"
            images_folder.mkdir(parents=True, exist_ok=True)
            
            # Generate unique ID for this question
            question_id = str(uuid4())
            
            # Step 2: Generate multiple choice options FIRST (need them for visualization)
            correct_answer = reasoning_chain['answer_id']
            options, correct_answer_idx = MultiStepReasoningGenerator._generate_options(
                correct_answer,
                annotation,
                reasoning_chain,
                num_options
            )
            
            # Step 3: Visualize multi-view with target and ALL option objects (not just answer)
            # Build highlight list: target + all option objects
            highlight_objects = [
                {
                    'object_id': reasoning_chain['target_id'],
                    'color': (255, 0, 0),  # Blue in BGR
                    'label': '(target)'
                }
            ]
            
            # Add all option objects without revealing which is the answer
            option_colors = [
                (0, 255, 0),    # Green
                (255, 0, 255),  # Magenta
                (0, 255, 255),  # Cyan
                (255, 255, 0),  # Yellow
                (128, 0, 255),  # Purple
                (255, 128, 0),  # Orange
            ]
            
            for idx, (obj_id, obj_class) in enumerate(options):
                # Skip if this object is already the target
                if obj_id == reasoning_chain['target_id']:
                    continue
                
                color = option_colors[idx % len(option_colors)]
                highlight_objects.append({
                    'object_id': obj_id,
                    'color': color,
                    'label': ''  # No label - don't reveal anything
                })
            
            multiview_path = images_folder / f"{question_id}_multiview.jpg"
            visualize_multiview(
                data=sampled_data,
                output_path=str(multiview_path),
                source=source,
                show_object_ids=True,
                highlight_objects=highlight_objects,
                show_only_highlighted=True  # Only show target and option objects
            )
            
            # Step 4: Generate question text
            question_text = MultiStepReasoningGenerator.QUESTION_TEMPLATE.format(
                ego_action=reasoning_chain['ego_action'],
                interaction=reasoning_chain['interaction'],
                target_id=reasoning_chain['target_id'],
                relation=reasoning_chain['relation']
            )
            
            question_text += f"\n\nMulti-view visualization: <image>"
            question_text += f"\n\nOptions:\n"
            for idx, (obj_id, obj_class) in enumerate(options):
                question_text += f"({chr(65+idx)}) Object #{obj_id} ({obj_class})\n"
            
            # Step 5: Create question dictionary
            question_dict = {
                "question_id": question_id,
                "question_type": MultiStepReasoningGenerator.QUESTION_TYPE.value,
                "question": question_text,
                "images": [str(multiview_path)],
                "options": [f"Object #{obj_id} ({obj_class})" for obj_id, obj_class in options],
                "correct_answer": correct_answer_idx,
                "metadata": {
                    "scene_id": scene_id,
                    "frame_id": frame_id,
                    "source": source,
                    "record_id": sampled_data.get("id"),
                    "reasoning_chain": reasoning_chain
                }
            }
            
            return question_dict
            
        except Exception as e:
            logger.error(f"Error generating multi-step reasoning question: {e}", exc_info=True)
            return None

    @staticmethod
    def _get_object_distance_from_ego(
        track_id: str,
        annotation: Dict[str, Any]
    ) -> float:
        """
        Calculate distance from ego (0,0,0) to object center.
        
        Args:
            track_id: Object track ID
            annotation: Annotation data with track_3d and bbox_3d
            
        Returns:
            Distance in meters, or float('inf') if not found
        """
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
        
        # bbox_3d format: [x, y, z, ...]
        x, y, z = bbox[0], bbox[1], bbox[2]
        return math.sqrt(x**2 + y**2 + z**2)
    
    @staticmethod
    def _weighted_choice(
        items: List[Any],
        annotation: Dict[str, Any],
        track_id_getter: Callable[[Any], str],
        near_threshold: float = 20.0,
        near_probability: float = 0.8
    ) -> Optional[Any]:
        """
        Select an item with 80% probability from near objects (< threshold from ego).
        
        Args:
            items: List of items to choose from
            annotation: Annotation data
            track_id_getter: Function to extract track_id from item
            near_threshold: Distance threshold for near objects (default: 20m)
            near_probability: Probability to select from near objects (default: 0.8)
            
        Returns:
            Selected item or None if empty
        """
        if not items:
            return None
        
        if len(items) == 1:
            return items[0]
        
        # Separate items into near and far groups
        near_items = []
        far_items = []
        for item in items:
            track_id = track_id_getter(item)
            distance = MultiStepReasoningGenerator._get_object_distance_from_ego(track_id, annotation)
            if distance < near_threshold:
                near_items.append(item)
            else:
                far_items.append(item)
        
        # Select with probability-based approach
        pick_near = random.random() < near_probability
        
        if pick_near and near_items:
            return random.choice(near_items)
        elif far_items:
            return random.choice(far_items)
        elif near_items:
            return random.choice(near_items)
        else:
            return random.choice(items)
    
    @staticmethod
    def _is_action_plausible_for_object(
        ego_action: str,
        interaction: str,
        object_position: Optional[str],
        object_same_direction: bool
    ) -> bool:
        """
        Check if the ego action/interaction is plausible for a given object.
        
        Key rules:
        - Cannot "decelerate/wait/stop" for an object ahead moving in SAME direction (we'd just follow it)
        - Cannot "follow" an object behind us
        - Cannot "overtake" an object behind us
        - Can "decelerate/wait" for objects coming towards us (opposite direction) or perpendicular
        
        Args:
            ego_action: The ego action (e.g., "decelerate", "moving forward")
            interaction: The interaction type (e.g., "waiting", "follow")
            object_position: Position of object relative to ego (ahead, behind, left, right)
            object_same_direction: Whether object is moving in same direction as ego
            
        Returns:
            True if the action is plausible for this object
        """
        action_lower = ego_action.lower()
        interaction_lower = interaction.lower()
        
        # Waiting/decelerate/stop actions
        waiting_actions = ['decelerate', 'stopped', 'stop', 'wait']
        is_waiting_action = any(w in action_lower for w in waiting_actions) or 'waiting' in interaction_lower
        
        # Following actions
        following_actions = ['follow', 'co moving', 'co-moving']
        is_following_action = any(f in interaction_lower for f in following_actions)
        
        # Overtake actions
        overtake_actions = ['overtake', 'passing']
        is_overtake_action = any(o in interaction_lower for o in overtake_actions)
        
        if object_position == 'ahead':
            # Object is ahead
            if is_waiting_action and object_same_direction:
                # Can't wait/decelerate for a car ahead moving SAME direction - we'd just follow
                return False
            # If object is ahead and moving opposite direction (coming at us), waiting makes sense
            
        elif object_position == 'behind':
            # Object is behind
            if is_following_action:
                # Can't follow an object behind us
                return False
            if is_overtake_action:
                # Can't overtake an object behind us
                return False
        
        return True
    
    @staticmethod
    def _build_movement_chain(
        ego_action: str,
        interaction: str,
        target_position: np.ndarray,
        ego_direction: np.ndarray,
        target_direction: np.ndarray,
        same_direction: bool,
        ego_velocity: float = 10.0
    ) -> EgoMovementChain:
        """
        Build a movement chain based on ego action and interaction.
        
        This replaces the if-else logic with a chain-based approach:
        
        Chain execution order:
        1. Apply rotation for turn actions (U-turn: 180°, left: +90°, right: -90°)
        2. Apply interaction-based orientation adjustment:
           - LEAD/FOLLOW/PASSING/CO_MOVING/OVERTAKE → align with target (same direction)
           - CROSSING/APPROACHING → face opposite to target
        3. Apply movement for approaching/crossing (only if different directions)
        
        The key insight is that the INTERACTION determines the final orientation:
        - After "turn left" + "follow": ego turns left then aligns to follow target (same direction)
        - After "turn left" + "approaching": ego turns left then faces opposite to target
        
        Args:
            ego_action: The ego action
            interaction: The interaction type
            target_position: Target object position
            ego_direction: Ego's current direction vector
            target_direction: Target's direction vector
            same_direction: Whether initially facing same direction
            ego_velocity: Ego's velocity in m/s
            
        Returns:
            EgoMovementChain with all operations applied
        """
        # Initialize chain with ego at origin
        # Initial orientation is based on whether ego initially faces same direction as target
        initial_state = EgoState(
            position=np.array([0.0, 0.0, 0.0]),
            orientation=0.0 if same_direction else 180.0,  # 0° = same as target, 180° = opposite
            velocity=ego_velocity
        )
        chain = EgoMovementChain(initial_state)
        
        # Step 1: Apply rotation based on action (turn left/right/u-turn)
        # This changes ego's absolute orientation
        chain.apply_rotation(ego_action)
        
        # Step 2: Apply interaction-based orientation adjustment
        # This is the KEY step that determines final relative orientation to target:
        # - LEAD/FOLLOW/PASSING/CO_MOVING/OVERTAKE → ego aligns with target (same direction = 0°)
        # - CROSSING/APPROACHING → ego faces opposite to target (opposite = 180°)
        # - WAITING/YIELDING → neutral, no change from step 1
        chain.apply_interaction_orientation(
            interaction=interaction,
            initial_same_direction=same_direction
        )
        
        # Step 3: Apply position movement for approaching/crossing
        # Only if objects are in different directions (angle > 45°)
        interaction_lower = interaction.lower()
        if 'approaching' in interaction_lower or 'crossing' in interaction_lower:
            chain.apply_approaching_or_crossing(
                target_position=target_position,
                ego_direction=ego_direction,
                target_direction=target_direction,
                time_delta=0.5
            )
        
        return chain
    
    @staticmethod
    def _get_final_ego_orientation_from_chain(
        ego_action: str,
        interaction: str,
        same_direction: bool,
        target_coords: Optional[Dict[str, float]] = None
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Determine ego's final orientation using the chain-based approach.
        
        Args:
            ego_action: The ego action
            interaction: The interaction type
            same_direction: Whether initially facing same direction
            target_coords: Optional target coordinates
            
        Returns:
            Tuple of (orientation_category, chain_summary)
        """
        # Build dummy positions for chain calculation
        target_position = np.array([10.0, 0.0, 0.0])  # Default: ahead
        if target_coords:
            forward = target_coords.get('forward', 10.0)
            right = target_coords.get('right', 0.0)
            target_position = np.array([forward, -right, 0.0])  # Convert to xyz
        
        # Build direction vectors
        ego_direction = np.array([1.0, 0.0]) if same_direction else np.array([-1.0, 0.0])
        target_direction = np.array([1.0, 0.0])  # Assume target moving forward
        
        # Build and execute chain
        chain = MultiStepReasoningGenerator._build_movement_chain(
            ego_action=ego_action,
            interaction=interaction,
            target_position=target_position,
            ego_direction=ego_direction,
            target_direction=target_direction,
            same_direction=same_direction
        )
        
        orientation = chain.get_final_orientation_category()
        summary = chain.get_chain_summary()
        
        logger.debug(f"Chain result: orientation={orientation}, total_rotation={chain.get_total_rotation()}°")
        
        return orientation, summary
    
    @staticmethod
    def _find_reasoning_chain(
        external_interactions: List[Dict[str, Any]],
        spatial_relations: Dict[str, Any],
        annotation: Dict[str, Any],
        internal_interactions: List[Dict[str, Any]] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Generate a reasoning chain by working BACKWARDS:
        1. Select the answer object first (prefer near objects < 20m)
        2. Find which target object ego should interact with to reach answer's position
        3. Generate plausible action/interaction that leads to the answer
        4. Use chain-based approach to determine final orientation
        5. Return the complete chain
        
        Args:
            external_interactions: Not used (kept for interface compatibility)
            spatial_relations: List of spatial relations
            annotation: Annotation data with track IDs and labels
            internal_interactions: List of internal interactions (for ego action)
            
        Returns:
            Reasoning chain dictionary or None
        """
        if not spatial_relations:
            logger.warning("No spatial relations found")
            return None
        
        track_3d = annotation.get('track_3d', [])
        label_3d = annotation.get('label_3d', [])
        track_2d = annotation.get('track_2d', [])
        
        if not track_3d or not label_3d or len(track_3d) < 3:  # Need at least 3: ego, target, answer
            logger.warning(f"Not enough tracks: {len(track_3d) if track_3d else 0}")
            return None
        
        # Build set of track IDs visible in camera views
        visible_track_ids: set = set()
        for camera_tracks in track_2d:
            if isinstance(camera_tracks, list):
                visible_track_ids.update(camera_tracks)
        
        if not visible_track_ids:
            logger.warning("No visible tracks in camera views (track_2d)")
            return None
        
        logger.info(f"Finding reasoning chain from {len(spatial_relations)} spatial relations, {len(visible_track_ids)} visible tracks")
        
        # Build spatial relation graphs
        ego_relations = {}  # ego -> object: {relation, coords}
        object_relations = {}  # object_A -> object_B: {relation, coords}
        
        for relation_data in spatial_relations:
            if not isinstance(relation_data, dict):
                continue
            
            source_obj = relation_data.get('source_object', {})
            target_obj = relation_data.get('target_object', {})
            relations = relation_data.get('relations', [])
            rel_coords = relation_data.get('rel_coords_in_source_frame', {})
            
            source_track = source_obj.get('track_id')
            target_track = target_obj.get('track_id')
            target_label = target_obj.get('label')
            
            if not source_track or not target_track or not relations:
                continue
            
            relation_type = relations[0] if isinstance(relations, list) else relations
            
            # Store ego relations
            if source_track == '00000':
                ego_relations[target_track] = {
                    'relation': relation_type,
                    'coords': rel_coords,
                    'label': target_label
                }
            
            # Store all object-to-object relations
            if source_track not in object_relations:
                object_relations[source_track] = {}
            object_relations[source_track][target_track] = {
                'relation': relation_type,
                'coords': rel_coords,
                'label': target_label
            }
        
        if not ego_relations or not object_relations:
            logger.warning("Insufficient spatial relations")
            return None
        
        logger.info(f"Found {len(ego_relations)} ego relations and {len(object_relations)} object relation sets")
        
        # Step 1: Select an answer object (one that has incoming relations from other objects)
        # Find objects that appear as targets in object_relations AND are visible in camera views
        potential_answers = set()
        for source_track, targets in object_relations.items():
            if source_track in ego_relations:  # Source is visible from ego
                for target_track in targets.keys():
                    # Must be in track_3d, not ego, AND visible in camera views (track_2d)
                    if (target_track in track_3d and 
                        target_track != '00000' and 
                        target_track in visible_track_ids):
                        potential_answers.add(target_track)
        
        if not potential_answers:
            logger.warning("No valid answer candidates")
            return None
        
        # Weighted selection: prefer near objects (< 20m from ego)
        answer_track = MultiStepReasoningGenerator._weighted_choice(
            items=list(potential_answers),
            annotation=annotation,
            track_id_getter=lambda x: x,  # Items are track_ids directly
            near_threshold=MultiStepReasoningGenerator.NEAR_OBJECT_THRESHOLD,
            near_probability=MultiStepReasoningGenerator.NEAR_OBJECT_PROBABILITY
        )
        answer_idx = track_3d.index(answer_track)
        answer_object_id = answer_idx + 1
        answer_label = label_3d[answer_idx] if answer_idx < len(label_3d) else "object"
        
        # Step 2: Find intermediate target objects that have relation to answer
        # These are objects that: (1) ego can see, (2) have spatial relation to answer, (3) visible in camera views
        intermediate_candidates = []
        for intermediate_track in ego_relations.keys():
            if intermediate_track in object_relations:
                if answer_track in object_relations[intermediate_track]:
                    # Must also be visible in camera views (track_2d)
                    if intermediate_track in visible_track_ids:
                        intermediate_candidates.append(intermediate_track)
        
        if not intermediate_candidates:
            logger.warning(f"No intermediate objects lead to answer {answer_track}")
            return None
        
        # Weighted selection: prefer near intermediate targets (< 20m from ego)
        target_track = MultiStepReasoningGenerator._weighted_choice(
            items=intermediate_candidates,
            annotation=annotation,
            track_id_getter=lambda x: x,  # Items are track_ids directly
            near_threshold=MultiStepReasoningGenerator.NEAR_OBJECT_THRESHOLD,
            near_probability=MultiStepReasoningGenerator.NEAR_OBJECT_PROBABILITY
        )
        target_idx = track_3d.index(target_track)
        target_object_id = target_idx + 1
        target_label = label_3d[target_idx] if target_idx < len(label_3d) else "object"
        
        # Step 3: Get the relation from intermediate target to answer
        answer_relation = object_relations[target_track][answer_track]
        answer_relation_type = answer_relation['relation']
        
        # Step 4: Get ego's relation to intermediate target (for generating plausible actions)
        ego_to_target = ego_relations[target_track]
        
        # Step 5: Generate plausible action/interaction based on spatial context
        spatial_context = MultiStepReasoningGenerator._analyze_spatial_context_from_coords(
            ego_to_target['relation'], ego_to_target['coords']
        )
        
        ego_action, interaction = MultiStepReasoningGenerator._generate_plausible_action_interaction(
            target_label=target_label,
            spatial_context=spatial_context,
            internal_interactions=internal_interactions,
            answer_label=answer_label,
            target_to_answer_relation=answer_relation_type
        )
        
        # Step 6: Use chain-based approach to compute final ego orientation
        ego_orientation, chain_summary = MultiStepReasoningGenerator._get_final_ego_orientation_from_chain(
            ego_action=ego_action,
            interaction=interaction,
            same_direction=spatial_context.get('same_direction', True),
            target_coords=ego_to_target.get('coords')
        )
        
        # Step 7: Transform the relation based on final ego orientation
        transformed_relation = RelationTransformer.transform(
            relation=answer_relation_type,
            orientation=ego_orientation
        )
        
        logger.info(
            f"Generated reasoning chain: ego {ego_action} → {interaction} object #{target_object_id} ({target_label}) "
            f"→ ego moves near target → answer: object #{answer_object_id} ({answer_label}) is {transformed_relation} "
            f"(original: {answer_relation_type}, orientation: {ego_orientation})"
        )
        
        # Step 8: Return the complete chain with answer relation info for option generation
        return {
            'ego_action': ego_action,
            'interaction': interaction,
            'target_id': target_object_id,
            'target_label': target_label,
            'target_track_id': target_track,
            'relation': transformed_relation.replace('_', ' '),  # Use transformed relation
            'answer_id': answer_object_id,
            'answer_label': answer_label,
            'answer_track_id': answer_track,
            'answer_relation_type': transformed_relation,  # Use transformed relation for filtering
            'original_relation_type': answer_relation_type,  # Keep original for reference
            'intermediate_object_relations': object_relations.get(target_track, {}),  # For option generation
            'ego_orientation': ego_orientation,  # For distractor filtering
            'ego_relations': ego_relations,  # Object positions relative to ego
            'spatial_context': spatial_context,  # Ego-target spatial info
            'movement_chain': chain_summary  # Chain-based movement summary
        }
    
    @staticmethod
    def _analyze_spatial_context_from_coords(
        relation: str,
        coords: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        Analyze spatial context from relation name and coordinates.
        
        Args:
            relation: Relation string (e.g., 'ahead_right_of')
            coords: Relative coordinates {'forward': float, 'right': float, 'up': float}
            
        Returns:
            Dictionary with position and direction info
        """
        relation_lower = relation.lower()
        forward = coords.get('forward', 0)
        right = coords.get('right', 0)
        
        # Determine position
        position = None
        if 'ahead' in relation_lower or 'front' in relation_lower:
            position = 'ahead'
        elif 'behind' in relation_lower or 'rear' in relation_lower or 'back' in relation_lower:
            position = 'behind'
        elif 'left' in relation_lower:
            position = 'left'
        elif 'right' in relation_lower:
            position = 'right'
        
        # Estimate direction based on forward distance
        same_direction = forward > 0
        similar_direction = forward > 0
        
        return {
            'position': position,
            'relation': relation_lower,
            'same_direction': same_direction,
            'similar_direction': similar_direction,
            'same_lane': None,
            'details': coords
        }
    
    @staticmethod
    def _generate_plausible_action_interaction(
        target_label: str,
        spatial_context: Dict[str, Any],
        internal_interactions: List[Dict[str, Any]],
        answer_label: str,
        target_to_answer_relation: str
    ) -> Tuple[str, str]:
        """
        Generate plausible ego action and interaction based on spatial context.
        
        Uses a structured approach instead of nested if-else:
        1. Classify target object type
        2. Determine answer position relative to target
        3. Select appropriate actions based on position and type
        
        Args:
            target_label: Label of the intermediate target object
            spatial_context: Spatial context of ego -> target
            internal_interactions: Internal interactions for possible ego action
            answer_label: Label of the answer object
            target_to_answer_relation: Relation from target to answer (e.g., 'rear_left_of')
            
        Returns:
            Tuple of (ego_action, interaction_type)
        """
        position = spatial_context.get('position')
        same_direction = spatial_context.get('same_direction')
        
        # Classify target object type
        target_lower = target_label.lower()
        is_vehicle = any(v in target_lower for v in [
            'car', 'vehicle', 'truck', 'bus', 'trailer', 'motorcycle',
            'articulated_bus', 'box_truck', 'large_vehicle', 'regular_vehicle', 'school_bus',
            'other_vehicle'
        ])
        is_vulnerable = any(v in target_lower for v in [
            'pedestrian', 'person', 'cyclist', 'bicycle', 'bicyclist', 'adult', 'child',
            'stroller', 'personal_mobility', 'police_officer', 'construction_worker',
            'motorcyclist'
        ])
        is_static = any(s in target_lower for s in [
            'cone', 'barrier', 'construction', 'debris', 'bollard', 'sign',
            'trafficcone', 'traffic_cone', 'construction_barrel', 'construction_cone',
            'mobile_pedestrian_crossing_sign', 'message_board_trailer', 'traffic_sign'
        ])
        
        # Parse answer position relative to target
        answer_relation_lower = target_to_answer_relation.lower()
        answer_position = None
        if 'ahead' in answer_relation_lower or 'front' in answer_relation_lower:
            answer_position = 'ahead'
        elif 'behind' in answer_relation_lower or 'rear' in answer_relation_lower or 'back' in answer_relation_lower:
            answer_position = 'behind'
        elif 'left' in answer_relation_lower:
            answer_position = 'left'
        elif 'right' in answer_relation_lower:
            answer_position = 'right'
        
        logger.info(f"Generating actions: ego->target position={position}, target->answer position={answer_position}")
        
        # Action/interaction selection rules based on position combinations
        action_rules: Dict[Tuple[Optional[str], Optional[str]], Dict[str, Any]] = {
            # (ego_to_target_position, target_to_answer_position): {actions, interactions_vehicle, interactions_other}
            ('ahead', 'ahead'): {
                'actions': [IntrinsicInteraction.MOVING_FORWARD, IntrinsicInteraction.DECELERATE, IntrinsicInteraction.STOPPED],
                'vehicle_interactions': [ExtrinsicInteraction.FOLLOW, ExtrinsicInteraction.APPROACHING, ExtrinsicInteraction.WAITING],
                'other_interactions': [ExtrinsicInteraction.APPROACHING, ExtrinsicInteraction.WAITING]
            },
            ('ahead', 'behind'): {
                'actions': [IntrinsicInteraction.MOVING_FORWARD, IntrinsicInteraction.ACCELERATE],
                'vehicle_interactions': [ExtrinsicInteraction.FOLLOW, ExtrinsicInteraction.APPROACHING, ExtrinsicInteraction.OVERTAKE],
                'other_interactions': [ExtrinsicInteraction.APPROACHING]
            },
            ('ahead', 'left'): {
                'actions': [IntrinsicInteraction.MOVING_FORWARD, IntrinsicInteraction.DECELERATE, IntrinsicInteraction.TURN_LEFT],
                'vehicle_interactions': [ExtrinsicInteraction.FOLLOW, ExtrinsicInteraction.APPROACHING, ExtrinsicInteraction.CO_MOVING],
                'other_interactions': [ExtrinsicInteraction.APPROACHING, ExtrinsicInteraction.YIELDING]
            },
            ('ahead', 'right'): {
                'actions': [IntrinsicInteraction.MOVING_FORWARD, IntrinsicInteraction.DECELERATE, IntrinsicInteraction.TURN_RIGHT],
                'vehicle_interactions': [ExtrinsicInteraction.FOLLOW, ExtrinsicInteraction.APPROACHING, ExtrinsicInteraction.CO_MOVING],
                'other_interactions': [ExtrinsicInteraction.APPROACHING, ExtrinsicInteraction.YIELDING]
            },
            ('behind', 'ahead'): {
                'actions': [IntrinsicInteraction.MOVING_BACKWARD, IntrinsicInteraction.STOPPED, IntrinsicInteraction.U_TURN],
                'vehicle_interactions': [ExtrinsicInteraction.APPROACHING, ExtrinsicInteraction.WAITING],
                'other_interactions': [ExtrinsicInteraction.APPROACHING, ExtrinsicInteraction.WAITING]
            },
            ('behind', 'behind'): {
                'actions': [IntrinsicInteraction.MOVING_BACKWARD, IntrinsicInteraction.DECELERATE, IntrinsicInteraction.U_TURN],
                'vehicle_interactions': [ExtrinsicInteraction.APPROACHING],
                'other_interactions': [ExtrinsicInteraction.APPROACHING]
            },
            ('behind', 'left'): {
                'actions': [IntrinsicInteraction.MOVING_BACKWARD, IntrinsicInteraction.STOPPED, IntrinsicInteraction.U_TURN],
                'vehicle_interactions': [ExtrinsicInteraction.APPROACHING, ExtrinsicInteraction.WAITING],
                'other_interactions': [ExtrinsicInteraction.APPROACHING, ExtrinsicInteraction.WAITING]
            },
            ('behind', 'right'): {
                'actions': [IntrinsicInteraction.MOVING_BACKWARD, IntrinsicInteraction.STOPPED, IntrinsicInteraction.U_TURN],
                'vehicle_interactions': [ExtrinsicInteraction.APPROACHING, ExtrinsicInteraction.WAITING],
                'other_interactions': [ExtrinsicInteraction.APPROACHING, ExtrinsicInteraction.WAITING]
            },
            ('left', 'left'): {
                'actions': [IntrinsicInteraction.LANE_CHANGE_LEFT, IntrinsicInteraction.TURN_LEFT, IntrinsicInteraction.MOVING_FORWARD],
                'vehicle_interactions': [ExtrinsicInteraction.PASSING, ExtrinsicInteraction.APPROACHING, ExtrinsicInteraction.CO_MOVING],
                'other_interactions': [ExtrinsicInteraction.YIELDING, ExtrinsicInteraction.APPROACHING]
            },
            ('left', 'right'): {
                'actions': [IntrinsicInteraction.LANE_CHANGE_LEFT, IntrinsicInteraction.TURN_LEFT, IntrinsicInteraction.MOVING_FORWARD],
                'vehicle_interactions': [ExtrinsicInteraction.PASSING, ExtrinsicInteraction.APPROACHING, ExtrinsicInteraction.CROSSING],
                'other_interactions': [ExtrinsicInteraction.PASSING, ExtrinsicInteraction.APPROACHING]
            },
            ('left', 'ahead'): {
                'actions': [IntrinsicInteraction.LANE_CHANGE_LEFT, IntrinsicInteraction.TURN_LEFT, IntrinsicInteraction.MOVING_FORWARD],
                'vehicle_interactions': [ExtrinsicInteraction.PASSING, ExtrinsicInteraction.APPROACHING],
                'other_interactions': [ExtrinsicInteraction.APPROACHING]
            },
            ('left', 'behind'): {
                'actions': [IntrinsicInteraction.LANE_CHANGE_LEFT, IntrinsicInteraction.TURN_LEFT],
                'vehicle_interactions': [ExtrinsicInteraction.PASSING, ExtrinsicInteraction.CO_MOVING],
                'other_interactions': [ExtrinsicInteraction.PASSING]
            },
            ('right', 'right'): {
                'actions': [IntrinsicInteraction.LANE_CHANGE_RIGHT, IntrinsicInteraction.TURN_RIGHT, IntrinsicInteraction.MOVING_FORWARD],
                'vehicle_interactions': [ExtrinsicInteraction.PASSING, ExtrinsicInteraction.APPROACHING, ExtrinsicInteraction.CO_MOVING],
                'other_interactions': [ExtrinsicInteraction.YIELDING, ExtrinsicInteraction.APPROACHING]
            },
            ('right', 'left'): {
                'actions': [IntrinsicInteraction.LANE_CHANGE_RIGHT, IntrinsicInteraction.TURN_RIGHT, IntrinsicInteraction.MOVING_FORWARD],
                'vehicle_interactions': [ExtrinsicInteraction.PASSING, ExtrinsicInteraction.APPROACHING, ExtrinsicInteraction.CROSSING],
                'other_interactions': [ExtrinsicInteraction.PASSING, ExtrinsicInteraction.APPROACHING]
            },
            ('right', 'ahead'): {
                'actions': [IntrinsicInteraction.LANE_CHANGE_RIGHT, IntrinsicInteraction.TURN_RIGHT, IntrinsicInteraction.MOVING_FORWARD],
                'vehicle_interactions': [ExtrinsicInteraction.PASSING, ExtrinsicInteraction.APPROACHING],
                'other_interactions': [ExtrinsicInteraction.APPROACHING]
            },
            ('right', 'behind'): {
                'actions': [IntrinsicInteraction.LANE_CHANGE_RIGHT, IntrinsicInteraction.TURN_RIGHT],
                'vehicle_interactions': [ExtrinsicInteraction.PASSING, ExtrinsicInteraction.CO_MOVING],
                'other_interactions': [ExtrinsicInteraction.PASSING]
            },
        }
        
        # Get action/interaction rules for this position combination
        rule_key = (position, answer_position)
        rules = action_rules.get(rule_key)
        
        # Fallback to default rules if no specific rule found
        if not rules:
            rules = {
                'actions': [IntrinsicInteraction.MOVING_FORWARD, IntrinsicInteraction.DECELERATE],
                'vehicle_interactions': [ExtrinsicInteraction.APPROACHING, ExtrinsicInteraction.CO_MOVING],
                'other_interactions': [ExtrinsicInteraction.APPROACHING, ExtrinsicInteraction.CO_MOVING]
            }
        
        plausible_actions = rules['actions']
        plausible_interactions = rules['vehicle_interactions'] if is_vehicle else rules['other_interactions']
        
        # Special handling for vulnerable road users
        if is_vulnerable:
            plausible_interactions = [ExtrinsicInteraction.YIELDING, ExtrinsicInteraction.APPROACHING]
        
        logger.info(f"Selected {len(plausible_actions)} actions and {len(plausible_interactions)} interactions")
        
        # Try to use real action from internal_interactions if it's in plausible list
        ego_action_enum = None
        if internal_interactions and plausible_actions:
            for int_interaction in internal_interactions:
                if isinstance(int_interaction, dict):
                    action = int_interaction.get('action', '')
                    if action and 'IntrinsicInteraction.' in action:
                        action_name = action.split('.')[-1].lower()
                        for intrinsic in plausible_actions:
                            if intrinsic.value.lower() == action_name:
                                ego_action_enum = intrinsic
                                break
                    if ego_action_enum:
                        break
        
        # If no action found or not plausible, randomly select from plausible list
        if not ego_action_enum and plausible_actions:
            ego_action_enum = random.choice(plausible_actions)
        elif not ego_action_enum:
            ego_action_enum = IntrinsicInteraction.MOVING_FORWARD
        
        ego_action = ego_action_enum.value.replace('_', ' ')
        
        # Randomly select from plausible interactions
        if plausible_interactions:
            interaction_enum = random.choice(plausible_interactions)
        else:
            interaction_enum = ExtrinsicInteraction.APPROACHING
        
        interaction = interaction_enum.value.replace('_', ' ')
        return ego_action, interaction
    
    @staticmethod
    def _generate_options(
        correct_id: int,
        annotation: Dict[str, Any],
        reasoning_chain: Dict[str, Any],
        num_options: int = 4
    ) -> Tuple[List[Tuple[int, str]], int]:
        """
        Generate multiple choice options with direction-aware filtering.
        
        Ensures:
        1. Other options are NOT in the same transformed relation as the correct answer
        2. The ego action/interaction is PLAUSIBLE for all distractor objects
        
        Args:
            correct_id: Correct object ID (1-indexed)
            annotation: Annotation data
            reasoning_chain: Reasoning chain with relation info, ego action, and spatial context
            num_options: Number of options
            
        Returns:
            Tuple of (options list, correct answer index)
        """
        track_3d = annotation.get('track_3d', [])
        label_3d = annotation.get('label_3d', [])
        track_2d = annotation.get('track_2d', [])
        
        # Build set of track IDs visible in camera views
        visible_track_ids: set = set()
        for camera_tracks in track_2d:
            if isinstance(camera_tracks, list):
                visible_track_ids.update(camera_tracks)
        
        if correct_id < 1 or correct_id > len(track_3d):
            correct_id = 1
        
        correct_class = label_3d[correct_id - 1] if correct_id <= len(label_3d) else "unknown"
        correct_track = track_3d[correct_id - 1]
        
        # Get action/interaction for plausibility check
        ego_action = reasoning_chain.get('ego_action', '')
        interaction = reasoning_chain.get('interaction', '')
        ego_orientation = reasoning_chain.get('ego_orientation', 'same')
        ego_relations = reasoning_chain.get('ego_relations', {})
        
        # Get the answer relation type (already transformed) to filter conflicts
        answer_relation_type = reasoning_chain.get('answer_relation_type', '').lower()
        intermediate_relations = reasoning_chain.get('intermediate_object_relations', {})
        
        # Helper to get object position and direction from ego_relations
        def get_object_spatial_info(obj_track_id: str) -> Tuple[Optional[str], bool]:
            """Get object position and same_direction flag."""
            if obj_track_id not in ego_relations:
                return None, True  # Unknown, assume same direction
            
            rel_info = ego_relations[obj_track_id]
            relation = rel_info.get('relation', '').lower()
            coords = rel_info.get('coords', {})
            
            # Parse position
            position = None
            if 'ahead' in relation or 'front' in relation:
                position = 'ahead'
            elif 'behind' in relation or 'rear' in relation or 'back' in relation:
                position = 'behind'
            elif 'left' in relation:
                position = 'left'
            elif 'right' in relation:
                position = 'right'
            
            # Determine direction from forward coordinate
            forward = coords.get('forward', 0)
            same_direction = forward > 0  # Positive forward = moving same direction
            
            return position, same_direction
        
        # Find objects that share the SAME transformed relation type (these are conflicts)
        conflicting_tracks = set()
        for target_track, rel_info in intermediate_relations.items():
            if target_track == correct_track:
                continue
            rel_type = rel_info.get('relation', '').lower()
            
            # Transform this relation too for fair comparison
            transformed_rel = RelationTransformer.transform(
                relation=rel_type,
                orientation=ego_orientation
            )
            
            if transformed_rel == answer_relation_type:
                # This object has the same transformed relation as the answer
                if target_track in track_3d:
                    conflicting_obj_id = track_3d.index(target_track) + 1
                    conflicting_tracks.add(conflicting_obj_id)
        
        logger.info(f"Found {len(conflicting_tracks)} conflicting objects with same transformed relation '{answer_relation_type}'")
        
        # Find objects for which the action is implausible
        implausible_tracks = set()
        for obj_idx, obj_track in enumerate(track_3d):
            obj_id = obj_idx + 1
            if obj_id == correct_id:
                continue  # Skip correct answer
            
            position, same_direction = get_object_spatial_info(obj_track)
            
            if not MultiStepReasoningGenerator._is_action_plausible_for_object(
                ego_action=ego_action,
                interaction=interaction,
                object_position=position,
                object_same_direction=same_direction
            ):
                implausible_tracks.add(obj_id)
        
        logger.info(f"Found {len(implausible_tracks)} objects for which action '{ego_action}/{interaction}' is implausible")
        
        # Create options: correct answer + plausible non-conflicting distractors
        all_options = [(correct_id, correct_class)]
        
        # Priority 1: Add plausible, non-conflicting distractors (must be visible in camera views)
        plausible_non_conflicting = []
        for obj_id in range(1, len(track_3d) + 1):
            obj_track = track_3d[obj_id - 1]
            # Must be visible in camera views (track_2d), not correct answer, not conflicting, not implausible
            if (obj_id != correct_id and 
                obj_id not in conflicting_tracks and 
                obj_id not in implausible_tracks and
                obj_track in visible_track_ids):
                obj_class = label_3d[obj_id - 1] if obj_id <= len(label_3d) else "unknown"
                plausible_non_conflicting.append((obj_id, obj_class))
        
        all_options.extend(plausible_non_conflicting)
        
        # Priority 2: If not enough, add implausible but non-conflicting (less ideal, must be visible)
        if len(all_options) < num_options:
            logger.warning(f"Not enough plausible options ({len(all_options)}), adding implausible ones")
            for obj_id in implausible_tracks:
                if len(all_options) >= num_options:
                    break
                obj_track = track_3d[obj_id - 1] if obj_id <= len(track_3d) else None
                if obj_id not in conflicting_tracks and obj_track in visible_track_ids:
                    obj_class = label_3d[obj_id - 1] if obj_id <= len(label_3d) else "unknown"
                    all_options.append((obj_id, obj_class))
        
        # Priority 3: If still not enough, add conflicting ones as last resort (must be visible)
        if len(all_options) < num_options:
            logger.warning(f"Still not enough options ({len(all_options)}), adding conflicting ones")
            for obj_id in conflicting_tracks:
                if len(all_options) >= num_options:
                    break
                obj_track = track_3d[obj_id - 1] if obj_id <= len(track_3d) else None
                if obj_track in visible_track_ids:
                    obj_class = label_3d[obj_id - 1] if obj_id <= len(label_3d) else "unknown"
                    all_options.append((obj_id, obj_class))
        
        # Limit to num_options
        if len(all_options) > num_options:
            # Keep correct answer and sample distractors
            distractors = [opt for opt in all_options if opt[0] != correct_id]
            selected_distractors = random.sample(distractors, num_options - 1)
            all_options = [(correct_id, correct_class)] + selected_distractors
        
        # Shuffle
        correct_answer = 0
        shuffled_indices = list(range(len(all_options)))
        random.shuffle(shuffled_indices)
        shuffled_options = [all_options[i] for i in shuffled_indices]
        correct_answer = shuffled_indices.index(0)
        
        return shuffled_options, correct_answer