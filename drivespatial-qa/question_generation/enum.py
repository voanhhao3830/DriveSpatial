from enum import Enum

class ConstructionQuestionType(str, Enum):
    COGNITIVE_SCENE = "cognitive_scene"
    PERSPECTIVE_CAMERA_MATCHING = "perspective_camera_matching"
    EGO_ROTATION = "ego_rotation"
    CAMERA_ORDERING = "camera_ordering"
    LEAVE_ONE_CAMERA_OUT = "leave_one_camera_out"

class RelationQuestionType(str, Enum):
    MULTI_STEPS_REASONING = "multi_steps_reasoning"
    ALLOCENTRIC_IMAGINATION = "allocentric_imagination"
    SPATIAL_COMPABILITY = "spatial_compability"
    MULTI_VIEW_OBJECT_MATCHING = "multi_view_object_matching"
    MULTI_VIEW_DEPTH_AWARENESS = "multi_view_depth_awareness"
    DIRECTION = "direction"
    DISTANCE = "distance"
    ABSOLUTE_DISTANCE = "absolute_distance"
    ABSOLUTE_SCENE_COUNTING = "absolute_scene_counting"
    
class CausualityReasoningQuestionType(str, Enum):
    EVENT_ORDERING = "event_ordering"
    TRAJECTORY_REASONING = "trajectory_reasoning"
    OCCLUSION_AWARENESS = "occlusion_awareness"
    OBJECT_MANIPULATION = "object_manipulation"
    ACTION_REASONING = "action_reasoning"
    INTERACTION_REASONING = "interaction_reasoning"