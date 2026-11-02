from typing import Dict

# Camera ordering for different datasets (layout order for visualization)
CAMERA_ORDER = {
    'nuscenes': [
        'CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
        'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT'
    ],
    # ONCE layout: row1 = cam06, cam03, cam01, cam08; row2 = cam05, cam09, cam07 (centered)
    'once': ['cam06', 'cam03', 'cam01', 'cam08', 'cam05', 'cam09', 'cam07'],
    'waymo': ['FRONT_LEFT', 'FRONT', 'FRONT_RIGHT', 'SIDE_LEFT', 'SIDE_RIGHT'],
    'truckscenes': [
        'CAMERA_LEFT_FRONT', 'CAMERA_RIGHT_FRONT',
        'CAMERA_LEFT_BACK', 'CAMERA_RIGHT_BACK'
    ],
    'argoverse2': [
        'ring_front_left', 'ring_front_center', 'ring_front_right',
        'ring_side_left', 'ring_side_right',
        'ring_rear_left', 'ring_rear_right'
    ]
}

# Display names for camera labels (e.g. ONCE: show "front left" instead of "cam06")
CAMERA_DISPLAY_NAMES: Dict[str, Dict[str, str]] = {
    'nuscenes': {
        'CAM_FRONT_LEFT': 'front left',
        'CAM_FRONT': 'front',
        'CAM_FRONT_RIGHT': 'front right',
        'CAM_BACK_LEFT': 'rear left',
        'CAM_BACK': 'rear',
        'CAM_BACK_RIGHT': 'rear right',
    },
    'once': {
        'cam06': 'front left',
        'cam03': 'front',
        'cam01': 'front',
        'cam08': 'front right',
        'cam05': 'rear left',
        'cam09': 'rear',
        'cam07': 'rear right',    # swapped: was rear right
    },
    'waymo': {
        'FRONT_LEFT': 'front left',
        'FRONT': 'front',
        'FRONT_RIGHT': 'front right',
        'SIDE_LEFT': 'side left',
        'SIDE_RIGHT': 'side right',
    },
    'truckscenes': {
        'CAMERA_LEFT_FRONT': 'front left',
        'CAMERA_RIGHT_FRONT': 'front right',
        'CAMERA_LEFT_BACK': 'rear left',
        'CAMERA_RIGHT_BACK': 'rear right',
    },
    'argoverse2': {
        'ring_front_left': 'front left',
        'ring_front_center': 'front',
        'ring_front_right': 'front right',
        'ring_side_left': 'side left',
        'ring_side_right': 'side right',
        'ring_rear_left': 'rear left',
        'ring_rear_right': 'rear right',
    }
}