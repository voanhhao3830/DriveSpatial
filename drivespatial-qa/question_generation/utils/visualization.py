"""
Unified Visualization Utilities for Question Generation

Provides consistent visualization functions for multi-view camera displays
with optional object ID annotations and bounding boxes.
"""

import os
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
from PIL import Image, ImageDraw, ImageFont

from sources.question_generation.utils.constants import CAMERA_ORDER, CAMERA_DISPLAY_NAMES


def visualize_multiview(
    data: Dict[str, Any],
    output_path: str,
    source: str,
    show_object_ids: bool = False,
    highlight_objects: Optional[List[Dict[str, Any]]] = None,
    show_only_highlighted: bool = False,
    hide_object_ids_for_highlighted: bool = False
) -> None:
    """
    Unified multi-view visualization with optional object ID annotations.
    
    Args:
        data: Data record containing views and annotations
        output_path: Path to save the visualization
        source: Data source name (e.g., 'nuscenes', 'waymo')
        show_object_ids: Whether to show object IDs and bounding boxes
        highlight_objects: Optional list of objects to highlight with special colors
                          Each dict should contain EITHER:
                          - 'object_id': int - The object ID to highlight (global across cameras)
                          OR:
                          - 'camera': str - Camera name (e.g., 'CAM_FRONT')
                          - 'bbox_idx_in_camera': int - Index of bbox in that camera's bbox list
                          Plus:
                          - 'color': tuple - BGR color (e.g., (255, 0, 0))
                          - 'label': str - Label to display (e.g., '(you)', '(target)')
        show_only_highlighted: If True, only show highlighted objects (hide others)
        hide_object_ids_for_highlighted: If True, only show custom labels for highlighted objects (no #ID)
    
    Example:
        # Without IDs (clean view)
        visualize_multiview(data, "output.jpg", "nuscenes")
        
        # With IDs
        visualize_multiview(data, "output.jpg", "nuscenes", show_object_ids=True)
        
        # With highlighted objects (object_id-based)
        visualize_multiview(
            data, "output.jpg", "nuscenes", 
            show_object_ids=True,
            highlight_objects=[
                {'object_id': 5, 'color': (255, 0, 0), 'label': '(ego)'},
                {'object_id': 12, 'color': (0, 0, 255), 'label': '(target)'}
            ]
        )
        
        # With highlighted objects (camera-based, for same object in different views)
        visualize_multiview(
            data, "output.jpg", "nuscenes",
            show_object_ids=True,
            highlight_objects=[
                {'camera': 'CAM_FRONT', 'bbox_idx_in_camera': 2, 'color': (255, 0, 0), 'label': '(1)'},
                {'camera': 'CAM_BACK', 'bbox_idx_in_camera': 5, 'color': (255, 0, 255), 'label': '(2)'}
            ],
            show_only_highlighted=True,
            hide_object_ids_for_highlighted=True
        )
    """
    views = data.get("views", {})
    annotation = data.get("annotation", {})
    
    # Get annotation data for object IDs
    bbox_2d = annotation.get('bbox_2d', [])
    track_2d = annotation.get('track_2d', [])
    track_3d = annotation.get('track_3d', [])
    label_3d = annotation.get('label_3d', [])
    
    source_lower = source.lower() if source else None
    camera_order = CAMERA_ORDER.get(source_lower, [])
    
    # Create mapping from camera name to original index
    original_camera_order = list(views.keys())
    camera_name_to_original_idx = {name: idx for idx, name in enumerate(original_camera_order)}
    
    # Create highlight lookup for faster access
    # Support three modes: object_id-based, track_id-based, or camera+bbox_idx-based
    highlight_lookup = {}  # object_id -> highlight_info
    track_id_highlight_lookup = {}  # track_id -> highlight_info (includes optional object_id for consistent labeling)
    camera_based_highlight = {}
    if highlight_objects:
        for obj_info in highlight_objects:
            # Check if using camera-specific identification
            if 'camera' in obj_info and 'bbox_idx_in_camera' in obj_info:
                cam = obj_info['camera']
                bbox_idx = obj_info['bbox_idx_in_camera']
                if cam not in camera_based_highlight:
                    camera_based_highlight[cam] = {}
                camera_based_highlight[cam][bbox_idx] = {
                    'color': obj_info.get('color', (0, 255, 0)),
                    'label_suffix': obj_info.get('label', ''),
                    'object_id': obj_info.get('object_id')  # Optional consistent object_id
                }
            # Check for track_id-based highlighting
            elif 'track_id' in obj_info:
                track_id = obj_info['track_id']
                track_id_highlight_lookup[track_id] = {
                    'color': obj_info.get('color', (0, 255, 0)),
                    'label_suffix': obj_info.get('label', ''),
                    'object_id': obj_info.get('object_id')  # Optional consistent object_id (from reference frame)
                }
            # Otherwise use object_id-based (original behavior)
            elif 'object_id' in obj_info:
                obj_id = obj_info['object_id']
                highlight_lookup[obj_id] = {
                    'color': obj_info.get('color', (0, 255, 0)),
                    'label_suffix': obj_info.get('label', '')
                }
    
    # Default color for non-highlighted objects
    default_color = (0, 255, 0)  # Green in BGR
    
    # Load and process camera views
    camera_images = []
    for cam_name in camera_order:
        if cam_name not in views:
            continue
        
        cam_path = views[cam_name]
        if isinstance(cam_path, dict):
            cam_path = cam_path.get("path")
        
        if not cam_path or not os.path.exists(cam_path):
            continue
        
        # Load image
        img = cv2.imread(cam_path)
        if img is None:
            continue
        
        # Add object IDs and bounding boxes if requested
        if show_object_ids:
            original_cam_idx = camera_name_to_original_idx.get(cam_name)
            
            if original_cam_idx is not None and original_cam_idx < len(bbox_2d) and original_cam_idx < len(track_2d):
                boxes = bbox_2d[original_cam_idx]
                tracks = track_2d[original_cam_idx]
                
                # Collect all objects and separate highlighted vs non-highlighted
                non_highlighted_objects = []
                highlighted_objects = []
                
                for bbox_idx_in_cam, (box, track_id) in enumerate(zip(boxes, tracks)):
                    if track_id not in track_3d:
                        continue
                    
                    # Default object_id from current frame's track_3d (may vary between frames!)
                    frame_object_id = track_3d.index(track_id) + 1  # 1-indexed
                    object_class = label_3d[track_3d.index(track_id)] if track_3d.index(track_id) < len(label_3d) else "unknown"
                    
                    # Check if this bbox is highlighted (camera-based, track_id-based, or object_id-based)
                    is_highlighted = False
                    highlight_info = None
                    
                    # First check camera-based highlighting
                    if cam_name in camera_based_highlight and bbox_idx_in_cam in camera_based_highlight[cam_name]:
                        is_highlighted = True
                        highlight_info = camera_based_highlight[cam_name][bbox_idx_in_cam]
                    # Then check track_id-based highlighting
                    elif track_id in track_id_highlight_lookup:
                        is_highlighted = True
                        highlight_info = track_id_highlight_lookup[track_id]
                    # Then check object_id-based highlighting
                    elif frame_object_id in highlight_lookup:
                        is_highlighted = True
                        highlight_info = highlight_lookup[frame_object_id]
                    
                    if is_highlighted and highlight_info:
                        color = highlight_info['color']
                        label_suffix = highlight_info['label_suffix']
                        # Use the consistent object_id from highlight_info if provided,
                        # otherwise fall back to frame-based object_id
                        # This ensures stable IDs across frames (fixes "object ID increasing over time" bug)
                        object_id = highlight_info.get('object_id') or frame_object_id
                        # Option to hide object IDs for highlighted objects
                        if hide_object_ids_for_highlighted:
                            label = label_suffix  # Only show custom label like "(1)"
                        else:
                            label = f"#{object_id} {label_suffix}"
                        highlighted_objects.append((box, object_id, label, color))
                    else:
                        # Skip non-highlighted objects if show_only_highlighted is True
                        if show_only_highlighted:
                            continue
                        object_id = frame_object_id
                        color = default_color
                        label = f"#{object_id}"
                        non_highlighted_objects.append((box, object_id, label, color))
                
                # Draw non-highlighted objects first (bottom layer)
                for box, object_id, label, color in non_highlighted_objects:
                    x1, y1, x2, y2 = map(int, box)
                    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                    
                    # Draw ID label
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 0.9  # Increased from 0.7
                    thickness = 2
                    
                    # Get text size for background
                    (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, thickness)
                    
                    # Draw background rectangle for label
                    padding = 8  # Increased from 5
                    cv2.rectangle(img, (x1, y1 - text_height - padding*2), (x1 + text_width + padding*2, y1), color, -1)
                    
                    # Draw text in black
                    cv2.putText(img, label, (x1 + padding, y1 - padding), font, font_scale, (0, 0, 0), thickness)
                
                # Draw highlighted objects last (top layer) with thicker lines
                for box, object_id, label, color in highlighted_objects:
                    x1, y1, x2, y2 = map(int, box)
                    # Use thicker lines for highlighted objects
                    cv2.rectangle(img, (x1, y1), (x2, y2), color, 4)
                    
                    # Draw ID label
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 1.0  # Increased from 0.8 for highlighted objects
                    thickness = 2
                    
                    # Get text size for background
                    (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, thickness)
                    
                    # Draw background rectangle for label (thicker for visibility)
                    padding = 10  # Increased from 6
                    cv2.rectangle(img, (x1, y1 - text_height - padding*2), (x1 + text_width + padding*2, y1), color, -1)
                    
                    # Draw text in black
                    cv2.putText(img, label, (x1 + padding, y1 - padding), font, font_scale, (0, 0, 0), thickness)
        
        # Add camera name label (use display name for ONCE etc.)
        display_name = CAMERA_DISPLAY_NAMES.get(source_lower, {}).get(cam_name, cam_name)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.2  # Increased from 1.0
        thickness = 3  # Increased from 2
        
        # Get text size
        (text_width, text_height), baseline = cv2.getTextSize(display_name, font, font_scale, thickness)
        
        # Draw black background rectangle with larger padding
        padding = 10  # Increased from 5
        bg_x1 = 10 - padding
        bg_y1 = 35 - text_height - padding  # Adjusted Y position
        bg_x2 = 10 + text_width + padding
        bg_y2 = 35 + baseline + padding
        cv2.rectangle(img, (bg_x1, bg_y1), (bg_x2, bg_y2), (0, 0, 0), -1)
        
        # Draw black text on black background (will use white background for camera names)
        cv2.rectangle(img, (bg_x1, bg_y1), (bg_x2, bg_y2), (255, 255, 255), -1)
        cv2.putText(img, display_name, (10, 35), font, font_scale, (0, 0, 0), thickness)
        
        camera_images.append((cam_name, img))
    
    if not camera_images:
        # Create placeholder if no images available
        placeholder = Image.new('RGB', (1920, 1080), color=(128, 128, 128))
        placeholder.save(output_path)
        return
    
    # Resize images to uniform size
    target_height = 480
    resized_images = []
    for cam_name, img in camera_images:
        h, w = img.shape[:2]
        aspect_ratio = w / h
        new_width = int(target_height * aspect_ratio)
        resized = cv2.resize(img, (new_width, target_height))
        resized_images.append((cam_name, resized))
    
    # Find max width
    max_width = max(img.shape[1] for _, img in resized_images)
    
    # Special layout for Waymo: 2 rows - Front cameras on top, side cameras on bottom
    if source_lower == 'waymo':
        # Custom Waymo layout:
        # Row 1: FRONT_LEFT, FRONT, FRONT_RIGHT (centered, no gaps)
        # Row 2: SIDE_LEFT [space] SIDE_RIGHT
        canvas_width = max_width * 3  # Width for 3 front cameras
        canvas_height = target_height * 2  # 2 rows
        canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)
        
        for cam_name, img in resized_images:
            h, w = img.shape[:2]
            if cam_name == 'FRONT_LEFT':
                x, y = 0, 0  # Top left
                canvas[y:y+h, x:x+w] = img
            elif cam_name == 'FRONT':
                x, y = max_width, 0  # Top center
                canvas[y:y+h, x:x+w] = img
            elif cam_name == 'FRONT_RIGHT':
                x, y = max_width * 2, 0  # Top right
                canvas[y:y+h, x:x+w] = img
            elif cam_name == 'SIDE_LEFT':
                x, y = 0, target_height  # Bottom left
                canvas[y:y+h, x:x+w] = img
            elif cam_name == 'SIDE_RIGHT':
                x, y = max_width * 2, target_height  # Bottom right (with space in middle)
                canvas[y:y+h, x:x+w] = img
    elif source_lower == 'argoverse2':
        # AV2 layout: center camera center-aligned; 2-cam rows centered under it
        # Row 1: ring_front_left | ring_front_center (center) | ring_front_right
        # Row 2: ring_side_left | ring_side_right (row centered)
        # Row 3: ring_rear_left | ring_rear_right (row centered)
        canvas_width = max_width * 3
        canvas_height = target_height * 3
        canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)
        row2_offset_x = (canvas_width - 2 * max_width) // 2  # center the 2-cam row
        row3_offset_x = row2_offset_x
        for cam_name, img in resized_images:
            h, w = img.shape[:2]
            if cam_name == 'ring_front_left':
                x, y = 0, 0
                canvas[y:y+h, x:x+w] = img
            elif cam_name == 'ring_front_center':
                x, y = max_width, 0  # center of top row
                canvas[y:y+h, x:x+w] = img
            elif cam_name == 'ring_front_right':
                x, y = max_width * 2, 0
                canvas[y:y+h, x:x+w] = img
            elif cam_name == 'ring_side_left':
                x, y = row2_offset_x, target_height
                canvas[y:y+h, x:x+w] = img
            elif cam_name == 'ring_side_right':
                x, y = row2_offset_x + max_width, target_height
                canvas[y:y+h, x:x+w] = img
            elif cam_name == 'ring_rear_left':
                x, y = row3_offset_x, target_height * 2
                canvas[y:y+h, x:x+w] = img
            elif cam_name == 'ring_rear_right':
                x, y = row3_offset_x + max_width, target_height * 2
                canvas[y:y+h, x:x+w] = img
    elif source_lower == 'once':
        # ONCE layout: row1 = cam06, cam03, cam01, cam08 (4 cams); row2 = cam05, cam09, cam07 (3 cams, centered)
        canvas_width = max_width * 4
        canvas_height = target_height * 2
        canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)
        row2_offset_x = (canvas_width - 3 * max_width) // 2
        for cam_name, img in resized_images:
            h, w = img.shape[:2]
            if cam_name == 'cam06':
                x, y = 0, 0
                canvas[y:y+h, x:x+w] = img
            elif cam_name == 'cam03':
                x, y = max_width, 0
                canvas[y:y+h, x:x+w] = img
            elif cam_name == 'cam01':
                x, y = max_width * 2, 0
                canvas[y:y+h, x:x+w] = img
            elif cam_name == 'cam08':
                x, y = max_width * 3, 0
                canvas[y:y+h, x:x+w] = img
            elif cam_name == 'cam05':
                x, y = row2_offset_x, target_height
                canvas[y:y+h, x:x+w] = img
            elif cam_name == 'cam09':
                x, y = row2_offset_x + max_width, target_height
                canvas[y:y+h, x:x+w] = img
            elif cam_name == 'cam07':
                x, y = row2_offset_x + max_width * 2, target_height
                canvas[y:y+h, x:x+w] = img
    else:
        # Create grid layout for other datasets
        if source_lower == 'truckscenes':
            cols = 2
        else:
            cols = 3
        
        rows = (len(resized_images) + cols - 1) // cols
        
        # Create canvas
        canvas_width = cols * max_width
        canvas_height = rows * target_height
        canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)
        
        # Place images on canvas, center-aligning each row
        for row_idx in range(rows):
            row_images = resized_images[row_idx * cols : (row_idx + 1) * cols]
            num_in_row = len(row_images)
            offset_x = (canvas_width - num_in_row * max_width) // 2
            for col_idx, (cam_name, img) in enumerate(row_images):
                h, w = img.shape[:2]
                x = offset_x + col_idx * max_width
                y = row_idx * target_height
                canvas[y:y+h, x:x+w] = img
    
    # Save
    cv2.imwrite(output_path, canvas)


def concatenate_camera_views(
    camera_views: Dict[str, str],
    output_path: str,
    source: str = None,
    add_labels: bool = True
) -> None:
    """
    Concatenate multiple camera images in a grid layout with optional labels.
    Uses PIL for clean image composition (alternative to visualize_multiview without annotations).
    
    Args:
        camera_views: Dictionary mapping camera names to image paths
        output_path: Path to save the concatenated image
        source: Data source name to determine camera ordering and grid layout
        add_labels: Whether to add camera name labels (default: True)
    """
    if not camera_views:
        # Create a placeholder image if no images are available
        placeholder = Image.new('RGB', (800, 600), color=(128, 128, 128))
        placeholder.save(output_path)
        return
    
    # Determine camera order based on source
    source_key = source.lower() if source else None
    camera_order = CAMERA_ORDER.get(source_key, [])
    
    # Sort cameras according to predefined order
    ordered_cameras = []
    for cam_name in camera_order:
        if cam_name in camera_views:
            ordered_cameras.append((cam_name, camera_views[cam_name]))
    
    # Add any remaining cameras not in the predefined order
    for cam_name, cam_path in camera_views.items():
        if cam_name not in camera_order:
            ordered_cameras.append((cam_name, cam_path))
    
    # Load images
    images_with_names = []
    for cam_name, cam_path in ordered_cameras:
        try:
            if isinstance(cam_path, dict):
                cam_path = cam_path.get("path", str(cam_path))
                
            if cam_path and os.path.exists(cam_path):
                img = Image.open(cam_path)
                images_with_names.append((cam_name, img))
        except Exception as e:
            print(f"Warning: Failed to load image {cam_path}: {e}")
    
    if not images_with_names:
        placeholder = Image.new('RGB', (800, 600), color=(128, 128, 128))
        placeholder.save(output_path)
        return
    
    # Determine grid layout
    num_images = len(images_with_names)
    
    # Target size for each image (width, height) - defined before layout branches
    target_width = 640
    target_height = 480
    label_height = 30 if add_labels else 0  # Height for camera name label
    
    # Create labeled images if requested (needed for waymo/argoverse2/once custom layouts)
    labeled_images = []
    for cam_name, img in images_with_names:
        # Resize image to target size
        resized_img = img.resize((target_width, target_height), Image.LANCZOS)
        display_name = CAMERA_DISPLAY_NAMES.get(source_key, {}).get(cam_name, cam_name)

        if add_labels:
            # Create a new image with space for label
            labeled_img = Image.new('RGB', (target_width, target_height + label_height), color=(0, 0, 0))
            
            # Paste the resized image below the label area
            labeled_img.paste(resized_img, (0, label_height))
            
            # Draw camera name on top (use display name for ONCE etc.)
            draw = ImageDraw.Draw(labeled_img)
            try:
                # Try to use a nice font
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
            except Exception:
                # Fall back to default font
                font = ImageFont.load_default()
            
            # Draw background rectangle for text
            text_bbox = draw.textbbox((0, 0), display_name, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]
            
            padding = 5
            draw.rectangle(
                [padding, padding, text_width + 2*padding, text_height + 2*padding],
                fill=(255, 255, 255)  # White background for black text
            )
            
            # Draw text in black
            draw.text((padding, padding), display_name, fill=(0, 0, 0), font=font)
            
            labeled_images.append(labeled_img)
        else:
            labeled_images.append(resized_img)
    
    # Special layout for Waymo: 2 rows - Front cameras on top, side cameras on bottom
    if source_key == 'waymo':
        canvas_width = target_width * 3
        canvas_height = (target_height + label_height) * 2
        canvas = Image.new('RGB', (canvas_width, canvas_height), color=(0, 0, 0))
        cell_height = target_height + label_height
        for cam_name, labeled_img in zip([name for name, _ in images_with_names], labeled_images):
            if cam_name == 'FRONT_LEFT':
                canvas.paste(labeled_img, (0, 0))
            elif cam_name == 'FRONT':
                canvas.paste(labeled_img, (target_width, 0))
            elif cam_name == 'FRONT_RIGHT':
                canvas.paste(labeled_img, (target_width * 2, 0))
            elif cam_name == 'SIDE_LEFT':
                canvas.paste(labeled_img, (0, cell_height))
            elif cam_name == 'SIDE_RIGHT':
                canvas.paste(labeled_img, (target_width * 2, cell_height))
        canvas.save(output_path)
        return
    
    # Special layout for AV2 (Argoverse2): 3 rows, center camera center-aligned
    if source_key == 'argoverse2':
        # Row 1: ring_front_left | ring_front_center (center) | ring_front_right
        # Row 2: ring_side_left | ring_side_right (centered)
        # Row 3: ring_rear_left | ring_rear_right (centered)
        canvas_width = target_width * 3
        canvas_height = (target_height + label_height) * 3
        canvas = Image.new('RGB', (canvas_width, canvas_height), color=(0, 0, 0))
        cell_height = target_height + label_height
        row2_offset_x = (canvas_width - 2 * target_width) // 2
        for cam_name, labeled_img in zip([name for name, _ in images_with_names], labeled_images):
            if cam_name == 'ring_front_left':
                canvas.paste(labeled_img, (0, 0))
            elif cam_name == 'ring_front_center':
                canvas.paste(labeled_img, (target_width, 0))
            elif cam_name == 'ring_front_right':
                canvas.paste(labeled_img, (target_width * 2, 0))
            elif cam_name == 'ring_side_left':
                canvas.paste(labeled_img, (row2_offset_x, cell_height))
            elif cam_name == 'ring_side_right':
                canvas.paste(labeled_img, (row2_offset_x + target_width, cell_height))
            elif cam_name == 'ring_rear_left':
                canvas.paste(labeled_img, (row2_offset_x, cell_height * 2))
            elif cam_name == 'ring_rear_right':
                canvas.paste(labeled_img, (row2_offset_x + target_width, cell_height * 2))
        canvas.save(output_path)
        return
    
    # Special layout for ONCE: row1 = 4 cams, row2 = 3 cams (centered)
    if source_key == 'once':
        canvas_width = target_width * 4
        canvas_height = (target_height + label_height) * 2
        canvas = Image.new('RGB', (canvas_width, canvas_height), color=(0, 0, 0))
        cell_height = target_height + label_height
        row2_offset_x = (canvas_width - 3 * target_width) // 2
        for cam_name, labeled_img in zip([name for name, _ in images_with_names], labeled_images):
            if cam_name == 'cam06':
                canvas.paste(labeled_img, (0, 0))
            elif cam_name == 'cam03':
                canvas.paste(labeled_img, (target_width, 0))
            elif cam_name == 'cam01':
                canvas.paste(labeled_img, (target_width * 2, 0))
            elif cam_name == 'cam08':
                canvas.paste(labeled_img, (target_width * 3, 0))
            elif cam_name == 'cam05':
                canvas.paste(labeled_img, (row2_offset_x, cell_height))
            elif cam_name == 'cam09':
                canvas.paste(labeled_img, (row2_offset_x + target_width, cell_height))
            elif cam_name == 'cam07':
                canvas.paste(labeled_img, (row2_offset_x + target_width * 2, cell_height))
        canvas.save(output_path)
        return
    
    # Default grid layout
    if source_key == 'truckscenes':
        cols = 2
    else:
        cols = 3
    rows = (num_images + cols - 1) // cols
    
    canvas_width = cols * target_width
    canvas_height = rows * (target_height + label_height)
    canvas = Image.new('RGB', (canvas_width, canvas_height), color=(0, 0, 0))
    cell_height = target_height + label_height
    for row_idx in range(rows):
        row_imgs = labeled_images[row_idx * cols : (row_idx + 1) * cols]
        num_in_row = len(row_imgs)
        offset_x = (canvas_width - num_in_row * target_width) // 2
        for col_idx, img in enumerate(row_imgs):
            x = offset_x + col_idx * target_width
            y = row_idx * cell_height
            canvas.paste(img, (x, y))
    
    canvas.save(output_path)


def create_video_from_frames(
    frame_sequence: List[Dict[str, Any]],
    output_path: str,
    source: str,
    fps: int = 2,
    show_object_ids: bool = True,
    highlight_objects: Optional[List[Dict[str, Any]]] = None,
    show_only_highlighted: bool = False,
    hide_object_ids_for_highlighted: bool = False
) -> None:
    """
    Create a video (.mp4) from a sequence of frames with multi-view visualization.
    
    Args:
        frame_sequence: List of frame data dictionaries
        output_path: Path to save the video (.mp4)
        source: Data source name (e.g., 'nuscenes', 'waymo')
        fps: Frames per second for the output video (default: 2)
        show_object_ids: Whether to show object IDs and bounding boxes
        highlight_objects: Optional list of objects to highlight (same format as visualize_multiview)
        show_only_highlighted: If True, only show highlighted objects
        hide_object_ids_for_highlighted: If True, only show custom labels for highlighted objects
    
    Example:
        # Create video with highlighted objects
        create_video_from_frames(
            frame_sequence=frames,
            output_path="sequence.mp4",
            source="nuscenes",
            fps=2,
            show_object_ids=True,
            highlight_objects=[
                {'track_id': 'abc123', 'color': (255, 0, 0), 'label': '(1)'}
            ]
        )
    """
    import cv2
    import tempfile
    import shutil
    
    if not frame_sequence:
        logger.warning("Empty frame sequence provided")
        return
    
    # Create temporary directory for frame images
    temp_dir = tempfile.mkdtemp()
    temp_frames = []
    
    try:
        # Generate visualization for each frame
        for frame_idx, frame in enumerate(frame_sequence):
            temp_frame_path = os.path.join(temp_dir, f"frame_{frame_idx:04d}.jpg")
            
            # Update highlight objects with camera-specific info if needed
            frame_highlight_objects = None
            if highlight_objects:
                frame_highlight_objects = []
                track_3d = frame['annotation'].get('track_3d', [])
                
                for highlight_info in highlight_objects:
                    # Check if using track_id (need to find camera position)
                    if 'track_id' in highlight_info:
                        track_id = highlight_info['track_id']
                        if track_id in track_3d:
                            obj_idx = track_3d.index(track_id)
                            frame_highlight_objects.append({
                                'object_id': obj_idx + 1,
                                'color': highlight_info.get('color', (255, 0, 0)),
                                'label': highlight_info.get('label', '')
                            })
                    else:
                        # Pass through as-is (object_id or camera-based)
                        frame_highlight_objects.append(highlight_info)
            
            # Create multi-view visualization for this frame
            visualize_multiview(
                data=frame,
                output_path=temp_frame_path,
                source=source,
                show_object_ids=show_object_ids,
                highlight_objects=frame_highlight_objects,
                show_only_highlighted=show_only_highlighted,
                hide_object_ids_for_highlighted=hide_object_ids_for_highlighted
            )
            
            # Add frame counter overlay
            img = cv2.imread(temp_frame_path)
            if img is not None:
                cv2.imwrite(temp_frame_path, img)
                temp_frames.append(temp_frame_path)
        
        if not temp_frames:
            logger.warning("No frames generated for video")
            return
        
        # Get frame dimensions from first frame
        first_frame = cv2.imread(temp_frames[0])
        height, width = first_frame.shape[:2]
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Write all frames to video
        for frame_path in temp_frames:
            frame = cv2.imread(frame_path)
            if frame is not None:
                video_writer.write(frame)
        
        video_writer.release()
        logger.info(f"Created video with {len(temp_frames)} frames at {output_path}")
        
    except Exception as e:
        logger.error(f"Error creating video: {e}", exc_info=True)
    finally:
        # Clean up temporary directory
        try:
            shutil.rmtree(temp_dir)
        except Exception as e:
            logger.warning(f"Failed to clean up temp directory: {e}")


def save_frames_to_folder(
    frame_sequence: List[Dict[str, Any]],
    output_folder: str,
    source: str,
    show_object_ids: bool = True,
    highlight_objects: Optional[List[Dict[str, Any]]] = None,
    show_only_highlighted: bool = False,
    hide_object_ids_for_highlighted: bool = False,
    show_object_ids_only_on_frames: Optional[List[int]] = None,
) -> List[str]:
    """
    Save individual frames to a folder with frame IDs as filenames.

    Args:
        frame_sequence: List of frame data dictionaries
        output_folder: Path to folder where frames will be saved
        source: Data source name (e.g., 'nuscenes', 'waymo')
        show_object_ids: Whether to show object IDs and bounding boxes (used when show_object_ids_only_on_frames is None)
        highlight_objects: Optional list of objects to highlight
        show_only_highlighted: If True, only show highlighted objects
        hide_object_ids_for_highlighted: If True, only show custom labels for highlighted objects
        show_object_ids_only_on_frames: If set, only frames whose index is in this list get object IDs drawn.
            Other frames are rendered with show_object_ids=False. E.g. [0] = only first frame shows IDs.

    Returns:
        List of saved frame paths relative to output_folder
    """
    import os
    from pathlib import Path

    if not frame_sequence:
        logger.warning("Empty frame sequence provided")
        return []

    output_folder_path = Path(output_folder)
    output_folder_path.mkdir(parents=True, exist_ok=True)

    saved_paths = []

    try:
        for frame_idx, frame_data in enumerate(frame_sequence):
            record_id = frame_data.get('id')

            if record_id is not None:
                filename = f"frame_{frame_idx:04d}_record_{record_id}.jpg"
            else:
                filename = f"frame_{frame_idx:04d}.jpg"

            frame_path = output_folder_path / filename

            # Per-frame control: show object IDs only on specified frame indices
            if show_object_ids_only_on_frames is not None:
                show_ids_this_frame = frame_idx in show_object_ids_only_on_frames
            else:
                show_ids_this_frame = show_object_ids

            visualize_multiview(
                data=frame_data,
                output_path=str(frame_path),
                source=source,
                show_object_ids=show_ids_this_frame,
                highlight_objects=highlight_objects,
                show_only_highlighted=show_only_highlighted,
                hide_object_ids_for_highlighted=hide_object_ids_for_highlighted
            )

            saved_paths.append(filename)

        logger.info(f"Saved {len(saved_paths)} frames to {output_folder}")
        return saved_paths

    except Exception as e:
        logger.error(f"Error saving frames to folder: {e}", exc_info=True)
        return saved_paths


import logging
logger = logging.getLogger(__name__)
