import os
import json
import random
from pathlib import Path
from typing import Dict, Any, List, Tuple
from PIL import Image, ImageDraw, ImageFont
from uuid import uuid4
from itertools import permutations

from sources.question_generation.enum import ConstructionQuestionType
from sources.db.sql import SQLiteController
from sources.data_collection.enum.data_sources import DataSource


class CameraOrderingGenerator:
    QUESTION_TYPE = ConstructionQuestionType.CAMERA_ORDERING
    QUESTION_TEMPLATE = """
Given a shuffled set of camera views arranged around an autonomous vehicle, determine the correct spatial ordering of the cameras.
"""
    
    # Camera ordering for different datasets (clockwise starting from Front)
    CAMERA_ORDER = {
        'nuscenes': [
            'CAM_FRONT',           # Front (starting point)
            'CAM_FRONT_RIGHT',     # Clockwise
            'CAM_BACK_RIGHT',      # Continue clockwise
            'CAM_BACK',            # Back
            'CAM_BACK_LEFT',       # Continue clockwise
            'CAM_FRONT_LEFT'       # Complete circle back to front
        ],
        'once': [
            'cam01',  # Front (starting point)
            'cam07',  # Back-Right
            'cam09',  # Back
            'cam05',  # Back-Left
            'cam06',  # Front-Left
            'cam08',  # Front-Right
        ],
        'waymo': [
            'FRONT',         # Front (starting point)
            'FRONT_RIGHT',   # Clockwise
            'SIDE_RIGHT',    # Continue clockwise
            'SIDE_LEFT',     # Back side
            'FRONT_LEFT'     # Complete circle
        ],
        'truckscenes': [
            'CAMERA_RIGHT_FRONT',  # Front-Right (starting point)
            'CAMERA_RIGHT_BACK',   # Clockwise
            'CAMERA_LEFT_BACK',    # Continue
            'CAMERA_LEFT_FRONT'    # Complete circle
        ],
        'argoverse2': [
            'ring_front_center',  # Front (starting point)
            'ring_front_right',   # Clockwise
            'ring_side_right',    # Continue clockwise
            'ring_rear_right',    # Back-Right
            'ring_rear_left',     # Back-Left
            'ring_side_left',     # Continue
            'ring_front_left'     # Complete circle
        ]
    }
    
    # Camera descriptions for better understanding (updated for clockwise order)
    CAMERA_DESCRIPTIONS = {
        'nuscenes': {
            'CAM_FRONT': 'Front',
            'CAM_FRONT_RIGHT': 'Front-Right',
            'CAM_BACK_RIGHT': 'Back-Right',
            'CAM_BACK': 'Back',
            'CAM_BACK_LEFT': 'Back-Left',
            'CAM_FRONT_LEFT': 'Front-Left'
        },
        'once': {
            'cam01': 'Front',
            'cam07': 'Back-Right',
            'cam09': 'Back',
            'cam05': 'Back-Left',
            'cam06': 'Front-Left',
            'cam08': 'Front-Right',
        },
        'waymo': {
            'FRONT': 'Front',
            'FRONT_RIGHT': 'Front-Right',
            'SIDE_RIGHT': 'Right-Side',
            'SIDE_LEFT': 'Left-Side',
            'FRONT_LEFT': 'Front-Left'
        },
        'truckscenes': {
            'CAMERA_RIGHT_FRONT': 'Front-Right',
            'CAMERA_RIGHT_BACK': 'Back-Right',
            'CAMERA_LEFT_BACK': 'Back-Left',
            'CAMERA_LEFT_FRONT': 'Front-Left'
        },
        'argoverse2': {
            'ring_front_center': 'Front',
            'ring_front_right': 'Front-Right',
            'ring_side_right': 'Right-Side',
            'ring_rear_right': 'Back-Right',
            'ring_rear_left': 'Back-Left',
            'ring_side_left': 'Left-Side',
            'ring_front_left': 'Front-Left'
        }
    }
    
    @staticmethod
    def generate_question(
        sampled_data: Dict[str, Any],
        db_path: str,
        save_path: str,
        num_options: int = 4
    ) -> Dict[str, Any]:
        """
        Generate a camera ordering question with shuffled camera views.
        
        Args:
            sampled_data: Sampled data record from database
            db_path: Path to the database
            save_path: Path to save questions and images
            num_options: Number of options to generate (default: 4)
            
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
            question_folder = Path(save_path) / "construct" / CameraOrderingGenerator.QUESTION_TYPE.value
            images_folder = question_folder / "images"
            images_folder.mkdir(parents=True, exist_ok=True)
            
            # Generate unique ID for this question
            question_id = str(uuid4())
            
            # Step 1: Get camera views in correct order
            source_key = source.lower() if source else None
            camera_order = CameraOrderingGenerator.CAMERA_ORDER.get(source_key, [])
            camera_descriptions = CameraOrderingGenerator.CAMERA_DESCRIPTIONS.get(source_key, {})
            
            if not camera_order:
                raise Exception(f"No camera order defined for source: {source}")
            
            views = sampled_data.get("views", {})
            
            # Get available cameras in correct order
            available_cameras = []
            for cam_name in camera_order:
                if cam_name in views:
                    cam_path = views[cam_name]
                    if isinstance(cam_path, dict):
                        cam_path = cam_path.get("path")
                    if cam_path and os.path.exists(cam_path):
                        available_cameras.append((cam_name, cam_path))
            
            if len(available_cameras) < 3:
                raise Exception(f"Not enough cameras available: {len(available_cameras)}")
            
            # Step 2: Create shuffled camera image with descriptive labels
            shuffled_cameras, shuffled_labels = CameraOrderingGenerator._shuffle_cameras(
                available_cameras,
                camera_descriptions
            )
            
            shuffled_image_path = images_folder / f"{question_id}_shuffled_cameras.jpg"
            CameraOrderingGenerator._create_camera_grid(
                shuffled_cameras,
                str(shuffled_image_path),
                show_letter_labels=True,
                source=source
            )
            
            # Step 3: Generate multiple choice options with descriptive labels
            correct_sequence_labels = [camera_descriptions.get(cam[0]) for cam in available_cameras if cam[0] in camera_descriptions]
            options, correct_answer = CameraOrderingGenerator._generate_ordering_options(
                correct_sequence_labels,
                shuffled_labels,
                num_options
            )
            
            # Step 4: Format question
            question_text = CameraOrderingGenerator.QUESTION_TEMPLATE.strip()
            question_text += f"\n\nThe shuffled camera views are labeled A, B, C, D, etc. in the image above."
            question_text += f"\n\nIdentify the correct clockwise ordering starting from the Front camera:"
            question_text += f"\n\nOptions:\n"
            for idx, ordering in enumerate(options):
                question_text += f"({chr(65+idx)}) {ordering}\n"
            
            # Step 5: Create question dictionary
            question_dict = {
                "question_id": question_id,
                "question_type": CameraOrderingGenerator.QUESTION_TYPE.value,
                "question": question_text,
                "images": str(shuffled_image_path),
                "options": options,
                "correct_answer": correct_answer,
                "metadata": {
                    "scene_id": scene_id,
                    "frame_id": frame_id,
                    "source": source,
                    "num_cameras": len(available_cameras),
                    "correct_order": correct_sequence_labels,
                    "shuffled_order": shuffled_labels,
                    "record_id": sampled_data.get("id")
                }
            }
            
            return question_dict
            
        except Exception as e:
            raise Exception(f"Failed to generate camera ordering question: {e}")
    
    @staticmethod
    def _shuffle_cameras(
        cameras: List[Tuple[str, str]],
        camera_descriptions: Dict[str, str]
    ) -> Tuple[List[Tuple[str, str, str]], List[str]]:
        """
        Shuffle camera list and return both shuffled list with labels.
        
        Args:
            cameras: List of (camera_name, camera_path) tuples in correct order
            camera_descriptions: Dictionary mapping camera names to descriptions
            
        Returns:
            Tuple of (shuffled cameras with labels, shuffled descriptive labels)
        """
        # Create cameras with descriptive labels
        cameras_with_labels = [
            (cam_name, cam_path, camera_descriptions.get(cam_name, cam_name))
            for cam_name, cam_path in cameras
        ]
        
        indices = list(range(len(cameras_with_labels)))
        random.shuffle(indices)
        
        shuffled_cameras = [cameras_with_labels[i] for i in indices]
        
        # Get the descriptive labels in shuffled order
        shuffled_labels = [cameras_with_labels[i][2] for i in indices]
        
        return shuffled_cameras, shuffled_labels
    
    @staticmethod
    def _generate_ordering_options(
        correct_labels: List[str],
        shuffled_labels: List[str],
        num_options: int = 4
    ) -> Tuple[List[str], int]:
        """
        Generate multiple choice options for camera ordering using descriptive labels.
        
        Args:
            correct_labels: Correct order with descriptive labels
            shuffled_labels: Current shuffled order with descriptive labels
            num_options: Number of options
            
        Returns:
            Tuple of (options list, correct answer index)
        """
        # The correct answer maps from letter labels (A, B, C...) to correct spatial order
        # If shuffled_labels = ["Front-Right", "Front", "Front-Left"]
        # and correct = ["Front", "Front-Left", "Front-Right"]
        # Then correct answer is: B → C → A
        
        letter_to_shuffled = {chr(65 + i): label for i, label in enumerate(shuffled_labels)}
        
        # Find the letter sequence that produces the correct order
        correct_letter_sequence = []
        for correct_label in correct_labels:
            for letter, shuffled_label in letter_to_shuffled.items():
                if shuffled_label == correct_label:
                    correct_letter_sequence.append(letter)
                    break
        
        correct_option = " → ".join(correct_letter_sequence)
        
        # Generate distractor options
        all_options = [correct_option]
        
        letters = [chr(65 + i) for i in range(len(shuffled_labels))]
        max_attempts = 100
        attempts = 0
        
        while len(all_options) < num_options and attempts < max_attempts:
            # Create a random permutation
            distractor = letters.copy()
            random.shuffle(distractor)
            
            distractor_str = " → ".join(distractor)
            
            if distractor_str not in all_options:
                all_options.append(distractor_str)
            
            attempts += 1
        
        # Shuffle options
        correct_answer = 0
        shuffled_indices = list(range(len(all_options)))
        random.shuffle(shuffled_indices)
        shuffled_options = [all_options[i] for i in shuffled_indices]
        correct_answer = shuffled_indices.index(0)
        
        return shuffled_options, correct_answer
    
    @staticmethod
    def _create_camera_grid(
        cameras: List[Tuple[str, str, str]],
        output_path: str,
        show_letter_labels: bool = True,
        source: str = None
    ) -> None:
        """
        Create a grid of camera images with labels.
        
        Args:
            cameras: List of (camera_name, camera_path, descriptive_label) tuples
            output_path: Path to save the grid image
            show_letter_labels: Whether to show letter labels (A, B, C, etc.)
            source: Data source name
        """
        if not cameras:
            placeholder = Image.new('RGB', (800, 600), color=(128, 128, 128))
            placeholder.save(output_path)
            return
        
        # Load images
        images_with_info = []
        for idx, (cam_name, cam_path, desc_label) in enumerate(cameras):
            try:
                img = Image.open(cam_path)
                # Use letter label (A, B, C) for the shuffled display
                label = chr(65 + idx) if show_letter_labels else desc_label
                images_with_info.append((img, label))
            except Exception as e:
                print(f"Warning: Failed to load image {cam_path}: {e}")
        
        if not images_with_info:
            placeholder = Image.new('RGB', (800, 600), color=(128, 128, 128))
            placeholder.save(output_path)
            return
        
        # Determine grid layout
        num_images = len(images_with_info)
        source_lower = source.lower() if source else None
        
        if source_lower == 'truckscenes':
            cols = 2  # 2x2 layout
        else:
            cols = 3  # Default: 3 columns
        
        rows = (num_images + cols - 1) // cols
        
        # Target size for each image
        target_width = 640
        target_height = 480
        label_height = 40
        
        # Create labeled images
        labeled_images = []
        for img, label in images_with_info:
            # Resize image
            resized_img = img.resize((target_width, target_height), Image.LANCZOS)
            
            # Create image with label space
            labeled_img = Image.new('RGB', (target_width, target_height + label_height), color=(0, 0, 0))
            labeled_img.paste(resized_img, (0, label_height))
            
            # Draw label
            draw = ImageDraw.Draw(labeled_img)
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24)
            except:
                font = ImageFont.load_default()
            
            # Draw background for text
            text_bbox = draw.textbbox((0, 0), label, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]
            
            padding = 8
            draw.rectangle(
                [padding, padding, text_width + 2*padding, text_height + 2*padding],
                fill=(0, 0, 0, 180)
            )
            
            # Draw text
            draw.text((padding, padding), label, fill=(255, 255, 255), font=font)
            
            labeled_images.append(labeled_img)
        
        # Create canvas
        canvas_width = cols * target_width
        canvas_height = rows * (target_height + label_height)
        canvas = Image.new('RGB', (canvas_width, canvas_height), color=(0, 0, 0))
        
        # Paste images in grid
        for idx, img in enumerate(labeled_images):
            row = idx // cols
            col = idx % cols
            x = col * target_width
            y = row * (target_height + label_height)
            canvas.paste(img, (x, y))
        
        canvas.save(output_path)


if __name__ == "__main__":
    # Test the generator
    from sources.utils import load_config
    from sources.db.sql import SQLiteController
    
    config = load_config()
    sqlite_db = SQLiteController(db_path=config["app"]["db"]["sqlite_path"])
    
    # Get a sample record
    data_records = sqlite_db.get_all_av_data(limit=10)
    
    if data_records:
        sample_data = data_records[0]
        
        question = CameraOrderingGenerator.generate_question(
            sampled_data=sample_data,
            db_path=config["app"]["db"]["sqlite_path"],
            save_path=config["app"]["questions_config"]["save_path"]
        )
        
        print("Generated question:")
        print(json.dumps(question, indent=2))
