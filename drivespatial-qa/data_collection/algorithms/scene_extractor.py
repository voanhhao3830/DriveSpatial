import io
import os
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from tqdm import tqdm
from uuid import uuid4
from collections import Counter
from typing import List, Dict, Any

from sources.data_collection.enum import SceneType
from sources.data_collection.enum.data_sources import DataSource
from sources.data_collection.models.algorithms.constants import SCENE_PROMPT
from sources.data_collection.models.ai import SemanticExtractor, VLMExtractor
from sources.data_collection.models.data import (
    nuScenesDataset, 
    Argoverse2Dataset, 
    ONCEDataset, 
    WaymoDataset,
    TruckscenesDataset
)

class SceneExtractor:
    @staticmethod
    def extract_scene(
        extractor:  VLMExtractor, 
        sampled_frames: List[Dict[str, Any]],
        limit: int = 50, # limit range of the lidar in meters
    ) -> str:
        tmp_path = f"tmp_{uuid4()}.png"
        predicted_scenes = []
        for idx, frame in tqdm(enumerate(sampled_frames)):
            if frame['metadata']['source'] == DataSource.NUSCENES.value:
                sample = nuScenesDataset.NUSCE.get('sample', frame['metadata']['frame_token'])
                nuScenesDataset.NUSCE.render_sample_data(sample['data']['LIDAR_TOP'], with_anns=True, out_path=tmp_path)
            elif frame['metadata']['source'] == DataSource.ARGOVERSE2.value:
                Argoverse2Dataset.render_topdown_map_with_hd(
                    log_id=frame['metadata']['scene_token'],
                    lidar_timestamp_ns=frame['metadata']['frame_token'],
                    out_path=tmp_path
                )
            elif frame['metadata']['source'] == DataSource.ONCE.value:
                ONCEDataset.render_topdown_map(
                    lidar_path=frame['lidar']['path'],
                    ego_pose=None,
                    bbox_3d=frame['annotation']['bbox_3d'],
                    label_3d=frame['annotation']['label_3d'],
                    out_path=tmp_path
                )
            elif frame['metadata']['source'] == DataSource.WAYMO.value:
                WaymoDataset.render_topdown_map(
                    lidar_path=frame['lidar']['path'],
                    ego_pose=None,
                    bboxes_3d=frame['annotation']['bbox_3d'],
                    labels_3d=frame['annotation']['label_3d'],
                    out_path=tmp_path
                )
            elif frame['metadata']['source'] == DataSource.TRUCKSCENES.value:
                sample = TruckscenesDataset.TRUCKSCE.get('sample', frame['metadata']['frame_token'])
                TruckscenesDataset.visualize_fused_lidar(
                    truckscenes_instance=TruckscenesDataset.TRUCKSCE,
                    sample_token=sample['token'],
                    out_path=tmp_path
                )
            scene_type = extractor(image=[tmp_path, frame['CAM_FRONT']['path']])
            predicted_scenes.append(scene_type)
            try:
                os.remove(tmp_path)
            except:
                print(f"Error removing file {tmp_path}")

        # Vote for the most common scene
        scene_counter = Counter(predicted_scenes)
        most_common_scene = scene_counter.most_common(1)[0][0]
        return most_common_scene