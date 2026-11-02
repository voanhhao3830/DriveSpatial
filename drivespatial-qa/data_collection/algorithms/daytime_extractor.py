import torch
from datetime import datetime
from collections import Counter
from typing import List, Dict, Any

from sources.data_collection.enum import DaytimeType
from sources.data_collection.models.data import nuScenesDataset
from sources.data_collection.enum.data_sources import DataSource
from sources.data_collection.models.data import TruckscenesDataset
from sources.data_collection.models.algorithms.constants import DAYTIME_PROMPT
from sources.data_collection.models.ai.semantic_extract import SemanticExtractor

class DaytimeExtractor:
    @staticmethod
    def extract_daytime(
        semantic_extractor: SemanticExtractor,
        sampled_frames: List[Dict[str, Any]],
    ) -> str:
        if sampled_frames[0]['metadata']['source'] == DataSource.NUSCENES.value:
            sample = nuScenesDataset.NUSCE.get('scene', sampled_frames[0]['metadata']['scene_token'])
            log = nuScenesDataset.NUSCE.get('log', sample['log_token'])
            s = log['logfile']
            ts = s.split('-', 1)[1]      # remove vehicle id
            dt = datetime.strptime(ts, "%Y-%m-%d-%H-%M-%S%z")
            if dt.hour < 12 and dt.hour >= 5:
                return DaytimeType.MORNING.value
            elif dt.hour < 14 and dt.hour >= 12:
                return DaytimeType.NOON.value
            elif dt.hour < 18 and dt.hour >= 14:
                return DaytimeType.AFTERNOON.value
            else:
                return DaytimeType.NIGHT.value
        elif sampled_frames[0]['metadata']['source'] == DataSource.ONCE.value:
            periods = [item['metadata']['period'] for item in sampled_frames]
            period_counter = Counter(periods)
            most_common_period = period_counter.most_common(1)[0][0]
            return most_common_period
        elif sampled_frames[0]['metadata']['source'] == DataSource.WAYMO.value:
            periods = [item['metadata']['time_of_day'] for item in sampled_frames]
            period_counter = Counter(periods)
            most_common_period = period_counter.most_common(1)[0][0]
            return most_common_period     
        elif sampled_frames[0]['metadata']['source'] == DataSource.TRUCKSCENES.value:
            daytime = []
            for frame in sampled_frames:
                sample = TruckscenesDataset.TRUCKSCE.get('sample', frame['metadata']['frame_token'])
                scene = TruckscenesDataset.TRUCKSCE.get('scene', sample['scene_token'])
                desc = scene['description']
                daytime.append(desc.split(';')[2].split('.')[-1].lower())
            daytime_counter = Counter(daytime)
            most_common_daytime = daytime_counter.most_common(1)[0][0]
            return most_common_daytime
        elif sampled_frames[0]['metadata']['source'] == DataSource.ARGOVERSE2.value:
            queries = [DAYTIME_PROMPT.format(daytime=daytime.value) for daytime in DaytimeType]
            list_daytimes = [daytime.value for daytime in DaytimeType]
            text_features = semantic_extractor(text=queries)
            predicted_daytimes = []
            for frame in sampled_frames:
                for _, image_path in frame.get('views', {}).items():
                    image_features = semantic_extractor(image=image_path)
                    cosine_similarity = SemanticExtractor.cosine_similarity(text_features, image_features)
                    predicted_daytimes.append(list_daytimes[cosine_similarity.argmax().item()])
            daytime_counter = Counter(predicted_daytimes)
            most_common_daytime = daytime_counter.most_common(1)[0][0]
            return most_common_daytime
