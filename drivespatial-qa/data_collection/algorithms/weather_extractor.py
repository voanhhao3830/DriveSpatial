import torch
from typing import List, Dict, Any
from collections import Counter

from sources.data_collection.enum import WeatherType
from sources.data_collection.enum.data_sources import DataSource
from sources.data_collection.models.data import TruckscenesDataset
from sources.data_collection.models.algorithms.constants import WEATHER_PROMPT
from sources.data_collection.models.ai.semantic_extract import SemanticExtractor

class WeatherExtractor:
    @staticmethod
    def extract_weather(
        semantic_extractor: SemanticExtractor, 
        sampled_frames: List[Dict[str, Any]],
    ) -> str:
        if sampled_frames[0]['metadata']['source'] in [DataSource.NUSCENES.value, DataSource.ARGOVERSE2.value]:
            queries = [WEATHER_PROMPT.format(weather=weather.value) for weather in WeatherType]
            list_weathers = [weather.value for weather in WeatherType]
            
            text_features = semantic_extractor(text=queries)
            
            predicted_weathers = []
            for frame in sampled_frames:
                for _, image_path in frame.get('views', {}).items():
                    image_features = semantic_extractor(image=image_path)
                    cosine_similarity = SemanticExtractor.cosine_similarity(text_features, image_features)
                    predicted_weathers.append(list_weathers[cosine_similarity.argmax().item()])
                
            # calcuate average over all dimensions    
            weather_counter = Counter(predicted_weathers)
            most_common_weather = weather_counter.most_common(1)[0][0]
            return most_common_weather
        elif sampled_frames[0]['metadata']['source'] in [DataSource.ONCE.value, DataSource.WAYMO.value]:
            weathers = [item['metadata']['weather'] for item in sampled_frames]
            weather_counter = Counter(weathers)
            most_common_weather = weather_counter.most_common(1)[0][0]
            return most_common_weather
        elif sampled_frames[0]['metadata']['source'] == DataSource.TRUCKSCENES.value:
            weather = []
            for frame in sampled_frames:
                sample = TruckscenesDataset.TRUCKSCE.get('sample', frame['metadata']['frame_token'])
                scene = TruckscenesDataset.TRUCKSCE.get('scene', sample['scene_token'])
                desc = scene['description']
                weather.append(desc.split(';')[0].split('.')[-1].lower())
            weather_counter = Counter(weather)
            most_common_weather = weather_counter.most_common(1)[0][0]
            return most_common_weather
