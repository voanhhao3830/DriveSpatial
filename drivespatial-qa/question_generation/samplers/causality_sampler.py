import random
import logging
from typing import Dict, List, Any, Iterator
from collections import defaultdict

logger = logging.getLogger(__name__)


class CausalityReasoningSampler:
    def __init__(
        self, 
        data: List[Dict[str, Any]], 
        configs: Dict[str, Any],
        video_mode: bool = False,
        sequence_length: int = 5
    ):
        self.data = data
        self.total_question_per_dataset = configs["app"]["questions_config"]["question_per_type"]
        self.video_mode = video_mode
        self.sequence_length = sequence_length
        self._group_data()
    
    def _group_data(self) -> None:
        """Group data records by source, scene, then by weather and daytime combination."""
        # First group by source and scene for video sequences
        self.source_scene_groups = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        
        for record in self.data:
            metadata = record.get("metadata", {})
            source = metadata.get("source", "unknown")
            scene_id = record.get("scene_id", "unknown")
            weather = metadata.get("weather", "unknown")
            daytime = metadata.get("daytime", "unknown")
            if source == "once":
                continue
            # Create a key combining weather and daytime
            condition_key = f"{weather}_{daytime}"
            
            self.source_scene_groups[source][scene_id][condition_key].append(record)
        
        # Sort frames within each scene by record id for video sequences
        if self.video_mode:
            for source in self.source_scene_groups:
                for scene_id in self.source_scene_groups[source]:
                    for condition_key in self.source_scene_groups[source][scene_id]:
                        # Sort by record id to maintain temporal order
                        self.source_scene_groups[source][scene_id][condition_key].sort(
                            key=lambda x: x.get("id", 0)
                        )
    
    def _sample_video_sequence(
        self, 
        records: List[Dict[str, Any]], 
        target_length: int
    ) -> List[Dict[str, Any]]:
        """
        Sample a consecutive sequence of frames from a scene.
        
        Args:
            records: Sorted list of frames from a scene
            target_length: Desired sequence length
            
        Returns:
            List of consecutive frames (may be shorter if not enough frames available)
        """
        if len(records) < target_length:
            return records
        
        # Randomly select a starting point
        max_start_idx = len(records) - target_length
        start_idx = random.randint(0, max_start_idx)
        
        return records[start_idx:start_idx + target_length]
    
    def __iter__(self) -> Iterator[Any]:
        """
        Iterate through sampled records or video sequences.
        
        For each source, samples total_question_per_dataset items
        equally distributed across weather-daytime combinations.
        
        Yields:
            If video_mode=False: Dict[str, Any] - Single data record
            If video_mode=True: List[Dict[str, Any]] - Video sequence of frames
        """
        # Process each source separately
        for source in self.source_scene_groups:
            # Flatten scene-condition structure for sampling
            all_samples = []
            
            for scene_id in self.source_scene_groups[source]:
                for condition_key, records in self.source_scene_groups[source][scene_id].items():
                    if self.video_mode:
                        # Sample video sequence from this scene
                        if len(records) >= 2:  # Need at least 2 frames for a sequence
                            video_sequence = self._sample_video_sequence(records, self.sequence_length)
                            if len(video_sequence) >= 2:  # Only add if we got a valid sequence
                                all_samples.append({
                                    'sequence': video_sequence,
                                    'condition_key': condition_key,
                                    'scene_id': scene_id
                                })
                    else:
                        # Sample single frames
                        for record in records:
                            all_samples.append({
                                'record': record,
                                'condition_key': condition_key
                            })
            
            # Sample from all available samples for this source
            if len(all_samples) <= self.total_question_per_dataset:
                samples = all_samples
            else:
                samples = random.sample(all_samples, self.total_question_per_dataset)
            
            # Yield samples
            for sample in samples:
                if self.video_mode:
                    yield sample['sequence']
                else:
                    yield sample['record']
    
    def __len__(self) -> int:
        """Return the total number of samples that will be yielded."""
        total = 0
        for source in self.source_scene_groups:
            source_total = 0
            for scene_id in self.source_scene_groups[source]:
                for records in self.source_scene_groups[source][scene_id].values():
                    if self.video_mode:
                        if len(records) >= 2:
                            source_total += 1
                    else:
                        source_total += len(records)
            total += min(self.total_question_per_dataset, source_total)
        return total
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the data distribution.
        
        Returns:
            Dictionary with distribution statistics
        """
        stats = {}
        for source in self.source_scene_groups:
            scenes_count = len(self.source_scene_groups[source])
            total_frames = 0
            total_sequences = 0
            
            for scene_id in self.source_scene_groups[source]:
                for records in self.source_scene_groups[source][scene_id].values():
                    total_frames += len(records)
                    if len(records) >= 2:
                        total_sequences += 1
            
            stats[source] = {
                "total_scenes": scenes_count,
                "total_frames": total_frames,
                "total_sequences": total_sequences,
                "mode": "video" if self.video_mode else "frame"
            }
        return stats