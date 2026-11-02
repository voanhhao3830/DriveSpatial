import random
from typing import Dict, List, Any, Iterator
from collections import defaultdict


class ConstructSampler:
    
    def __init__(self, data: List[Dict[str, Any]], configs: Dict[str, Any]):
        self.data = data
        self.total_question_per_dataset = configs["app"]["questions_config"]["question_per_type"]
        self._group_data()
    
    def _group_data(self) -> None:
        """Group data records by source, then by scene_type."""
        # First group by source
        self.source_groups = defaultdict(lambda: defaultdict(list))
        
        for record in self.data:
            metadata = record.get("metadata", {})
            source = metadata.get("source", "unknown")
            scene_type = metadata.get("scene_type", "unknown")
            
            self.source_groups[source][scene_type].append(record)
    
    def __iter__(self) -> Iterator[Dict[str, Any]]:
        """
        Iterate through sampled records.
        
        For each source, samples total_question_per_dataset records
        equally distributed across scene types.
        
        Yields:
            Dict[str, Any]: Data record containing all information including id
        """
        # Process each source separately
        for source, scene_type_groups in self.source_groups.items():
            num_scene_types = len(scene_type_groups)
            
            if num_scene_types == 0:
                continue
            
            # Calculate samples per scene type for this source
            samples_per_type = self.total_question_per_dataset // num_scene_types
            remaining = self.total_question_per_dataset % num_scene_types
            
            # Sample equally from each scene type within this source
            for idx, (scene_type, records) in enumerate(scene_type_groups.items()):
                # Add extra sample to first few types if there's a remainder
                num_samples = samples_per_type + (1 if idx < remaining else 0)
                
                # Sample from this scene type
                if len(records) <= num_samples:
                    # If not enough records, use all
                    samples = records
                else:
                    # Randomly sample without replacement
                    samples = random.sample(records, num_samples)
                
                # Yield each sampled record
                for record in samples:
                    yield record
    
    def __len__(self) -> int:
        """Return the total number of samples that will be yielded."""
        total = 0
        for source, scene_type_groups in self.source_groups.items():
            # Each source contributes total_question_per_dataset samples
            source_total = sum(len(records) for records in scene_type_groups.values())
            total += min(self.total_question_per_dataset, source_total)
        return total
