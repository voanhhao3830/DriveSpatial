import random
from typing import Dict, List, Any, Iterator
from collections import defaultdict


class RelationalSampler:
    """
    Sampler that yields data records per source with equal distribution across 
    weather conditions and time of day.
    
    For each source (e.g., "once", "nuScenes"), samples total_question_per_dataset records
    with equal distribution across different weather and daytime combinations.
    
    Args:
        data: List of data records from database
        configs: Configuration dictionary
    """
    
    def __init__(self, data: List[Dict[str, Any]], configs: Dict[str, Any]):
        self.data = data
        self.total_question_per_dataset = configs["app"]["questions_config"]["question_per_type"]
        self._group_data()
    
    def _group_data(self) -> None:
        """Group data records by source, then by weather and daytime combination."""
        # First group by source, then by weather-daytime combination
        self.source_groups = defaultdict(lambda: defaultdict(list))
        
        for record in self.data:
            metadata = record.get("metadata", {})
            source = metadata.get("source", "unknown")
            weather = metadata.get("weather", "unknown")
            daytime = metadata.get("daytime", "unknown")
            
            # Create a key combining weather and daytime
            condition_key = f"{weather}_{daytime}"
            
            self.source_groups[source][condition_key].append(record)
    
    def __iter__(self) -> Iterator[Dict[str, Any]]:
        """
        Iterate through sampled records.
        
        For each source, samples total_question_per_dataset records
        equally distributed across weather-daytime combinations.
        
        Yields:
            Dict[str, Any]: Data record containing all information including id
        """
        # Process each source separately
        for source, condition_groups in self.source_groups.items():
            num_conditions = len(condition_groups)
            
            if num_conditions == 0:
                continue
            
            # Calculate samples per condition for this source
            samples_per_condition = self.total_question_per_dataset // num_conditions
            remaining = self.total_question_per_dataset % num_conditions
            
            # Sample equally from each condition within this source
            for idx, (condition_key, records) in enumerate(condition_groups.items()):
                # Add extra sample to first few conditions if there's a remainder
                num_samples = samples_per_condition + (1 if idx < remaining else 0)
                
                # Sample from this condition
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
        for source, condition_groups in self.source_groups.items():
            # Each source contributes total_question_per_dataset samples
            source_total = sum(len(records) for records in condition_groups.values())
            total += min(self.total_question_per_dataset, source_total)
        return total
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the data distribution.
        
        Returns:
            Dictionary with distribution statistics
        """
        stats = {}
        for source, condition_groups in self.source_groups.items():
            source_stats = {}
            for condition_key, records in condition_groups.items():
                weather, daytime = condition_key.split('_', 1)
                source_stats[condition_key] = {
                    "weather": weather,
                    "daytime": daytime,
                    "count": len(records)
                }
            stats[source] = {
                "conditions": source_stats,
                "total_records": sum(len(records) for records in condition_groups.values()),
                "num_conditions": len(condition_groups)
            }
        return stats