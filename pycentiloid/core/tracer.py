"""
Core tracer definitions and parameters.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple
from ..config import Config

@dataclass
class TracerParams:
    """Parameters for specific PET tracers."""
    name: str
    half_life: float
    ref_region: str
    scaling_factor: float
    dynamic_frames: Tuple[int, int]
    thresholds: Dict[str, float]
    template: Path
    reference_mask: Path
    target_mask: Path

class TracerRegistry:
    """Registry of supported PET tracers."""
    
    def __init__(self):
        self.config = Config()
        self.tracers = {}
        self._initialize_tracers()
    
    def _initialize_tracers(self):
        """Initialize tracer parameters from config."""
        for tracer_id, params in self.config.TRACERS.items():
            self.tracers[tracer_id] = TracerParams(**params)
    
    def get_tracer(self, tracer_id: str) -> TracerParams:
        """Get tracer parameters by ID."""
        if tracer_id not in self.tracers:
            raise ValueError(f"Unsupported tracer: {tracer_id}")
        return self.tracers[tracer_id]