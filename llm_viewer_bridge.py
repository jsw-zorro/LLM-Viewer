"""Bridge between llm_viewer's modeling and our performance analysis framework."""

import os
import sys
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import math

# Add parent directory to path to import from token_analysis
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from token_analysis.llm_performance_models import (
    LLMPerformanceModel, HardwareSpecs, ModelSpecs, PerformanceEstimate
)

from roofline_model import roofline_analyze
from model_analyzer import ModelAnalyzer

@dataclass
class LayerResults:
    """Results for a single layer"""
    ops: float
    memory_access: float
    arithmetic_intensity: float
    performance: float
    bound: str
    load_weight: float
    load_act: float
    store_act: float
    load_kv_cache: float
    store_kv_cache: float
    inference_time: float

class LLMViewerBridge(LLMPerformanceModel):
    """Adapter for llm_viewer's modeling approach to our framework"""
    
    def __init__(self, hardware: HardwareSpecs, model: ModelSpecs):
        super().__init__(hardware, model)
        # Create llm_viewer analyzer
        self.analyzer = ModelAnalyzer(
            model_id=model.name,
            hardware=hardware.name,
            source="huggingface"
        )
        
    def _aggregate_results(self, results: Dict[str, Dict[str, Any]], stage: str) -> Tuple[float, float, float, str]:
        """Aggregate layer-wise results into total metrics"""
        total_ops = 0
        total_memory = 0
        total_time = 0
        bound_counts = {"memory": 0, "compute": 0}
        
        for layer_name, layer_results in results[stage].items():
            total_ops += layer_results["OPs"]
            total_memory += layer_results["memory_access"]
            total_time += layer_results["inference_time"]
            bound_counts[layer_results["bound"]] += 1
            
        # Determine overall bound
        overall_bound = "Memory" if bound_counts["memory"] > bound_counts["compute"] else "Compute"
        
        return total_ops, total_memory, total_time, overall_bound
    
    def estimate_performance(
        self,
        batch_size: int,
        seq_lengths: List[int],
        w_bit: int = 16,
        a_bit: int = 16,
        kv_bit: Optional[int] = None,
        use_flashattention: bool = False,
        **kwargs
    ) -> PerformanceEstimate:
        """Estimate performance using llm_viewer's detailed modeling"""
        # Use the first sequence length for now since llm_viewer doesn't support varying lengths
        seq_length = seq_lengths[0] if seq_lengths else 512
        
        # Run llm_viewer analysis
        results = self.analyzer.analyze(
            seqlen=seq_length,
            batchsize=batch_size,
            w_bit=w_bit,
            a_bit=a_bit,
            kv_bit=kv_bit,
            use_flashattention=use_flashattention
        )
        
        # Aggregate prefill results
        prefill_ops, prefill_memory, prefill_time, prefill_bound = self._aggregate_results(results, "prefill")
        prefill_tflops = prefill_ops / (prefill_time * 1e12) if prefill_time > 0 else 0
        
        # Aggregate decode results
        decode_ops, decode_memory, decode_time, decode_bound = self._aggregate_results(results, "decode")
        decode_tflops = decode_ops / (decode_time * 1e12) if decode_time > 0 else 0
        
        # Calculate total memory including weights and KV cache
        total_memory = (
            # Model weights
            (12 * self.model.n_layers * self.model.d_model**2 + 
             2 * self.model.d_model * self.model.vocab_size) * (w_bit / 8) +
            # KV cache
            2 * 2 * self.model.n_layers * self.model.n_heads * self.model.d_head * 
            sum(seq_lengths) * batch_size * ((kv_bit or a_bit) / 8)
        ) / 1e9  # Convert to GB
        
        # Get memory and compute times from roofline analysis
        _, prefill_performance, _ = roofline_analyze(
            self.hardware.memory_bandwidth * 1e9,  # Convert to bytes/s
            self.hardware.tflops * 1e12,  # Convert to OPS/s
            prefill_ops,
            prefill_memory
        )
        prefill_compute_time = prefill_ops / (self.hardware.tflops * 1e12)
        prefill_memory_time = prefill_memory / (self.hardware.memory_bandwidth * 1e9)
        
        _, decode_performance, _ = roofline_analyze(
            self.hardware.memory_bandwidth * 1e9,
            self.hardware.tflops * 1e12,
            decode_ops,
            decode_memory
        )
        decode_compute_time = decode_ops / (self.hardware.tflops * 1e12)
        decode_memory_time = decode_memory / (self.hardware.memory_bandwidth * 1e9)
        
        return PerformanceEstimate(
            model_name=self.model.name,
            gpu_name=self.hardware.name,
            parameters_b=(12 * self.model.n_layers * self.model.d_model**2 + 
                        2 * self.model.d_model * self.model.vocab_size) / 1e9,
            prefill_tflops=prefill_tflops,
            decode_tflops=decode_tflops,
            total_memory_gb=total_memory,
            prefill_bound=prefill_bound,
            decode_bound=decode_bound,
            prefill_memory_time=prefill_memory_time,
            decode_memory_time=decode_memory_time,
            prefill_compute_time=prefill_compute_time,
            decode_compute_time=decode_compute_time,
            prefill_estimated_latency=prefill_time,
            decode_estimated_latency=decode_time
        ) 