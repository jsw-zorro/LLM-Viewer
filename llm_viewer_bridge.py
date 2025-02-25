"""Bridge between llm_viewer's modeling and our performance analysis framework."""

import os
import sys
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import math

# Add parent directory to path to import from token_analysis
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from token_analysis.model_base import (
    LLMPerformanceModel, HardwareSpecs, ModelSpecs, PerformanceEstimate
)

from roofline_model import roofline_analyze
from model_analyzer import ModelAnalyzer
import ipdb

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
        self.hardware = hardware
        
    def _aggregate_results(self, results: Dict[str, Dict[str, Any]], stage: str) -> Tuple[float, float, float, str]:
        """Aggregate layer-wise results into total metrics"""
       
        OPs = results['total_results'][stage]["OPs"]
        memory_access = results['total_results'][stage]["memory_access"]
        total_time = results['total_results'][stage]["inference_time"]
        bandwidth, max_OPS, onchip_buffer = self.analyzer.get_hardware_info()
        arithmetic_intensity, performance, bound = roofline_analyze(bandwidth, max_OPS, OPs, memory_access)
       
        return OPs, memory_access, total_time, bound

    def estimate_performance(
        self,
        batch_size: int,
        seq_lengths: List[int],
        w_bit: int = 16,
        a_bit: int = 16,
        kv_bit: Optional[int] = None,
        use_flashattention: bool = True,
        **kwargs
    ) -> PerformanceEstimate:
        """Estimate performance using llm_viewer's detailed modeling"""
        # Use the first sequence length for now since llm_viewer doesn't support varying lengths
        seq_length = seq_lengths[0] if seq_lengths else 512
        
        # Run llm_viewer analysis for both prefill and decode
        results = self.analyzer.analyze_varying_full(
            seqlens=seq_lengths,
            batchsize=batch_size,
            w_bit=w_bit,
            a_bit=a_bit,
            kv_bit=kv_bit,
            use_flashattention=use_flashattention
        )
        
        # # need to check where results1 differs from results2 and cause the difference.
        # for stage in ["prefill", "decode"]:
        #     for layer in results1[stage]:
        #         if results1[stage][layer] != results[stage][layer]:
        #             print(f"results1[{stage}][{layer}] = {results1[stage][layer]}")
        #             print(f"results[{stage}][{layer}] = {results[stage][layer]}")
        
        
        # TODO: I think we don't need to do the aggregatio anymore, all the necessary information are in total files.

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
        
        prefill_compute_time = prefill_ops / (self.hardware.tflops * 1e12)
        prefill_memory_time = prefill_memory / (self.hardware.memory_bandwidth * 1e9)
      
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

    def estimate_performance_with_type(
        self,
        batch_size: int,
        seq_lengths: List[int],
        request_types: List[str],
        w_bit: int = 16,
        a_bit: int = 16,
        kv_bit: Optional[int] = None,
        use_flashattention: bool = True,
        **kwargs
    ) -> PerformanceEstimate:
        """
        Estimate performance for mixed prefill and decode operations within the same batch.
        
        This function is designed to handle the "chunk prefill" mode where some sequences in a batch
        may be in prefill mode while others are in decode mode. It analyzes each sequence based on
        its specified request type.
        
        Args:
            batch_size: The batch size for the request
            seq_lengths: List of sequence lengths for each request
            request_types: List of request types ('1' for prefill, '0' for decode) for each sequence
                           Format example: '1-0-0' means the first sequence is prefill, the next two are decode
            w_bit: Weight precision in bits
            a_bit: Activation precision in bits
            kv_bit: KV cache precision in bits (defaults to a_bit if None)
            use_flashattention: Whether to use Flash Attention for performance modeling
            
        Returns:
            PerformanceEstimate: Performance estimates for the mixed batch
        """
        # Validate inputs
        assert len(seq_lengths) == len(request_types.split('-')), "Number of sequence lengths must match number of request types"
        
        # Split sequences by request type
        prefill_seq_lengths = []
        decode_seq_lengths = []
        
        for seq_len, req_type in zip(seq_lengths, request_types.split('-')):
            if req_type == '1':  # Prefill
                prefill_seq_lengths.append(seq_len)
            elif req_type == '0':  # Decode
                decode_seq_lengths.append(seq_len)
            else:
                raise ValueError(f"Invalid request type: {req_type}. Must be '0' (decode) or '1' (prefill)")
        
        # Initialize results
        prefill_ops = 0
        prefill_memory = 0
        prefill_time = 0
        prefill_bound = ""
        
        decode_ops = 0
        decode_memory = 0
        decode_time = 0
        decode_bound = ""
        
        # Analyze prefill sequences if any
        if prefill_seq_lengths:
            prefill_results = self.analyzer.analyze_varying_full_with_type(
                seqlens=prefill_seq_lengths,
                batchsize=len(prefill_seq_lengths),  # Use the number of prefill sequences as batch size
                op_type=1,  # 1 for prefill
                w_bit=w_bit,
                a_bit=a_bit,
                kv_bit=kv_bit,
                use_flashattention=use_flashattention
            )
            
            prefill_ops, prefill_memory, prefill_time, prefill_bound = self._aggregate_results(prefill_results, "prefill")
        
        # Analyze decode sequences if any
        if decode_seq_lengths:
            decode_results = self.analyzer.analyze_varying_full_with_type(
                seqlens=decode_seq_lengths,
                batchsize=len(decode_seq_lengths),  # Use the number of decode sequences as batch size
                op_type=0,  # 0 for decode
                w_bit=w_bit,
                a_bit=a_bit,
                kv_bit=kv_bit,
                use_flashattention=use_flashattention
            )
            
            decode_ops, decode_memory, decode_time, decode_bound = self._aggregate_results(decode_results, "decode")
        
        # Calculate TFLOPS
        prefill_tflops = prefill_ops / (prefill_time * 1e12) if prefill_time > 0 else 0
        decode_tflops = decode_ops / (decode_time * 1e12) if decode_time > 0 else 0
        
        # Calculate total memory including weights and KV cache
        total_memory = (
            # Model weights
            (12 * self.model.n_layers * self.model.d_model**2 + 
             2 * self.model.d_model * self.model.vocab_size) * (w_bit / 8) +
            # KV cache
            2 * 2 * self.model.n_layers * self.model.n_heads * self.model.d_head * 
            sum(seq_lengths) * ((kv_bit or a_bit) / 8)
        ) / 1e9  # Convert to GB
        
        # Calculate compute and memory times
        prefill_compute_time = prefill_ops / (self.hardware.tflops * 1e12) if prefill_ops > 0 else 0
        prefill_memory_time = prefill_memory / (self.hardware.memory_bandwidth * 1e9) if prefill_memory > 0 else 0
        
        decode_compute_time = decode_ops / (self.hardware.tflops * 1e12) if decode_ops > 0 else 0
        decode_memory_time = decode_memory / (self.hardware.memory_bandwidth * 1e9) if decode_memory > 0 else 0
        
        # Return combined performance estimate
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