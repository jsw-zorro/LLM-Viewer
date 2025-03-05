"""Tests for the LLMViewerBridge class."""

import unittest
import numpy as np
import sys
import os

# Add the project root directory to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from llm_viewer.llm_viewer_bridge import LLMViewerBridge, HardwareSpecs, ModelSpecs
from token_analysis.hardware_configs import get_gpu_config
import ipdb

class TestLLMViewerBridge(unittest.TestCase):
    """Test cases for LLMViewerBridge class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Define model and GPU configurations
        model_name = "meta-llama/Llama-2-7b-hf"
        gpu_name = "NVIDIA A100 80GB PCIe"
        
        # Create model specs
        self.model = ModelSpecs.from_hf_config(model_name)
        
        # Get GPU configuration
        gpu_config = get_gpu_config(gpu_name)
        
        # Define test hardware specs
        self.hardware = HardwareSpecs(
            tflops=gpu_config.tflops,
            memory_bandwidth=gpu_config.memory_bandwidth,
            memory_capacity=gpu_config.memory_capacity,
            name=gpu_name
        )
        
        # Create LLMViewerBridge instance
        self.bridge = LLMViewerBridge(self.hardware, self.model)

    def test_performance_estimates_equivalence(self):
        """Test if estimate_performance and unified_estimate_performance give same results for equivalent inputs."""
        # Test parameters
        batch_size = 1
        seq_lengths = [512, 256]  # Single sequence length for basic test
        w_bit = 16
        a_bit = 16
        kv_bit = None
        use_flashattention = True
        
        # Get estimate from estimate_performance
        regular_estimate = self.bridge.estimate_performance(
            batch_size=batch_size,
            seq_lengths=seq_lengths,
            w_bit=w_bit,
            a_bit=a_bit,
            kv_bit=kv_bit,
            use_flashattention=use_flashattention
        )
        
        # Get estimate from unified_estimate_performance with equivalent parameters
        unified_estimate_prefill = self.bridge.unified_estimate_performance(
            batchsize=batch_size,
            w_bit=w_bit,
            a_bit=a_bit,
            kv_bit=kv_bit,
            use_flashattention=use_flashattention,
            previous_tokens=[0, 0],  # No previous tokens
            chunk_sizes=[seq_lengths[0], seq_lengths[1]]  # Same as seq_lengths
        )
        
        unified_estimate_decode = self.bridge.unified_estimate_performance(
            batchsize=batch_size,
            w_bit=w_bit,
            a_bit=a_bit,
            kv_bit=kv_bit,
            use_flashattention=use_flashattention,
            previous_tokens=[seq_lengths[0], seq_lengths[1]],
            chunk_sizes=[1, 1],
        )
        
        # ipdb.set_trace()
        self.assertEqual(unified_estimate_prefill.total_time, regular_estimate.prefill_estimated_latency)
        self.assertEqual(unified_estimate_decode.total_time, regular_estimate.decode_estimated_latency)
        # note for decode, there will be slight difference, as we changed the effective seqlen to past_length + token_len, and the old method use the past_length as the main variable.  
     

if __name__ == '__main__':
    unittest.main() 