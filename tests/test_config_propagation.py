
import unittest
import tomllib
import torch
import sys
import os
from types import SimpleNamespace

# Add workspace to path
if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())

from low_bits_training.quantization.te_parity_linear_triton import TritonTEParityLinear
from transformer_engine.pytorch.experimental.quantization_custom_triton import RM_TOWARD_ZERO, RM_TIES_TO_EVEN, RM_STOCHASTIC, RM_TOWARD_POSITIVE, RM_TOWARD_NEGATIVE

class MockMXConfig:
    def __init__(self, config_dict):
        for k, v in config_dict.items():
            setattr(self, k, v)
            
class TestConfigPropagation(unittest.TestCase):
    def test_toml_propagation(self):
        toml_path = "train_configs/llama3_1B_e5m3_bf16_simulation_encode_neg_wikitext.toml"
        print(f"Reading config from: {toml_path}")
        
        with open(toml_path, "rb") as f:
            data = tomllib.load(f)
            
        # Extract mxfp_custom section
        if "mxfp_custom" not in data:
            self.fail("Could not find [mxfp_custom] section in TOML")
            
        mxfp_cfg = data["mxfp_custom"]
        config_obj = MockMXConfig(mxfp_cfg)
        
        print("Config Loaded:")
        for k, v in mxfp_cfg.items():
            print(f"  {k} = {v}")
            
        # Instantiate Layer
        layer = TritonTEParityLinear(
            in_features=128, 
            out_features=128, 
            bias=False, 
            mx_config=config_obj
        )
        
        # 1. Check Quantizer Configs
        q_in = layer.input_quantizer
        
        # block_size
        self.assertEqual(q_in.block_size, mxfp_cfg.get("block_size", 32), "Block Size Mismatch")
        
        # scale_type -> scale_format
        self.assertEqual(q_in.scale_format, mxfp_cfg.get("scale_type", "E4M3"), "Scale Format Mismatch")
        print(q_in.fmt.max_val)
        # roundMode -> round_mode (String check)
        expected_rm = mxfp_cfg.get("roundMode", "TiesToEven")
        self.assertEqual(q_in.round_mode, expected_rm, "Round Mode String Mismatch")
        print(q_in.data_rm)
        # scale_round_mode
        expected_srm = mxfp_cfg.get("scale_round_mode", "TiesToEven")
        self.assertEqual(q_in.scale_round_mode, expected_srm, "Scale Round Mode String Mismatch")
        print(q_in.scale_rm)

        # use_fp32_matmul
        expected_fp32 = mxfp_cfg.get("use_fp32_matmul", False)
        self.assertEqual(layer.use_fp32_matmul, expected_fp32, "FP32 Matmul Flag Mismatch")
        
        # use_2d_weights
        expected_2d = mxfp_cfg.get("use_2d_weights", False)
        self.assertEqual(q_in.with_2d_weights, expected_2d, "Use 2D Weights Mismatch")
        
        print("\nSUCCESS: All config flags propagated correctly!")

if __name__ == "__main__":
    unittest.main()
