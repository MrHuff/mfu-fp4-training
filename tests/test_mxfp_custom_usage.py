import sys
import os
import torch

# Allow relative imports if needed (though we use absolute paths in imports)
sys.path.append("/opt/mfu/EXTERNAL_PATH")

try:
    print("Attempting to import low_bits_training.quantization.mxfp_custom")
    import low_bits_training.quantization.mxfp_custom
    print("Successfully imported low_bits_training.quantization.mxfp_custom")
except ImportError as e:
    print(f"Failed to import: {e}")
    sys.exit(1)
except Exception as e:
    print(f"An error occurred: {e}")
    sys.exit(1)

# Check if we can instantiate the converter (which calls the factory)
try:
    # Mock JobConfig
    class MockConfig:
        def __init__(self, use_2d=False):
            self.mxfp_custom = type('obj', (object,), {
                "mlp_recipe": "Custom",
                "attn_recipe": "None",
                "exclude_last_n_layers": 0,
                "verbose": True,
                "block_size": 32,
                "scale_type": "E8M0",
                "roundMode": "TiesToEven",
                "strategy": "encode",
                "use_global_scale": True,
                "use_rht": False,
                "scale_round_mode": "TiesToEven",
                "use_2d_weights": use_2d
            })
            self.model = type('obj', (object,), {"n_layers": 2})

    # Test Case 1: Default (No 2D)
    print("\n--- Test Case 1: Default (No 2D) ---")
    config = MockConfig(use_2d=False)
    converter = low_bits_training.quantization.mxfp_custom.TECustomConverter(config, None)
    recipe = converter._build_custom_recipe()
    
    # Check linear_weight quantizer
    # CustomRecipe(qfactory=...)
    # recipe.qfactory is the factory closure
    factory = recipe.qfactory
    q_weight = factory("linear_weight")
    print(f"Quantizer (No 2D) tile shape: {q_weight.quant_tile_shape}")
    assert q_weight.quant_tile_shape == (1, 32), f"Expected (1, 32), got {q_weight.quant_tile_shape}"

    # Test Case 2: With 2D Weights
    print("\n--- Test Case 2: With 2D Weights ---")
    config_2d = MockConfig(use_2d=True)
    converter_2d = low_bits_training.quantization.mxfp_custom.TECustomConverter(config_2d, None)
    recipe_2d = converter_2d._build_custom_recipe()
    
    factory_2d = recipe_2d.qfactory
    q_weight_2d = factory_2d("linear_weight")
    print(f"Quantizer (2D) tile shape: {q_weight_2d.quant_tile_shape}")
    assert q_weight_2d.quant_tile_shape == (32, 32), f"Expected (32, 32), got {q_weight_2d.quant_tile_shape}"

    print("\nSUCCESS: All tests passed!")
    
except Exception as e:
    print(f"Failed to use converter/factory: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
