#
# Copyright (c) 2025 Graphcore Ltd. All rights reserved.
#

import torch
from torch import nn

import tensor_tracker

from low_bits_training.umup.umup_llama import UmupTransformer, UmupTransformerModelArgs
from low_bits_training.umup.umup_loss import umup_nll_loss

# Given a reference model, the acceptable range is given by the min and max of 1000 random runs
# Where for each run the weights are randomly initialised and the inputs and targets are randomly generated
# And we calculate the standard deviation for each activation, activation gradient, and weight gradient

acceptable_ranges_activations = {
    "tok_embeddings": (0.9932193160057068, 1.0078219175338745),
    "wq": (0.9849173426628113, 1.0167585611343384),
    "wk": (0.9810352325439453, 1.0238533020019531),
    "wv": (0.9810855984687805, 1.0200368165969849),
    "wo": (0.9739148020744324, 1.151713252067566),
    "w1": (0.9822267889976501, 1.0162793397903442),
    "w2": (0.9485584497451782, 1.0597060918807983),
    "w3": (0.984933078289032, 1.0156807899475098),
    "gated_swish": (0.953790009021759, 1.0551731586456299),
    "output": (0.08772144466638565, 0.08896193653345108),
}

acceptable_ranges_activation_gradients = {
    "tok_embeddings": (1.1510810852050781, 1.2852238416671753),
    "wq": (0.25568535923957825, 0.2923278212547302),
    "wk": (0.21765558421611786, 0.2499304562807083),
    "wv": (0.861728847026825, 1.0830767154693604),
    "wo": (1.3080816268920898, 1.4428142309188843),
    "w1": (0.6421118378639221, 0.6898064017295837),
    "w2": (1.0478606224060059, 1.1123626232147217),
    "w3": (0.6194161176681519, 0.6722511053085327),
    "gated_swish": (0.6315210461616516, 0.6705107092857361),
    "output": (0.9999979138374329, 1.0000063180923462),
    "model": (0.0220916960388422, 0.0220916960388422),
}

acceptable_ranges_gradients = {
    "tok_embeddings": (1.1548209190368652, 1.3078813552856445),
    "wq": (0.2549141049385071, 0.29485252499580383),
    "wk": (0.25504833459854126, 0.2962283194065094),
    "wv": (0.8933900594711304, 1.0948114395141602),
    "wo": (0.888414740562439, 1.0794399976730347),
    "w1": (0.6416606307029724, 0.6945044994354248),
    "w2": (1.0284422636032104, 1.1177750825881958),
    "w3": (0.6119191646575928, 0.6844502091407776),
    "output": (0.9939054846763611, 1.0065935850143433),
}


def test__umup_model_scales():
    """Test that the stdevs of all activations and gradients of activations an weights are within the expected range.

    If this test starts failing, consider checking:
    - Whether the stdevs are only just slightly out of range (due to, say, changes to RNG)
        - There is, unfortunately, a small but unavoidable chance of this happening
        - This can be remedied by expanding the acceptable range slightly to accomodate the new value
    - Whether the default values for the model args have been changed
    - Whether the unit_scaling library APIs have changed
    - Whether the tensor_tracker library has changed or been removed from Github
    - Whether the names of the leaf modules (e.g. Linear layers) in the model definition have changed
    """
    # https://www.youtube.com/watch?v=HOp8w2PgHlM
    torch.manual_seed(17_2_44)
    vocab_size = 2048

    model_args = UmupTransformerModelArgs(
        dim=128,
        n_layers=1,
        n_heads=8,
        vocab_size=2048,
        multiple_of=32,
    )

    # tensor_tracker needs loss to be returned by module for full tracking
    class UmupModelWithLoss(nn.Module):
        def __init__(self, model_args):
            super().__init__()
            self.model = UmupTransformer(model_args)

        def forward(self, inputs, targets):
            outputs = self.model(inputs)
            loss = umup_nll_loss(outputs, targets)
            return loss

    # Random inputs, targets = inputs shifted by 1 + "EOS" token
    inputs = torch.randint(low=0, high=vocab_size - 1, size=[4, 256], dtype=torch.int64)
    targets = torch.cat(
        (
            inputs[:, 1:],
            torch.full(size=[4, 1], fill_value=vocab_size - 1, dtype=torch.int64),
        ),
        dim=-1,
    )
    module = UmupModelWithLoss(model_args)

    with tensor_tracker.track(module) as tracker:
        loss = module(inputs, targets)
        loss.backward()

    # Small margin for error on either side, especially in cases where min acceptable = max acceptable
    epsilon = 1e-8

    # Check activations and activation gradients

    # We want to make sure that all expected values are checked, so we track which ones have been
    activations_found = set()
    activation_grads_found = set()

    for stash in tracker.stashes:
        # There is only one transformer layer so there should be only one layer with each name
        module_id = stash.name.split(".")[-1]
        is_grad = stash.grad
        value = stash.value if isinstance(stash.value, torch.Tensor) else stash.value[0]

        if is_grad:
            # Check that activation gradient is not a duplicate and is in expected range
            if module_id in acceptable_ranges_activation_gradients:
                # Check not duplicate
                if module_id in activation_grads_found:
                    raise Exception(
                        f"Activation gradient for module ID {module_id} not unique"
                    )
                activation_grads_found.add(module_id)

                # Check in expected range
                min_accept, max_accept = acceptable_ranges_activation_gradients[module_id]
                std = value.std()
                if std < min_accept - epsilon or std > max_accept + epsilon:
                    raise Exception(
                        f"Std of activation grad of module named {module_id} "
                        f"outside acceptable range ([{min_accept}, {max_accept}])"
                    )

        else:
            # Check that activation is not a duplicate and is in expected range
            if module_id in acceptable_ranges_activations:
                # Check not duplicate
                if module_id in activations_found:
                    raise Exception(f"Activation for module ID {module_id} not unique")
                activations_found.add(module_id)

                # Check in expected range
                min_accept, max_accept = acceptable_ranges_activations[module_id]
                std = value.std()
                if std < min_accept - epsilon or std > max_accept + epsilon:
                    raise Exception(
                        f"Std of activation of module named {module_id} "
                        f"outside acceptable range ([{min_accept}, {max_accept}])"
                    )

    # Check that all expected values have been checked
    if activations_found != set(acceptable_ranges_activations.keys()):
        raise Exception("Not all expected activations found")

    if activation_grads_found != set(acceptable_ranges_activation_gradients.keys()):
        raise Exception("Not all expected activation gradients found")

    # Now check parameter gradients

    # We want to make sure that all expected values are checked, so we track which ones have been
    grads_found = set()

    for name, parameter in module.named_parameters():
        # Check not duplicate
        # Parameter name is e.g. <module name>.weight so we take second-last item of name.split('.')
        module_id = name.split(".")[-2]
        if module_id in grads_found:
            raise Exception(f"Gradient for module ID {module_id} is not unique")
        grads_found.add(module_id)

        # Check in expected range
        min_accept, max_accept = acceptable_ranges_gradients[module_id]
        assert parameter.grad is not None
        std = parameter.grad.std()
        if std < min_accept - epsilon or std > max_accept + epsilon:
            raise Exception(
                f"Std of gradient of weight of module named {module_id} "
                f"outside acceptable range ([{min_accept}, {max_accept}])"
            )

    # Check that all expected values have been checked
    if grads_found != set(acceptable_ranges_gradients.keys()):
        raise Exception("Not all expected weight gradients found")
