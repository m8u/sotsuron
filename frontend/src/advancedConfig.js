window.resetAdvancedConfig = function() {
    document.querySelector("#config-train-test-ratio").value = 0.8;
    document.querySelector("#config-epochs").value = 5;
    document.querySelector("#config-batch-size").value = 10;
    document.querySelector("#config-mutation-multiplier").value = 1.0;
    document.querySelector("#config-max-conv-max-pooling-pairs").value = 3;
    document.querySelector("#config-max-conv-output").value = 16;
    document.querySelector("#config-max-conv-kernel-size").value = 8;
    document.querySelector("#config-max-conv-pad").value = 2;
    document.querySelector("#config-max-conv-stride").value = 1;
    document.querySelector("#config-max-pool-kernel-size").value = 8;
    document.querySelector("#config-max-pool-pad").value = 2;
    document.querySelector("#config-max-pool-stride").value = 1;
    document.querySelector("#config-max-dense-layers").value = 1;
    document.querySelector("#config-max-dense-size").value = 128;
    document.querySelector("#config-min-resolution-width").value = 16;
    document.querySelector("#config-min-resolution-height").value = 16;
}

export function getAdvancedConfig() {
    return {
        Epochs: parseInt(document.querySelector("#config-epochs").value),
        BatchSize: parseInt(document.querySelector("#config-batch-size").value),

        MutationMultiplier: parseFloat(document.querySelector("#config-mutation-multiplier").value),

        MaxConvMaxPoolingPairs: parseInt(document.querySelector("#config-max-conv-max-pooling-pairs").value),
        MaxConvOutput: parseInt(document.querySelector("#config-max-conv-output").value),
        MaxConvKernelSize: parseInt(document.querySelector("#config-max-conv-kernel-size").value),
        MaxConvPad: parseInt(document.querySelector("#config-max-conv-pad").value),
        MaxConvStride: parseInt(document.querySelector("#config-max-conv-stride").value),
        MaxPoolKernelSize: parseInt(document.querySelector("#config-max-pool-kernel-size").value),
        MaxPoolPad: parseInt(document.querySelector("#config-max-pool-pad").value),
        MaxPoolStride: parseInt(document.querySelector("#config-max-pool-stride").value),
        MaxDenseLayers: parseInt(document.querySelector("#config-max-dense-layers").value),
        MaxDenseSize: parseInt(document.querySelector("#config-max-dense-size").value),
        MinResolutionWidth: parseInt(document.querySelector("#config-min-resolution-width").value),
        MinResolutionHeight: parseInt(document.querySelector("#config-min-resolution-height").value),
    }
}
