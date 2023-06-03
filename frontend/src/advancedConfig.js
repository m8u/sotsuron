window.resetConfig = function() {
    document.querySelector("#config-train-test-ratio").value = 0.8;
    document.querySelector("#config-epochs").value = 5;
    document.querySelector("#config-batch-size").value = 10;
    document.querySelector("#config-mutation-chance").value = 0.3;
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
