import {EventsEmit, EventsOff, EventsOn} from "../wailsjs/runtime";
import {Evolve} from "../wailsjs/go/main/App";
import {initAllChart, initBestChart} from "./charts";

window.evolve = function() {
    let trainTestRatio = parseFloat(document.querySelector("#config-train-test-ratio").value);
    let numIndividuals = parseInt(document.querySelector("#config-num-individuals").value);
    let numGenerations = parseInt(document.querySelector("#config-num-generations").value);
    let advCfg = {
        Epochs: parseInt(document.querySelector("#config-epochs").value),
        BatchSize: parseInt(document.querySelector("#config-batch-size").value),

        MutationChance: parseFloat(document.querySelector("#config-mutation-chance").value),

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

    let progressBar = document.querySelector("#evo-progress-bar");
    let progressBarFill = document.querySelector("#evo-progress-bar-fill");
    let progressStatus = document.querySelector("#evo-progress-status");
    let startButton = document.querySelector("#evo-start-button");
    let cancelButton = document.querySelector("#evo-cancel-button");
    startButton.classList.add("visually-hidden");
    cancelButton.classList.remove("visually-hidden");
    progressStatus.innerHTML = "Подготовка...";
    progressStatus.classList.remove("visually-hidden");
    progressBar.classList.remove("visually-hidden");

    EventsOn("evo-progress", (progress) => {
        if (progress.Generation === -1) {
            EventsOff("evo-progress");

            startButton.classList.remove("visually-hidden");
            cancelButton.classList.add("visually-hidden");
            progressBar.classList.add("visually-hidden");
            progressBarFill.style.width = "0%";
            progressStatus.innerHTML = window.isAborting ? "Прервано" : "Завершено";
            if (window.isAborting) {
                window.isAborting = false;
            }
            console.log("Evolution finished (frontend)");
            return;
        }
        if (window.isAborting) {
            return;
        }
        progressBarFill.style.width = progress.Individual / (numIndividuals * numGenerations) * 100 + "%";

        let eta = "";
        if (progress.ETASeconds > 1) {
            let minutes = Math.ceil(progress.ETASeconds / 60);
            eta = `Осталось ~ ${minutes} мин.`;
        }
        progressStatus.innerHTML = `Поколение ${progress.Generation+1} из ${numGenerations}
            ${eta ? " &nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp; " : ""}${eta}`

        if (progress.Generation > window.currentGeneration) {
            initAllChart(advCfg.Epochs);
        }
        window.currentGeneration = progress.Generation;
    });

    EventsOn("evo-all-chart", allChartData => {
        let i = window.allChart.data.datasets.findIndex(ds => ds.label === allChartData.Name)
        if (i === -1) {
            window.allChart.data.datasets.push({
                label: allChartData.Name,
                data: [allChartData.Accuracy],
                borderColor: "#"+Math.floor(Math.random()*16777215).toString(16),
                tension: 0.1,
                pointRadius: 1,
            });
        } else {
            window.allChart.data.datasets[i].data.push(allChartData.Accuracy);
        }
        window.allChart.update("none");
    });

    EventsOn("evo-best-chart", bestAccuracy => {
        console.log("evo-best-chart", bestAccuracy)
        window.bestChart.data.datasets[0].data.push(bestAccuracy);
        window.bestChart.update("none");
    });

    initAllChart(advCfg.Epochs);
    initBestChart(numGenerations);

    Evolve(advCfg, trainTestRatio, numIndividuals, numGenerations).then(() => {
        setTimeout(function() {
            EventsOff("evo-all-chart", "evo-best-chart");
        }, 1000);
    });
}

window.isAborting = false;
window.abortEvolution = function() {
    window.isAborting = true;
    let progressStatus = document.querySelector("#evo-progress-status");
    progressStatus.innerHTML = "Остановка...";
    EventsEmit("evo-abort");
}
