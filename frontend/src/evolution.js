import {EventsEmit, EventsOff, EventsOn, LogDebug} from "../wailsjs/runtime";
import {Evolve} from "../wailsjs/go/main/App";
import {initAllChart, initBestChart, updateAllChart, updateBestChart} from "./charts";
import {getAdvancedConfig} from "./advancedConfig";
import {initVisualization, pushBestLayers} from "./visualization";

window.evolve = function() {
    let trainTestRatio = parseFloat(document.querySelector("#config-train-test-ratio").value);
    let numIndividuals = parseInt(document.querySelector("#config-num-individuals").value);
    let numGenerations = parseInt(document.querySelector("#config-num-generations").value);
    LogDebug(numGenerations.toString());
    let advCfg= getAdvancedConfig();

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

    EventsOff("evo-progress", "evo-all-chart", "evo-best-chart", "evo-best-layers");

    EventsOn("evo-progress", (progress) => {
        if (progress.Generation === -1) {
            startButton.classList.remove("visually-hidden");
            cancelButton.classList.add("visually-hidden");
            progressBar.classList.add("visually-hidden");
            progressBarFill.style.width = "0%";
            progressStatus.innerHTML = window.isAborting ? "Прервано" : "Завершено";
            if (window.isAborting) {
                window.isAborting = false;
            }

            LogDebug("Evolution finished (frontend)");
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
        LogDebug(numGenerations.toString());
        progressStatus.innerHTML = `Поколение ${progress.Generation+1} из ${numGenerations}
            ${eta ? " &nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp; " : ""}${eta}`

        if (progress.Generation > window.currentGeneration) {
            initAllChart(advCfg.Epochs);
        }
        window.currentGeneration = progress.Generation;
    });

    EventsOn("evo-all-chart", allChartData => {
        updateAllChart(allChartData);
    });
    EventsOn("evo-best-chart", bestAccuracy => {
        updateBestChart(bestAccuracy);
    });
    EventsOn("evo-best-layers", bestLayers => {
        pushBestLayers(bestLayers);
    });
    initAllChart(advCfg.Epochs);
    initBestChart(numGenerations);
    initVisualization();

    Evolve(advCfg, trainTestRatio, numIndividuals, numGenerations).then(() => {});
}

window.isAborting = false;
window.abortEvolution = function() {
    window.isAborting = true;
    let progressStatus = document.querySelector("#evo-progress-status");
    progressStatus.innerHTML = "Остановка...";
    EventsEmit("evo-abort");
}
