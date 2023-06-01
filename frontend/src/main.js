import "bootstrap/dist/js/bootstrap.bundle.min.js";
import "bootstrap/dist/css/bootstrap.min.css";
import {Chart} from "chart.js/auto";
import {Evolve, LoadDataset, LoadImage, Predict} from '../wailsjs/go/main/App';
import {EventsEmit, EventsOff, EventsOn} from "../wailsjs/runtime";


// Chart.defaults.responsive = true;
Chart.defaults.devicePixelRatio = 2;
Chart.defaults.maintainAspectRatio = false;

function resizeCanvases() {
    const controls = document.querySelector("#controls");
    const height = window.innerHeight - controls.offsetHeight - 30;
    window.allChart.resize(window.allChart.width, height);
    window.bestChart.resize(window.bestChart.width, height);
}

function initAllChart(epochs=10) {
    if (window.allChart != null) {
        window.allChart.destroy();
    }
    window.allChart = new Chart(
        document.getElementById("all-chart-canvas"),
        {
            type: "line",
            data: {
                labels: Array.from({length: epochs}, (_, i) => i + 1),
                datasets: [],
            },
            options: {
                plugins: {
                    title: {
                        display: true,
                        text: "Точность всех особей в текущем поколении"
                    },
                    legend: {
                        display: false
                    }
                },
                scales: {
                    x: {
                        title: {
                            display: true,
                            text: "Итерация обучения"
                        }
                    }
                }
            }
        }
    );
}

function initBestChart(generations=10) {
    if (window.bestChart != null) {
        window.bestChart.destroy();
    }
    window.bestChart = new Chart(
        document.getElementById("best-chart-canvas"),
        {
            type: "line",
            data: {
                labels: Array.from({length: generations}, (_, i) => i + 1),
                datasets: [
                    {
                        data: [],
                        tension: 0.1,
                        pointRadius: 1,
                    }
                ],
            },
            options: {
                plugins: {
                    title: {
                        display: true,
                        text: "Приспособленность наилучшей особи"
                    },
                    legend: {
                        display: false
                    },
                },
                scales: {
                    x: {
                        title: {
                            display: true,
                            text: "Поколение"
                        }
                    }
                }
            }
        }
    );
}

window.onload = async function () {
    initAllChart();
    initBestChart();
    resizeCanvases();
    window.addEventListener("resize", resizeCanvases, false);
    resetConfig();

    let liveToast = document.querySelector("#live-toast")
    let liveToastBody = document.querySelector("#live-toast-body")
    EventsOn("error", (msg) => {
        liveToastBody.innerHTML = msg;
        const toast = new bootstrap.Toast(liveToast);
        toast.show()
    });
}

window.loadDataset = async function(grayscale) {
    let datasetInfo = await LoadDataset(grayscale);
    if (!datasetInfo) {
        return;
    }
    document.querySelector("#dataset-name").value = datasetInfo.Name;
    document.querySelector("#dataset-name").value = datasetInfo.Name;
    document.querySelector("#dataset-num-classes").value = datasetInfo.NumClasses;
    document.querySelector("#dataset-num-images").value = datasetInfo.NumImages;
    document.querySelector("#dataset-resolution").value =
        datasetInfo.Resolution.Width + "x" + datasetInfo.Resolution.Height;
    document.querySelector("#dataset-num-channels").value = grayscale ? 1 : 3;

    document.querySelector("#evo-start-button").disabled = false;
};

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
            startButton.classList.remove("visually-hidden");
            cancelButton.classList.add("visually-hidden");
            progressBar.classList.add("visually-hidden");
            progressBarFill.style.width = "0%";
            progressStatus.innerHTML = window.isAborting ? "Прервано" : "Завершено";
            EventsOff("evo-progress");
            if (window.isAborting) {
                window.isAborting = false;
            }
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
        EventsOff("evo-progress", "evo-all-chart", "evo-best-chart");
        console.log("Evolution finished");
    });
}

window.isAborting = false;
window.abortEvolution = function() {
    window.isAborting = true;
    let progressStatus = document.querySelector("#evo-progress-status");
    progressStatus.innerHTML = "Остановка...";
    EventsEmit("evo-abort");
}

window.toyTest = async function() {
    let loadedFilename = await LoadImage();
    if (!loadedFilename) {
        return;
    }
    document.querySelector("#toy-test-file-name").value = loadedFilename;
    let probabilities = await Predict()
    console.log(probabilities);
    document.querySelector("#toy-test-predicted-class").value = probabilities[0].ClassName;
    let probabilitiesContainer = document.querySelector("#toy-test-probabilities");
    probabilitiesContainer.innerHTML = "";
    for (let i = 0; i < 5; i++) {
        probabilitiesContainer.innerHTML += `
            <div class="mt-1">
                <p class="small text-muted mb-0 text-truncate">${probabilities[i].ClassName}</p>
                <div class="progress me-2" style="height: 3px;">
                    <div class="progress-bar" role="progressbar" 
                        style="width:${probabilities[i].Probability.toFixed(2)*100}%">
                    </div>
                </div>
            </div>
        `;
    }
}