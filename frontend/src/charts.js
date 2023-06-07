import {Chart} from "chart.js/auto";

// Chart.defaults.responsive = true;
Chart.defaults.devicePixelRatio = 2;
Chart.defaults.maintainAspectRatio = false;

export function initAllChart(epochs=10) {
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

export function initBestChart(generations=10) {
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

export function updateAllChart(data) {
    let i = window.allChart.data.datasets.findIndex(ds => ds.label === data.Name)
    if (i === -1) {
        window.allChart.data.datasets.push({
            label: data.Name,
            data: [data.Accuracy],
            borderColor: "#"+Math.floor(Math.random()*16777215).toString(16),
            tension: 0.1,
            pointRadius: 1,
        });
    } else {
        window.allChart.data.datasets[i].data.push(data.Accuracy);
    }
    window.allChart.update("none");
}

export function updateBestChart(accuracy) {
    window.bestChart.data.datasets[0].data.push(accuracy);
    window.bestChart.update("none");
}