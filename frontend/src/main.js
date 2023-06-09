import "bootstrap/dist/css/bootstrap.min.css";
import {Toast} from "bootstrap";
import {EventsOn} from "../wailsjs/runtime";
import {initAllChart, initBestChart} from "./charts";
import {initVisualization} from "./visualization";

function resizeCanvases() {
    const controls = document.querySelector("#controls");
    const height = window.innerHeight - controls.offsetHeight - 70;
    window.allChart.resize(window.allChart.width, height);
    window.bestChart.resize(window.bestChart.width, height);
    if (window.visualizationCanvas) {
        window.visualizationCanvas.width = window.visualizationCanvas.parentElement.clientWidth - 30;
        window.visualizationCanvas.height = height-10;
    }
}

window.onload = async function () {
    initAllChart();
    initBestChart();
    resetAdvancedConfig();

    resizeCanvases();
    window.addEventListener("resize", resizeCanvases, false);

    initVisualization();

    let liveToast = document.querySelector("#live-toast")
    let liveToastBody = document.querySelector("#live-toast-body")
    EventsOn("error", (msg) => {
        liveToastBody.innerHTML = msg;
        const toast = new Toast(liveToast);
        toast.show()
    });
}
