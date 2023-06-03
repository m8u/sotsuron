import "bootstrap/dist/css/bootstrap.min.css";
import {Toast} from "bootstrap";
import {EventsOn} from "../wailsjs/runtime";
import {initAllChart, initBestChart} from "./charts";

function resizeCanvases() {
    const controls = document.querySelector("#controls");
    const height = window.innerHeight - controls.offsetHeight - 30;
    window.allChart.resize(window.allChart.width, height);
    window.bestChart.resize(window.bestChart.width, height);
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
        const toast = new Toast(liveToast);
        toast.show()
    });
}
