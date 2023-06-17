import "bootstrap/dist/css/bootstrap.min.css";
import {Toast} from "bootstrap";
import {EventsOn} from "../wailsjs/runtime";
import {initAllChart, initBestChart} from "./charts";
import {initBestStructureBlock} from "./bestStructure";

window.onload = async function () {
    initAllChart();
    initBestChart();
    initBestStructureBlock();
    resetAdvancedConfig();

    let liveToast = document.querySelector("#live-toast")
    let liveToastBody = document.querySelector("#live-toast-body")
    EventsOn("error", (msg) => {
        liveToastBody.innerHTML = msg;
        const toast = new Toast(liveToast);
        toast.show()
    });
}
