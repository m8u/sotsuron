import './bootstrap.min.css'
import './bootstrap.bundle.min.js'
import {LoadDataset} from '../wailsjs/go/main/App';


function resizeCanvas() {
    const canvas = document.getElementById("my-canvas");
    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight * 0.5;
}

window.onload = function () {
    resizeCanvas();
    window.addEventListener('resize', resizeCanvas, false);
}

window.loadDataset = async function(grayscale) {
    let datasetInfo = await LoadDataset(grayscale);
    if (datasetInfo == null) {
        return;
    }
    document.querySelector("#dataset-name").value = datasetInfo.Name;
    document.querySelector("#dataset-name").value = datasetInfo.Name;
    document.querySelector("#dataset-num-classes").value = datasetInfo.NumClasses;
    document.querySelector("#dataset-num-images").value = datasetInfo.NumImages;
    document.querySelector("#dataset-num-channels").value = grayscale ? 1 : 3;
};