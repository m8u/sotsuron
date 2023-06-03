import {LoadDataset} from "../wailsjs/go/main/App";

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
