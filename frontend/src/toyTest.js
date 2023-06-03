import {LoadImage, Predict} from "../wailsjs/go/main/App";

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
