package main

import (
	"context"
	"fmt"
	"github.com/m8u/goro/pkg/v1/layer"
	"github.com/wailsapp/wails/v2/pkg/runtime"
	"gorgonia.org/tensor"
	goRuntime "runtime"
	"sotsuron/internal/datasets"
	"sotsuron/internal/evolution"
	"sotsuron/internal/utils"
	"strings"
)

// App struct
type App struct {
	ctx     context.Context
	dataset *datasets.Dataset
	species *evolution.Species
	testImg tensor.Tensor
}

// NewApp creates a new App application struct
func NewApp() *App {
	return &App{}
}

// startup is called when the app starts. The context is saved
// so we can call the runtime methods
func (a *App) startup(ctx context.Context) {
	a.ctx = ctx
}

func (a *App) LoadDataset(grayscale bool) *datasets.DatasetInfo {
	path, err := runtime.OpenDirectoryDialog(a.ctx, runtime.OpenDialogOptions{
		Title: "Укажите путь к датасету",
	})
	utils.MaybeCrash(err)
	if path == "" {
		return nil
	}
	a.dataset, err = datasets.LoadDataset(path, grayscale)
	if err != nil {
		runtime.EventsEmit(a.ctx, "error", "Не удалось загрузить датасет")
		return nil
	}
	return a.dataset.GetInfo()
}

func (a *App) Evolve(advCfg evolution.AdvancedConfig, trainTestRatio float32, numIndividuals, numGenerations int) {
	datasetInfo := a.dataset.GetInfo()
	a.species = evolution.NewSpecies(
		advCfg,
		numIndividuals,
		datasetInfo.Resolution.Width,
		datasetInfo.Resolution.Height,
		datasetInfo.NumClasses,
		datasetInfo.Grayscale,
	)
	xTrain, yTrain, xTest, yTest, err := a.dataset.SplitTrainTest(trainTestRatio)
	utils.MaybeCrash(err)

	shouldStop := false
	progressChan := make(chan evolution.Progress)
	go func() {
		for {
			progress := <-progressChan
			fmt.Println("GENERATION", progress.Generation)
			runtime.EventsEmit(a.ctx, "evo-progress", progress)
			if progress.Generation == -1 {
				return
			}
		}
	}()
	allChartChan := make(chan evolution.AllChartData)
	go func() {
		for !shouldStop {
			data := <-allChartChan
			runtime.EventsEmit(a.ctx, "evo-all-chart", data)
		}
	}()
	bestChartChan := make(chan float32)
	go func() {
		for !shouldStop {
			data := <-bestChartChan
			runtime.EventsEmit(a.ctx, "evo-best-chart", data)
		}
	}()
	bestLayersChan := make(chan []layer.Config)
	go func() {
		for !shouldStop {
			bestLayers := <-bestLayersChan
			if bestLayers == nil {
				return
			}
			runtime.EventsEmit(a.ctx, "evo-best-layers", evolution.SimplifyLayers(bestLayers))
		}
	}()
	ctx, cancel := context.WithCancel(a.ctx)
	runtime.EventsOnce(a.ctx, "evo-abort", func(optionalData ...interface{}) {
		cancel()
	})

	a.species.Evolve(ctx, advCfg, numGenerations, xTrain, yTrain, xTest, yTest, progressChan, allChartChan, bestChartChan, bestLayersChan)
	fmt.Println("Evolution finished (backend)")
	shouldStop = true
	close(progressChan)
	close(allChartChan)
	close(bestChartChan)
	close(bestLayersChan)
}

func (a *App) LoadImage() (loadedFilename string) {
	path, err := runtime.OpenFileDialog(a.ctx, runtime.OpenDialogOptions{
		Title: "Укажите путь к изображению",
	})
	utils.MaybeCrash(err)
	if path == "" {
		return ""
	}
	a.testImg, err = datasets.LoadImage(path, a.dataset.GetInfo().Grayscale)
	utils.MaybeCrash(err)

	if goRuntime.GOOS == "windows" {
		return strings.Split(path, "\\")[len(strings.Split(path, "\\"))-1]
	}
	return strings.Split(path, "/")[len(strings.Split(path, "/"))-1]
}

func (a *App) Predict() []evolution.ClassProbability {
	probabilities, err := a.species.Best().Predict(a.testImg, a.dataset.ClassNames())
	utils.MaybeCrash(err)
	fmt.Println(probabilities)
	return probabilities
}
