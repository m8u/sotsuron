package main

import (
	"context"
	"fmt"
	wails "github.com/wailsapp/wails/v2/pkg/runtime"
	"gorgonia.org/tensor"
	"runtime"
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
	path, err := wails.OpenDirectoryDialog(a.ctx, wails.OpenDialogOptions{
		Title: "Укажите путь к датасету",
	})
	utils.MaybeCrash(err)
	if path == "" {
		return nil
	}
	a.dataset, err = datasets.LoadDataset(path, grayscale)
	if err != nil {
		wails.EventsEmit(a.ctx, "error", "Не удалось загрузить датасет")
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

	ctx, cancel := context.WithCancel(a.ctx)
	wails.EventsOnce(a.ctx, "evo-abort", func(optionalData ...interface{}) {
		cancel()
	})
	progressChan := make(chan evolution.Progress)
	go func() {
		for {
			select {
			case progress := <-progressChan:
				wails.EventsEmit(a.ctx, "evo-progress", progress)
			case <-ctx.Done():
				wails.EventsEmit(a.ctx, "evo-progress", evolution.Progress{
					Generation: -1,
				})
				return
			default:
			}
		}
	}()
	allChartChan := make(chan evolution.AllChartData)
	go func() {
		for {
			select {
			case data := <-allChartChan:
				wails.EventsEmit(a.ctx, "evo-all-chart", data)
			case <-ctx.Done():
				return
			default:
			}
		}
	}()
	bestChartChan := make(chan float32)
	go func() {
		for {
			select {
			case data := <-bestChartChan:
				wails.EventsEmit(a.ctx, "evo-best-chart", data)
			case <-ctx.Done():
				return
			default:
			}
		}
	}()

	a.species.Evolve(ctx, advCfg, numGenerations, xTrain, yTrain, xTest, yTest, progressChan, allChartChan, bestChartChan)
	fmt.Println("Evolution finished (backend)")
	cancel()
	wails.EventsOffAll(a.ctx)
}

func (a *App) LoadImage() (loadedFilename string) {
	path, err := wails.OpenFileDialog(a.ctx, wails.OpenDialogOptions{
		Title: "Укажите путь к изображению",
	})
	utils.MaybeCrash(err)
	if path == "" {
		return ""
	}
	a.testImg, err = datasets.LoadImage(path, a.dataset.GetInfo().Grayscale)
	utils.MaybeCrash(err)

	if runtime.GOOS == "windows" {
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
