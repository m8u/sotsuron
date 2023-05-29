package main

import (
	"context"
	"github.com/wailsapp/wails/v2/pkg/runtime"
	"sotsuron/internal/datasets"
	"sotsuron/internal/evolution"
	"sotsuron/internal/utils"
)

// App struct
type App struct {
	ctx     context.Context
	dataset *datasets.Dataset
	species *evolution.Species
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

	progressChan := make(chan evolution.Progress)
	go func() {
		for {
			progress := <-progressChan
			runtime.EventsEmit(a.ctx, "evo-progress", progress)
			if progress.Generation == -1 {
				break
			}
		}
	}()
	ctx, cancel := context.WithCancel(a.ctx)
	runtime.EventsOnce(a.ctx, "evo-abort", func(optionalData ...interface{}) {
		cancel()
	})
	a.species.Evolve(ctx, advCfg, numGenerations, xTrain, yTrain, xTest, yTest, progressChan)
}
