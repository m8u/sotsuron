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
	species evolution.Species
}

// NewApp creates a new App application struct
func NewApp() *App {
	return &App{}
}

// startup is called when the app starts. The context is saved
// so we can call the runtime methods
func (a *App) startup(ctx context.Context) {
	a.ctx = ctx
	runtime.Show(ctx)
	runtime.WindowShow(ctx)
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
	utils.MaybeCrash(err)
	return a.dataset.GetInfo()

}
