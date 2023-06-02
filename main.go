package main

import (
	"embed"
	"github.com/wailsapp/wails/v2"
	"github.com/wailsapp/wails/v2/pkg/options"
	"github.com/wailsapp/wails/v2/pkg/options/assetserver"
	"golang.org/x/exp/rand"
	"time"
)

//go:embed all:frontend/dist
var assets embed.FS

func main() {
	rand.Seed(uint64(time.Now().UnixNano()))

	// Create an instance of the app structure
	app := NewApp()

	//Create application with options
	err := wails.Run(&options.App{
		Title:     "sotsuron",
		Width:     1200,
		Height:    700,
		MinWidth:  1200,
		MinHeight: 700,
		AssetServer: &assetserver.Options{
			Assets: assets,
		},
		OnStartup: app.startup,
		Bind: []interface{}{
			app,
		},
	})

	if err != nil {
		println("Error:", err.Error())
	}
}
