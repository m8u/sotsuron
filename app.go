package main

import (
	"context"
	"github.com/aunum/gold/pkg/v1/common/num"
	"github.com/aunum/gold/pkg/v1/common/require"
	"github.com/aunum/gold/pkg/v1/dense"
	"github.com/aunum/goro/pkg/v1/layer"
	m "github.com/aunum/goro/pkg/v1/model"
	"github.com/aunum/log"
	g "gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
	"sotsuron/internal/datasets"
	"sotsuron/internal/utils"
)

// App struct
type App struct {
	ctx context.Context
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

// DoStuff does stuff
func (a *App) DoStuff() {
	// load the dataset
	dataset, err := datasets.LoadDataset("/home/m8u/Downloads/mnist_png_ultralight")
	require.NoError(err)

	xTrain, yTrain, xTest, yTest, err := dataset.SplitTrainTest(0.8)
	utils.MaybeCrash(err)

	//create the model
	model, _ := m.NewSequential("mnist")

	//model.AddLayers(
	//	layer.Conv2D{Input: 3, Output: 10, Width: 3, Height: 3},
	//	layer.Conv2D{Input: 10, Output: 10, Width: 3, Height: 3},
	//	layer.MaxPooling2D{Kernel: []int{2, 2}, Stride: []int{2, 2}},
	//	layer.Conv2D{Input: 10, Output: 10, Width: 3, Height: 3},
	//	layer.Conv2D{Input: 10, Output: 10, Width: 3, Height: 3},
	//	layer.MaxPooling2D{Kernel: []int{2, 2}, Stride: []int{2, 2}},
	//	layer.Flatten{},
	//	layer.FC{Input: 10 * 7 * 7, Output: 10, Activation: layer.Softmax},
	//)

	// todo rand.Seed ломает goro? попробовать здесь с теми же параметрами

	model.AddLayers(
		layer.Conv2D{Input: 3, Output: 2, Width: 2, Height: 2, Pad: []int{0, 0}}, // W + 2P - (K-1)
		layer.MaxPooling2D{Kernel: []int{3, 3}, Pad: []int{2, 2}, Stride: []int{1, 1}},
		layer.Flatten{},
		layer.FC{Input: 2 * 29 * 29, Output: 10, Activation: layer.Softmax},
	)
	optimizer := g.NewAdamSolver()
	batchSize := 5
	err = model.Compile(
		m.NewInput("x", []int{1, 3, 28, 28}),
		m.NewInput("y", []int{1, dataset.NumClasses()}),
		m.WithOptimizer(optimizer),
		m.WithLoss(m.CrossEntropy),
		m.WithBatchSize(batchSize),
	)
	utils.MaybeCrash(err)

	// fit the model
	epochs := 10
	exampleSize := xTrain.Shape()[0]
	batches := exampleSize / batchSize

	log.Infov("epochs", epochs)
	for epoch := 0; epoch < epochs; epoch++ {
		for batch := 0; batch < batches; batch++ {
			start := batch * batchSize
			end := start + batchSize
			if start >= exampleSize {
				break
			}
			if end > exampleSize {
				end = exampleSize
			}

			xi, err := xTrain.Slice(dense.MakeRangedSlice(start, end))
			utils.MaybeCrash(err)
			err = xi.Reshape(batchSize, 3, 28, 28)
			utils.MaybeCrash(err)

			yi, err := yTrain.Slice(dense.MakeRangedSlice(start, end))
			utils.MaybeCrash(err)
			err = yi.Reshape(batchSize, 10)
			utils.MaybeCrash(err)

			err = model.FitBatch(xi, yi)
			utils.MaybeCrash(err)
			err = model.Tracker.LogStep(epoch, batch)
			utils.MaybeCrash(err)
		}
		accuracy, loss, err := evaluate(xTest.(*tensor.Dense), yTest.(*tensor.Dense), model, batchSize)
		utils.MaybeCrash(err)
		log.Infof("completed train epoch %v with accuracy %v and loss %v", epoch, accuracy, loss)
	}
	err = model.Tracker.Clear()

	log.Info("========================")
	img, err := datasets.LoadImage("/home/m8u/Downloads/two.png")
	utils.MaybeCrash(err)
	pred, err := model.Predict(img)
	utils.MaybeCrash(err)
	log.Info(pred)
}

func evaluate(x, y *tensor.Dense, model *m.Sequential, batchSize int) (acc, loss float32, err error) {
	exampleSize := x.Shape()[0]
	batches := exampleSize / batchSize

	accuracies := []float32{}
	for batch := 0; batch < batches; batch++ {
		start := batch * batchSize
		end := start + batchSize
		if start >= exampleSize {
			break
		}
		if end > exampleSize {
			end = exampleSize
		}

		xi, err := x.Slice(dense.MakeRangedSlice(start, end))
		utils.MaybeCrash(err)
		xi.Reshape(batchSize, 3, 28, 28)

		yi, err := y.Slice(dense.MakeRangedSlice(start, end))
		utils.MaybeCrash(err)
		yi.Reshape(batchSize, 10)

		yHat, err := model.PredictBatch(xi)
		utils.MaybeCrash(err)

		acc, err := accuracy(yHat.(*tensor.Dense), yi.(*tensor.Dense))
		utils.MaybeCrash(err)
		accuracies = append(accuracies, acc)
	}
	lossVal, err := model.Tracker.GetValue("mnist_train_batch_loss")
	utils.MaybeCrash(err)
	loss = float32(lossVal.Scalar())
	acc = num.Mean(accuracies)
	return
}

func accuracy(yHat, y *tensor.Dense) (float32, error) {
	yMax, err := y.Argmax(1)
	utils.MaybeCrash(err)

	yHatMax, err := yHat.Argmax(1)
	utils.MaybeCrash(err)

	eq, err := tensor.ElEq(yMax, yHatMax, tensor.AsSameType())
	utils.MaybeCrash(err)
	eqd := eq.(*tensor.Dense)

	numTrue, err := eqd.Sum()
	if err != nil {
		return 0, err
	}

	return float32(numTrue.Data().(int)) / float32(eqd.Len()), nil
}
