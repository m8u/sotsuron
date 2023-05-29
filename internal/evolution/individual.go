package evolution

import (
	"context"
	"errors"
	"fmt"
	"github.com/google/uuid"
	"github.com/m8u/gold/pkg/v1/common/num"
	"github.com/m8u/gold/pkg/v1/dense"
	"github.com/m8u/goro/pkg/v1/layer"
	m "github.com/m8u/goro/pkg/v1/model"
	"golang.org/x/exp/rand"
	"gonum.org/v1/gonum/stat"
	"gorgonia.org/tensor"
	"sotsuron/internal/utils"
	"time"
)

type Individual struct {
	name string
	*m.Sequential
	inputRes    utils.Resolution
	isGrayscale bool
	numClasses  int
	fitness     float32
	trained     bool
	lives       int
}

func NewIndividual(advCfg AdvancedConfig, inputWidth, inputHeight, numClasses int, grayscale bool) (individual *Individual) {
	name := uuid.New().String()
	model, _ := m.NewSequential(name) // TODO: specify metrics
	model.AddLayers(GenerateRandomStructure(advCfg, inputWidth, inputHeight, numClasses, grayscale)...)
	var channels int
	if grayscale {
		channels = 1
	} else {
		channels = 3
	}
	err := model.Compile(
		m.NewInput("x", []int{1, channels, inputHeight, inputWidth}),
		m.NewInput("y", []int{1, numClasses}),
		m.WithBatchSize(advCfg.BatchSize),
	)
	utils.MaybeCrash(err)
	individual = &Individual{
		name:        name,
		Sequential:  model,
		inputRes:    utils.Resolution{Width: inputWidth, Height: inputHeight},
		isGrayscale: grayscale,
		numClasses:  numClasses,
		lives:       1,
	}
	return
}

func (individual *Individual) evaluateBatch(ctx context.Context, x, y tensor.Tensor, batchSize int) (accuracy, loss float32, err error) {
	exampleSize := x.Shape()[0]
	batches := exampleSize / batchSize

	var channels int
	if individual.isGrayscale {
		channels = 1
	} else {
		channels = 3
	}

	var accuracies []float32
	for batch := 0; batch < batches; batch++ {
		select {
		case <-ctx.Done():
			return 0, 0, context.Canceled
		default:
		}

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
		err = xi.Reshape(batchSize, channels, individual.inputRes.Height, individual.inputRes.Width)
		utils.MaybeCrash(err)

		yi, err := y.Slice(dense.MakeRangedSlice(start, end))
		utils.MaybeCrash(err)
		err = yi.Reshape(batchSize, individual.numClasses)
		utils.MaybeCrash(err)

		yHat, err := individual.PredictBatch(xi)
		utils.MaybeCrash(err)

		acc, err := calculateAccuracy(yHat.(*tensor.Dense), yi.(*tensor.Dense))
		utils.MaybeCrash(err)
		accuracies = append(accuracies, acc)
	}
	lossVal, err := individual.Tracker.GetValue(fmt.Sprintf("%s_train_batch_loss", individual.name))
	utils.MaybeCrash(err)
	loss = float32(lossVal.Scalar())
	accuracy = num.Mean(accuracies)
	return
}

func calculateAccuracy(yHat, y tensor.Tensor) (accuracy float32, err error) {
	yMax, err := y.(*tensor.Dense).Argmax(1)
	utils.MaybeCrash(err)

	yHatMax, err := yHat.(*tensor.Dense).Argmax(1)
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

func (individual *Individual) CalculateFitnessBatch(ctx context.Context, advCfg AdvancedConfig, xTrain, yTrain, xTest, yTest tensor.Tensor) (fitness float32, err error) {
	epochs := 5
	exampleSize := xTrain.Shape()[0]
	batches := exampleSize / advCfg.BatchSize

	var channels int
	if individual.isGrayscale {
		channels = 1
	} else {
		channels = 3
	}

	var evalStartTime time.Time
	var evalDurations []float64
	var accuracy, loss float32

	for epoch := 0; epoch < epochs; epoch++ {
		for batch := 0; batch < batches; batch++ {
			select {
			case <-ctx.Done():
				return -1, context.Canceled
			default:
			}

			start := batch * advCfg.BatchSize
			end := start + advCfg.BatchSize
			if start >= exampleSize {
				break
			}
			if end > exampleSize {
				end = exampleSize
			}

			xi, err := xTrain.Slice(dense.MakeRangedSlice(start, end))
			utils.MaybeCrash(err) // todo maybe replace with return
			err = xi.Reshape(advCfg.BatchSize, channels, individual.inputRes.Height, individual.inputRes.Width)
			utils.MaybeCrash(err)

			yi, err := yTrain.Slice(dense.MakeRangedSlice(start, end))
			utils.MaybeCrash(err)
			err = yi.Reshape(advCfg.BatchSize, individual.numClasses)
			utils.MaybeCrash(err)

			err = individual.FitBatch(xi, yi) // FIXME: runtime error: invalid memory address or nil pointer dereference (goro@v0.1.3/pkg/v1/model/io.go:84)
			if err != nil {
				return -1, err
			}
			err = individual.Tracker.LogStep(epoch, batch) // FIXME: json: unsupported value: NaN
			if err != nil {
				return -1, err
			}
		}
		evalStartTime = time.Now()
		accuracy, loss, err = individual.evaluateBatch(ctx, xTest, yTest, advCfg.BatchSize)
		evalDurations = append(evalDurations, time.Since(evalStartTime).Seconds())
		if !errors.Is(err, context.Canceled) {
			utils.MaybeCrash(err)
		}
		//log.Infof("completed train epoch %v with accuracy %v and loss %v", epoch, accuracy, loss)
	}
	err = individual.Tracker.Clear()
	meanEvalDuration := float32(stat.Mean(evalDurations, nil))
	fmt.Println(individual.name, accuracy, loss, meanEvalDuration)
	return accuracy*1.0 + -1*loss*0.5, err
}

func getPrevOutput(layers []layer.Config, startIndex int, grayscale bool) int {
	lastConv2DIndex := startIndex
	for i := startIndex - 1; i >= 0; i-- {
		if _, ok := layers[i].(layer.Conv2D); ok {
			lastConv2DIndex = i
			break
		}
	}
	if lastConv2DIndex == startIndex {
		if grayscale {
			return 1
		}
		return 3
	}
	return layers[lastConv2DIndex].(layer.Conv2D).Output
}

func findFirstFCIndex(layers []layer.Config) int {
	firstFCIndex := len(layers) - 1
	for ; ; firstFCIndex-- {
		if _, ok := layers[firstFCIndex-1].(layer.Flatten); ok {
			return firstFCIndex
		}
	}
}

func findNextConv2DIndex(layers []layer.Config, startIndex int) int {
	nextConv2DIndex := startIndex
	firstFCIndex := findFirstFCIndex(layers)
	for i := startIndex + 1; i < firstFCIndex; i++ {
		if _, ok := layers[i].(layer.Conv2D); ok {
			nextConv2DIndex = i
			break
		}
	}
	if nextConv2DIndex == startIndex {
		return -1
	}
	return nextConv2DIndex
}

func (individual *Individual) Mutate(advCfg AdvancedConfig, mutationChance float32) (mutated *Individual, err error) {
	// get a slice of layers of a model
	layers := make([]layer.Config, len(individual.Chain.Layers))
	copy(layers, individual.Chain.Layers)

	input := (individual.Sequential.X().Inputs()[0].Shape())[2:]
	res := utils.Resolution{
		Width:  input[1],
		Height: input[0],
	}
	var channels int
	if individual.isGrayscale {
		channels = 1
	} else {
		channels = 3
	}

	// mutate layers (basically replace with new random ones)
	for i := 0; i < len(layers)-1; i++ {
		if _, ok := layers[i].(layer.Flatten); ok {
			continue
		}

		if rand.Float32() < mutationChance {
			//println("----------------------------- mutating layer", i, "-----------------------------")
			if _, ok := layers[i].(layer.Conv2D); ok {
				prevOutput := channels
				if i > 0 {
					prevOutput = getPrevOutput(layers, i, individual.isGrayscale)
				}
				conv2D, err := GenerateRandomConv2D(advCfg, prevOutput, res, layers[i+1:]...)
				if err != nil {
					return nil, err
				}
				layers[i] = conv2D
				// update input of next Conv2D layer, if any
				nextConv2DIndex := findNextConv2DIndex(layers, i)
				if nextConv2DIndex > -1 {
					nextConv2D := layers[nextConv2DIndex].(layer.Conv2D)
					nextConv2D.Input = conv2D.Output
					layers[nextConv2DIndex] = nextConv2D
				}
				res = res.After(conv2D)
			} else if _, ok := layers[i].(layer.MaxPooling2D); ok {
				maxPooling2D, err := GenerateRandomMaxPooling2D(advCfg, res, layers[i+1:]...)
				if err != nil {
					return nil, err
				}
				layers[i] = maxPooling2D
				res = res.After(maxPooling2D)
			} else if fc, ok := layers[i].(layer.FC); ok {
				fc = generateRandomFC(advCfg, fc.Input)
				layers[i] = fc
				// update input of next dense layer
				nextFC := layers[i+1].(layer.FC)
				nextFC.Input = fc.Output
				layers[i+1] = nextFC
			}
		} else {
			res = res.After(layers[i])
		}
	}
	// update input of first FC layer
	firstFCIndex := findFirstFCIndex(layers)
	firstFC := layers[firstFCIndex].(layer.FC)
	prevOutput := getPrevOutput(layers, firstFCIndex, individual.isGrayscale)
	firstFC.Input = prevOutput * res.Width * res.Height
	layers[firstFCIndex] = firstFC

	// create new model with mutated layers
	name := uuid.New().String()
	newModel, _ := m.NewSequential(name)
	// print layers
	//for _, l := range layers {
	//	fmt.Printf("%v\n", l)
	//}
	newModel.AddLayers(layers...)
	err = newModel.Compile(individual.X(), individual.Y(), m.WithBatchSize(advCfg.BatchSize))
	mutated = &Individual{
		name:        name,
		Sequential:  newModel,
		inputRes:    individual.inputRes,
		isGrayscale: individual.isGrayscale,
		numClasses:  individual.numClasses,
		lives:       1,
	}
	return
}

type CrossoverFailedError struct {
	recoverData interface{}
}

func (err *CrossoverFailedError) Error() string {
	return fmt.Sprintf("crossover failed: %v", err.recoverData)
}

func (individual *Individual) Crossover(advCfg AdvancedConfig, other *Individual) (child1, child2 *Individual, err1, err2 error) {
	// get slices of layers of both models
	layersLeft := make([]layer.Config, len(individual.Chain.Layers))
	layersRight := make([]layer.Config, len(other.Chain.Layers))
	copy(layersLeft, individual.Chain.Layers)
	copy(layersRight, other.Chain.Layers)

	// pick a random crossover point within the left model
	crossoverPointLeft := rand.Intn(len(layersLeft))

	// pick a random crossover point within the right model, but avoid illegal crossovers
	var crossoverPointRight int
	firstFCIndex := findFirstFCIndex(layersRight)
	if _, ok := layersLeft[crossoverPointLeft].(layer.Flatten); ok {
		crossoverPointRight = firstFCIndex - 1
	} else if _, ok := layersLeft[crossoverPointLeft].(layer.FC); ok {
		crossoverPointRight = firstFCIndex + rand.Intn(len(layersRight)-firstFCIndex)
	} else {
		crossoverPointRight = rand.Intn(firstFCIndex)
	}
	// swap layers
	glass := make([]layer.Config, len(layersLeft))
	copy(glass, layersLeft)
	layersLeft, layersRight =
		append(layersLeft[:crossoverPointLeft], layersRight[crossoverPointRight:]...),
		append(layersRight[:crossoverPointRight], glass[crossoverPointLeft:]...)

	// update inputs of layers at crossover points
	updateInputs := func(layers []layer.Config, crossoverPoint int) error {
		if conv2D, ok := layers[crossoverPoint].(layer.Conv2D); ok {
			if crossoverPoint == 0 {
				if individual.isGrayscale {
					conv2D.Input = 1
				} else {
					conv2D.Input = 3
				}
			} else {
				conv2D.Input = getPrevOutput(layers, crossoverPoint, individual.isGrayscale)
			}
			layers[crossoverPoint] = conv2D
		} else if _, ok := layers[crossoverPoint].(layer.MaxPooling2D); ok {
			nextConv2DIndex := findNextConv2DIndex(layers, crossoverPoint)
			if nextConv2DIndex != -1 {
				prevOutput := getPrevOutput(layers, crossoverPoint, individual.isGrayscale)
				nextConv2D := layers[nextConv2DIndex].(layer.Conv2D)
				nextConv2D.Input = prevOutput
				layers[nextConv2DIndex] = nextConv2D
			}
		} else if fc, ok := layers[crossoverPoint].(layer.FC); ok {
			if prevFC, ok := layers[crossoverPoint-1].(layer.FC); ok {
				fc.Input = prevFC.Output
				layers[crossoverPoint] = fc
				return nil
			}
		}
		firstFCIndex := findFirstFCIndex(layers)
		input := (individual.Sequential.X().Inputs()[0].Shape())[2:]
		res := (&utils.Resolution{Width: input[1], Height: input[0]}).AfterMany(layers[:firstFCIndex-1])
		fc := layers[firstFCIndex].(layer.FC)
		fc.Input = getPrevOutput(layers, firstFCIndex-1, individual.isGrayscale) * res.Width * res.Height
		if fc.Input == 0 {
			return errors.New("input is 0")
		}
		layers[firstFCIndex] = fc
		return nil
	}
	err1 = updateInputs(layersLeft, crossoverPointLeft)
	err2 = updateInputs(layersRight, crossoverPointRight)
	if err1 != nil || err2 != nil {
		return nil, nil, err1, err2
	}

	defer func() {
		if r := recover(); r != nil {
			err1, err2 = &CrossoverFailedError{r}, &CrossoverFailedError{r}
		}
	}()

	// create new models with swapped layers
	child1Name := uuid.New().String()
	child1Model, _ := m.NewSequential(child1Name)
	child1Model.AddLayers(layersLeft...)
	child2Name := uuid.New().String()
	child2Model, _ := m.NewSequential(child2Name)
	child2Model.AddLayers(layersRight...)

	// compile models
	err1 = child1Model.Compile(individual.X(), individual.Y(), m.WithBatchSize(advCfg.BatchSize))
	err2 = child2Model.Compile(individual.X(), individual.Y(), m.WithBatchSize(advCfg.BatchSize))

	return &Individual{
			name:        child1Name,
			Sequential:  child1Model,
			inputRes:    individual.inputRes,
			isGrayscale: individual.isGrayscale,
			numClasses:  individual.numClasses,
			lives:       1,
		}, &Individual{
			name:        child2Name,
			Sequential:  child2Model,
			inputRes:    individual.inputRes,
			isGrayscale: individual.isGrayscale,
			numClasses:  individual.numClasses,
			lives:       1,
		},
		err1, err2
}
