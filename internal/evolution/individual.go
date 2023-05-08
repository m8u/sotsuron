package evolution

import (
	"errors"
	"fmt"
	"github.com/aunum/gold/pkg/v1/common/num"
	"github.com/aunum/gold/pkg/v1/dense"
	"github.com/aunum/goro/pkg/v1/layer"
	m "github.com/aunum/goro/pkg/v1/model"
	"github.com/aunum/log"
	"github.com/google/uuid"
	"golang.org/x/exp/rand"
	"gorgonia.org/tensor"
	"sotsuron/internal/utils"
)

const (
	batchSize      = 100
	mutationChance = 0.3
)

var activationFns = []layer.ActivationFn{
	layer.Linear,
	layer.Sigmoid,
	//layer.Softmax,
	layer.Tanh,
	layer.ReLU,
	layer.LeakyReLU,
}

type Individual struct {
	name string
	*m.Sequential
	inputRes   resolution
	numClasses int
	fitness    float32
}

func NewIndividual(inputWidth, inputHeight, numClasses int) (individual *Individual) {
	name := uuid.New().String()
	model, _ := m.NewSequential(name) // TODO: specify metrics
	model.AddLayers(generateRandomStructure(inputWidth, inputHeight, numClasses)...)
	err := model.Compile(
		m.NewInput("x", []int{1, 3, inputHeight, inputWidth}),
		m.NewInput("y", []int{1, numClasses}),
		m.WithBatchSize(batchSize),
	)
	utils.MaybeCrash(err)
	individual = &Individual{
		name:       name,
		Sequential: model,
		inputRes:   resolution{inputWidth, inputHeight},
		numClasses: numClasses,
	}
	return
}

func (individual *Individual) evaluate(x, y tensor.Tensor, batchSize int) (accuracy, loss float32, err error) {
	exampleSize := x.Shape()[0]
	batches := exampleSize / batchSize
	var accuracies []float32
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
		err = xi.Reshape(batchSize, 3, individual.inputRes.height, individual.inputRes.width) // TODO: this
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

func (individual *Individual) CalculateFitness(xTrain, yTrain, xTest, yTest tensor.Tensor) (fitness float32, err error) {
	epochs := 10
	exampleSize := xTrain.Shape()[0]
	batches := exampleSize / batchSize

	var accuracy, loss float32

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
			utils.MaybeCrash(err)                                                                 // todo maybe replace with return
			err = xi.Reshape(batchSize, 3, individual.inputRes.height, individual.inputRes.width) // TODO: this
			utils.MaybeCrash(err)

			yi, err := yTrain.Slice(dense.MakeRangedSlice(start, end))
			utils.MaybeCrash(err)
			err = yi.Reshape(batchSize, individual.numClasses)
			utils.MaybeCrash(err)

			err = individual.FitBatch(xi, yi)
			utils.MaybeCrash(err)
			err = individual.Tracker.LogStep(epoch, batch)
			utils.MaybeCrash(err)
		}
		accuracy, loss, err = individual.evaluate(xTest, yTest, batchSize)
		utils.MaybeCrash(err)
		//log.Infof("completed train epoch %v with accuracy %v and loss %v", epoch, accuracy, loss)
	}
	err = individual.Tracker.Clear()
	return accuracy + 1/loss, err
}

func (individual *Individual) Mutate() error {
	// get a slice of layers of a model
	layers := make([]layer.Config, len(individual.Chain.Layers))
	copy(layers, individual.Chain.Layers)

	input := (individual.Sequential.X().Inputs()[0].Shape())[2:]
	res := resolution{
		width:  input[1],
		height: input[0],
	}

	// mutate layers (basically replace with new random ones)
	for i := 0; i < len(layers)-1; i++ {
		if _, ok := layers[i].(layer.Flatten); ok {
			continue
		}
		// don't forget to update the 1st dense layer regardless of mutationChance
		if fc, ok := layers[i].(layer.FC); ok && i >= 3 {
			if conv2D, ok := layers[i-3].(layer.Conv2D); ok {
				fc.Input = conv2D.Output * res.width * res.height
				layers[i] = fc
			}
		}

		if rand.Float32() < mutationChance {
			println("----------------------------- mutating layer", i, "-----------------------------")
			if _, ok := layers[i].(layer.Conv2D); ok {
				prevOutput := 3
				if i > 0 {
					prevOutput = layers[i-2].(layer.Conv2D).Output
				}
				conv2D, err := generateRandomConv2D(prevOutput, res, layers[i+1:]...)
				if err != nil {
					return err
				}
				layers[i] = conv2D
				// update input of next Conv2D layer, if any
				if nextConv2D, ok := layers[i+2].(layer.Conv2D); ok {
					nextConv2D.Input = conv2D.Output
					layers[i+2] = nextConv2D
				}
				res = res.after(conv2D)
			} else if _, ok := layers[i].(layer.MaxPooling2D); ok {
				maxPooling2D, err := generateRandomMaxPooling2D(res, layers[i+1:]...)
				if err != nil {
					return err
				}
				layers[i] = maxPooling2D
				res = res.after(maxPooling2D)
			} else if fc, ok := layers[i].(layer.FC); ok {
				fc = generateRandomFC(fc.Input)
				layers[i] = fc
				// update input of next dense layer
				nextFC := layers[i+1].(layer.FC)
				nextFC.Input = fc.Output
				layers[i+1] = nextFC
			}
		} else {
			res = res.after(layers[i])
		}
	}
	// create new model with mutated layers
	mutatedModel, _ := m.NewSequential(uuid.New().String())
	mutatedModel.AddLayers(layers...)
	err := mutatedModel.Compile(individual.X(), individual.Y(), m.WithBatchSize(batchSize))
	individual.Sequential = mutatedModel

	return err
}

type CrossoverFailedError struct {
	recoverData any
}

func (err *CrossoverFailedError) Error() string {
	return fmt.Sprintf("crossover failed: %v", err.recoverData)
}

func (individual *Individual) Crossover(other *Individual) (child1, child2 *Individual, err1, err2 error) {
	findFirstFCIndex := func(layers []layer.Config) int {
		firstFCIndex := len(layers) - 1
		for ; ; firstFCIndex-- {
			if _, ok := layers[firstFCIndex-1].(layer.Flatten); ok {
				return firstFCIndex
			}
		}
	}

	findNextConv2DIndex := func(layers []layer.Config, startIndex int) int {
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

	getPrevOutput := func(layers []layer.Config, startIndex int) int {
		lastConv2DIndex := startIndex
		for i := startIndex - 1; i >= 0; i-- {
			if _, ok := layers[i].(layer.Conv2D); ok {
				lastConv2DIndex = i
				break
			}
		}
		if lastConv2DIndex == startIndex {
			return 3
		}
		return layers[lastConv2DIndex].(layer.Conv2D).Output
	}

	// update inputs of layers at crossover points
	updateInputs := func(layers []layer.Config, crossoverPoint int) error {
		if conv2D, ok := layers[crossoverPoint].(layer.Conv2D); ok {
			if crossoverPoint == 0 {
				conv2D.Input = 3
			} else {
				conv2D.Input = getPrevOutput(layers, crossoverPoint)
			}
			layers[crossoverPoint] = conv2D
		} else if _, ok := layers[crossoverPoint].(layer.MaxPooling2D); ok {
			nextConv2DIndex := findNextConv2DIndex(layers, crossoverPoint)
			if nextConv2DIndex != -1 {
				prevOutput := getPrevOutput(layers, crossoverPoint)
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
		res := (&resolution{input[1], input[0]}).afterMany(layers[:firstFCIndex-1])
		fc := layers[firstFCIndex].(layer.FC)
		fc.Input = getPrevOutput(layers, firstFCIndex-1) * res.width * res.height
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
	err1 = child1Model.Compile(individual.X(), individual.Y(), m.WithBatchSize(batchSize))
	err2 = child2Model.Compile(individual.X(), individual.Y(), m.WithBatchSize(batchSize))

	return &Individual{
			name:       child1Name,
			Sequential: child1Model,
			inputRes:   individual.inputRes,
			numClasses: individual.numClasses,
		}, &Individual{
			name:       child2Name,
			Sequential: child2Model,
			inputRes:   individual.inputRes,
			numClasses: individual.numClasses,
		},
		err1, err2
}
