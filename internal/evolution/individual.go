package evolution

import (
	"github.com/aunum/goro/pkg/v1/layer"
	m "github.com/aunum/goro/pkg/v1/model"
	"github.com/google/uuid"
	"golang.org/x/exp/rand"
	"sotsuron/internal/utils"
)

const (
	maxConvMaxPoolingPairs = 3

	maxConvOutput     = 16
	maxConvKernelSize = 16
	maxConvPad        = 2
	maxConvStride     = 1

	maxPoolKernelSize = 16
	maxPoolPad        = 2
	maxPoolStride     = 1

	maxDenseLayers = 3
	maxDenseSize   = 1024

	minResolutionWidth  = 3
	minResolutionHeight = 3

	mutationChance = 0.1
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
	*m.Sequential
}

func NewIndividual(inputWidth, inputHeight, numClasses int) (individual *Individual) {
	model, _ := m.NewSequential(uuid.New().String()) // TODO: specify metrics
	model.AddLayers(generateRandomStructure(inputWidth, inputHeight, numClasses)...)
	err := model.Compile(
		m.NewInput("x", []int{1, 3, inputHeight, inputWidth}),
		m.NewInput("y", []int{1, numClasses}),
	)
	utils.MaybeCrash(err)
	individual = &Individual{model}

	return
}

func (individual *Individual) Mutate() error { // TODO: resolution.beforeMany я написал вчера чтобы вычислять ограничения на размеры слоев
	// get a slice of layers of a model
	layers := make([]layer.Config, len(individual.Chain.Layers))
	copy(layers, individual.Chain.Layers)

	input := (individual.Sequential.X().Inputs()[0].Shape())[2:]
	res := resolution{
		width:  input[0],
		height: input[1],
	}
	var newRes resolution

	// mutate layers (basically replace with new random ones)
	for i := 0; i < len(layers)-2; i++ {
		if _, ok := layers[i].(layer.Flatten); ok {
			continue
		}

		if rand.Float32() < mutationChance {
			println("----------------------------- mutating layer", i, "-----------------------------")
			if _, ok := layers[i].(layer.Conv2D); ok {
				prevOutput := 3
				if i > 0 {
					prevOutput = layers[i-2].(layer.Conv2D).Output
				}
				for {
					conv2D := generateRandomConv2D(prevOutput, res)
					if newRes = res.after(conv2D); !newRes.validate() {
						continue
					}
					layers[i] = conv2D
					// update input of next layer
					if nextConv2D, ok := layers[i+2].(layer.Conv2D); ok {
						nextConv2D.Input = conv2D.Output
						layers[i+2] = nextConv2D
					} else if nextFC, ok := layers[i+2].(layer.FC); ok {
						newRes = newRes.after(layers[i+1].(layer.MaxPooling2D))
						nextFC.Input = conv2D.Output * newRes.width * newRes.height
						layers[i+2] = nextFC
					}
					break
				}
			} else if _, ok := layers[i].(layer.MaxPooling2D); ok {
				maxPooling2D := generateRandomMaxPooling2D(res)
				if newRes = res.after(maxPooling2D); !newRes.validate() {
					continue
				}
				layers[i] = maxPooling2D
				// update input of next layer if it is a dense layer
				if nextFC, ok := layers[i+2].(layer.FC); ok {
					nextFC.Input = layers[i-1].(layer.Conv2D).Output * newRes.width * newRes.height
					layers[i+2] = nextFC
				}
			} else if fc, ok := layers[i].(layer.FC); ok {
				layers[i] = generateRandomFC(fc.Input)
				// update input of next dense layer
				nextFC := layers[i+2].(layer.FC)
				nextFC.Input = fc.Output
				layers[i+2] = nextFC
			}
		}
	}
	// create new model with mutated layers
	mutatedModel, _ := m.NewSequential(uuid.New().String())
	mutatedModel.AddLayers(layers...)
	err := mutatedModel.Compile(individual.X(), individual.Y())
	individual.Sequential = mutatedModel

	return err
}

// todo make a function that adds or deletes a layer (pair)

func (individual *Individual) Crossover(other *Individual) {
	// get slices of layers of both models
	layersLeft := make([]layer.Config, len(individual.Chain.Layers))
	layersRight := make([]layer.Config, len(other.Chain.Layers))
	copy(layersLeft, individual.Chain.Layers)
	copy(layersRight, other.Chain.Layers)

	// pick a random crossover point
	crossoverPoint := rand.Intn(len(layersLeft))

	// swap layers
	for i := crossoverPoint; i < len(layersLeft); i++ {
		layersLeft[i], layersRight[i] = layersRight[i], layersLeft[i]
	}

	// update inputs of layers at crossover point if they are Conv2D and not the first layers
	if conv2D, ok := layersLeft[crossoverPoint].(layer.Conv2D); ok && crossoverPoint > 0 {
		prevConv2D := layersLeft[crossoverPoint-2].(layer.Conv2D)
		conv2D.Input = prevConv2D.Output
		layersLeft[crossoverPoint] = conv2D

		conv2D = layersRight[crossoverPoint].(layer.Conv2D)
		prevConv2D = layersRight[crossoverPoint-2].(layer.Conv2D)
		conv2D.Input = prevConv2D.Output
		layersRight[crossoverPoint] = conv2D
	}
}
