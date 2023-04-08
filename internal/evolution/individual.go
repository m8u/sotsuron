package evolution

import (
	"github.com/aunum/goro/pkg/v1/layer"
	m "github.com/aunum/goro/pkg/v1/model"
	"github.com/google/uuid"
	"golang.org/x/exp/rand"
	"log"
	"math"
)

const (
	maxConvMaxPoolingPairs = 1

	maxConvOutput       = 16
	maxConvKernelWidth  = 4
	maxConvKernelHeight = 4
	maxConvPad          = 2
	maxConvStride       = 3

	maxPoolKernelWidth  = 4
	maxPoolKernelHeight = 4
	maxPoolPad          = 2
	maxPoolStride       = 3

	maxDenseLayers = 1
	maxDenseSize   = 1024
)

var activationFns = []layer.ActivationFn{
	layer.Linear,
	layer.Sigmoid,
	layer.Softmax,
	layer.Tanh,
	layer.ReLU,
	layer.LeakyReLU,
}

type Individual struct {
	*m.Sequential
}

func NewIndividual() (individual *Individual) {
	model, _ := m.NewSequential(uuid.New().String()) // TODO: specify metrics
	individual = &Individual{model}
	return
}

func generateRandomStructure(inputWidth, inputHeight, numClasses int) (layers []layer.Config) {
	rand.Seed(0)
	// 2, : Impossible width/kernel/pad combination
	// 1, 3, 4, : index out of range [3] with length 3
	// 5, 6, 7: Failed to infer shape. Op: A × B: Inner dimensions do not match up (35 != 63)

	// append some Conv2D-MaxPooling2D pairs with random parameters
	numConvPaxPoolingPairs := 1 + rand.Intn(maxConvMaxPoolingPairs)
	prevOutput := 3
	finalResolution := struct{ width, height int }{inputWidth, inputHeight}
	for i := 0; i < numConvPaxPoolingPairs; i++ {
		convOutput := 1 + rand.Intn(maxConvOutput)
		convHeight := 1 + rand.Intn(int(math.Min(maxConvKernelHeight, float64(finalResolution.height))))
		convWidth := 1 + rand.Intn(int(math.Min(maxConvKernelWidth, float64(finalResolution.width))))
		convActivation := activationFns[rand.Intn(len(activationFns))]
		convPad := rand.Intn(maxConvPad + 1)
		//convStride := 1 + rand.Intn(int(math.Min(
		//	maxConvStride,
		//	math.Min(float64(finalResolution.width), float64(finalResolution.height)),
		//)))
		finalResolution.width = (finalResolution.width - convWidth + 2*convPad) + 1
		finalResolution.height = (finalResolution.height - convHeight + 2*convPad) + 1

		poolHeight := 1 + rand.Intn(int(math.Min(maxPoolKernelHeight, float64(finalResolution.height))))
		poolWidth := 1 + rand.Intn(int(math.Min(maxPoolKernelWidth, float64(finalResolution.width))))
		poolPad := rand.Intn(maxPoolPad + 1)
		//poolStride := 1 + rand.Intn(int(math.Min(
		//	maxPoolStride,
		//	math.Min(float64(finalResolution.width), float64(finalResolution.height)),
		//)))
		finalResolution.width = (finalResolution.width - poolWidth + 2*poolPad) + 1
		finalResolution.height = (finalResolution.height - poolHeight + 2*poolPad) + 1

		layers = append(layers,
			layer.Conv2D{
				Input:      prevOutput,
				Output:     convOutput,
				Height:     convHeight,
				Width:      convWidth,
				Activation: convActivation,
				Pad:        []int{convPad, convPad},
				//Stride:     []int{convStride, convStride},
			},
			layer.MaxPooling2D{
				Kernel: []int{poolWidth, poolHeight},
				Pad:    []int{poolPad, poolPad},
				Stride: []int{1, 1},
			},
		)
		prevOutput = convOutput
	}
	log.Println(finalResolution)
	log.Println(prevOutput)

	layers = append(layers, layer.Flatten{})

	// append dense layers
	numDenseLayers := 1 + rand.Intn(maxDenseLayers)
	prevOutput = prevOutput * finalResolution.width * finalResolution.height
	for i := 0; i < numDenseLayers; i++ {
		denseSize := 1 + rand.Intn(maxDenseSize)
		denseActivation := activationFns[rand.Intn(len(activationFns))]
		layers = append(layers,
			layer.FC{
				Input:      prevOutput,
				Output:     denseSize,
				Activation: denseActivation,
			},
		)
		prevOutput = denseSize
	}
	// add last layer of size = num classes
	denseActivation := activationFns[rand.Intn(len(activationFns))]
	layers = append(layers,
		layer.FC{
			Input:      prevOutput,
			Output:     numClasses,
			Activation: denseActivation,
		},
	)

	return
}
