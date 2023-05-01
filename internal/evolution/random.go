package evolution

import (
	"fmt"
	"github.com/aunum/goro/pkg/v1/layer"
	"golang.org/x/exp/rand"
	"math"
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
)

type NoValidConfigFound struct {
	inputRes     resolution
	minOutputRes resolution
}

func (err NoValidConfigFound) Error() string {
	return fmt.Sprintf(
		"could not find a valid config for inputRes = %s, minOutputRes = %s",
		err.inputRes.String(), err.minOutputRes.String(),
	)
}

// getValidRandomConfig returns a random valid Conv2D or MaxPooling2D config, or an error if none are found.
func getValidRandomConfig(inputRes, minOutputRes resolution) (width, height, pad, stride int, err error) {
	var validConfigs [][]int
	for width = 2; width <= minOutputRes.width; width++ {
		for height = 2; height <= minOutputRes.height; height++ {
			for pad = 0; pad <= maxConvPad; pad++ {
				for stride = 1; stride <= maxConvStride; stride++ {
					outputRes := inputRes.after(layer.Conv2D{
						Height: height,
						Width:  width,
						Pad:    squareShapeSlice(pad),
						Stride: squareShapeSlice(stride),
					})
					if outputRes.width >= minOutputRes.width && outputRes.height >= minOutputRes.height {
						validConfigs = append(validConfigs, []int{width, height, pad, stride})
					}
				}
			}
		}
	}
	if len(validConfigs) > 0 {
		config := validConfigs[rand.Intn(len(validConfigs))]
		return config[0], config[1], config[2], config[3], nil
	}
	return 0, 0, 0, 0, NoValidConfigFound{inputRes, minOutputRes}
}

func generateRandomConv2D(prevOutput int, imageRes resolution, layers ...layer.Config) (layer.Conv2D, error) {
	if len(layers) > 0 {
		minOutputRes := (&resolution{
			minResolutionWidth,
			minResolutionHeight,
		}).calculateMinRequiredBefore(layers)
		width, height, pad, stride, err := getValidRandomConfig(imageRes, minOutputRes)
		if err != nil {
			return layer.Conv2D{}, err
		}
		return layer.Conv2D{
			Input:      prevOutput,
			Output:     1 + rand.Intn(maxConvOutput),
			Height:     height,
			Width:      width,
			Activation: activationFns[rand.Intn(len(activationFns))],
			Pad:        squareShapeSlice(pad),
			Stride:     squareShapeSlice(stride),
		}, nil
	}
	return layer.Conv2D{
		Input:      prevOutput,
		Output:     1 + rand.Intn(maxConvOutput),
		Height:     2 + rand.Intn(int(math.Min(maxConvKernelSize-1, float64(imageRes.height-2)))),
		Width:      2 + rand.Intn(int(math.Min(maxConvKernelSize-1, float64(imageRes.width-2)))),
		Activation: activationFns[rand.Intn(len(activationFns))],
		Pad:        squareShapeSlice(rand.Intn(maxConvPad + 1)),
		Stride: squareShapeSlice(1 + rand.Intn(int(math.Min(
			maxConvStride,
			math.Min(float64(imageRes.width), float64(imageRes.height))-1,
		)))),
	}, nil
}

func generateRandomMaxPooling2D(imageRes resolution, layers ...layer.Config) (layer.MaxPooling2D, error) {
	if len(layers) > 0 {
		minOutputRes := (&resolution{
			minResolutionWidth,
			minResolutionHeight,
		}).calculateMinRequiredBefore(layers)
		width, height, pad, stride, err := getValidRandomConfig(imageRes, minOutputRes)
		if err != nil {
			return layer.MaxPooling2D{}, err
		}
		return layer.MaxPooling2D{
			Kernel: []int{
				height,
				width,
			},
			Pad:    squareShapeSlice(pad),
			Stride: squareShapeSlice(stride),
		}, nil
	}

	return layer.MaxPooling2D{
		Kernel: []int{
			2 + rand.Intn(int(math.Min(maxPoolKernelSize-1, float64(imageRes.height-2)))),
			2 + rand.Intn(int(math.Min(maxPoolKernelSize-1, float64(imageRes.width-2)))), // see gorgonia's nn.go:255
		},
		Pad: squareShapeSlice(rand.Intn(maxPoolPad + 1)),
		Stride: squareShapeSlice(1 + rand.Intn(int(math.Min(
			maxPoolStride,
			math.Min(float64(imageRes.width), float64(imageRes.height))-1,
		)))),
	}, nil
}

func generateRandomFC(prevOutput int) layer.FC {
	return layer.FC{
		Input:      prevOutput,
		Output:     1 + rand.Intn(maxDenseSize),
		Activation: activationFns[rand.Intn(len(activationFns))],
	}
}

func generateRandomStructure(inputWidth, inputHeight, numClasses int) (layers []layer.Config) {
	// append some Conv2D-MaxPooling2D pairs with random parameters
	numConvPaxPoolingPairs := 1 + rand.Intn(maxConvMaxPoolingPairs)
	prevOutput := 3
	res := resolution{inputWidth, inputHeight}
	var newRes resolution
	for i := 0; i < numConvPaxPoolingPairs; i++ {
		conv2D, _ := generateRandomConv2D(prevOutput, res)
		if newRes = res.after(conv2D); !newRes.validate() {
			break
		}
		//fmt.Println(conv2D)

		maxPooling2D, _ := generateRandomMaxPooling2D(newRes)
		if newRes = newRes.after(maxPooling2D); !newRes.validate() {
			break
		}
		//fmt.Println(maxPooling2D)
		res = newRes
		//fmt.Println(res)

		layers = append(layers,
			conv2D,
			maxPooling2D,
		)
		prevOutput = conv2D.Output
	}

	// flatten
	layers = append(layers, layer.Flatten{})

	// append some dense layers
	numDenseLayers := 1 + rand.Intn(maxDenseLayers)
	prevOutput = prevOutput * res.width * res.height
	for i := 0; i < numDenseLayers; i++ {
		fc := generateRandomFC(prevOutput)
		//fmt.Printf("%v -> %v\n", prevOutput, fc.Output)
		layers = append(layers, fc)
		prevOutput = fc.Output
	}

	// append last dense layer of size = num classes
	layers = append(layers,
		layer.FC{
			Input:      prevOutput,
			Output:     numClasses,
			Activation: activationFns[rand.Intn(len(activationFns))],
		},
	)

	return
}
