package evolution

import (
	"github.com/aunum/goro/pkg/v1/layer"
	"golang.org/x/exp/rand"
	"math"
)

func generateRandomConv2D(prevOutput int, imageRes resolution) layer.Conv2D {
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
	}
}

func generateRandomMaxPooling2D(imageRes resolution) layer.MaxPooling2D {
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
	}
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
		conv2D := generateRandomConv2D(prevOutput, res)
		if newRes = res.after(conv2D); !newRes.validate() {
			break
		}
		//fmt.Println(conv2D)

		maxPooling2D := generateRandomMaxPooling2D(newRes)
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
