package evolution

import (
	"fmt"
	"github.com/m8u/gorgonia"
	"github.com/m8u/goro/pkg/v1/layer"
	"golang.org/x/exp/rand"
	"gorgonia.org/tensor"
	"math"
	"modernc.org/mathutil"
	"sort"
	"sotsuron/internal/utils"
)

var activationFns = []layer.ActivationFn{
	layer.Linear,
	layer.Sigmoid,
	//layer.Softmax,
	layer.Tanh,
	layer.ReLU,
	layer.LeakyReLU,
}

type NoValidConfigFound struct {
	inputRes     utils.Resolution
	minOutputRes utils.Resolution
}

func (err NoValidConfigFound) Error() string {
	return fmt.Sprintf(
		"could not find a valid config for inputRes = %s, minOutputRes = %s",
		err.inputRes.String(), err.minOutputRes.String(),
	)
}

// getValidRandomConfig returns a random valid Conv2D or MaxPooling2D config, or an error if none are found.
func getValidRandomConfig(advCfg AdvancedConfig, inputRes, minOutputRes utils.Resolution) (width, height, pad, stride int, err error) {
	var validConfigs [][]int
	for width = 2; width <= minOutputRes.Width; width++ {
		for height = 2; height <= minOutputRes.Height; height++ {
			for pad = 0; pad <= advCfg.MaxConvPad; pad++ {
				for stride = 1; stride <= advCfg.MaxConvStride; stride++ {
					outputRes := inputRes.After(layer.Conv2D{
						Height: height,
						Width:  width,
						Pad:    SquareShapeSlice(pad),
						Stride: SquareShapeSlice(stride),
					})
					if outputRes.Width >= minOutputRes.Width && outputRes.Height >= minOutputRes.Height {
						validConfigs = append(validConfigs, []int{width, height, pad, stride})
					}
				}
			}
		}
	}
	if len(validConfigs) > 0 {
		sort.Slice(validConfigs, func(i, j int) bool {
			return validConfigs[i][0]*validConfigs[i][1] < validConfigs[j][0]*validConfigs[j][1]
		})
		config := validConfigs[mathutil.Clamp(int(math.Abs(gorgonia.Gaussian64(0, float64(len(validConfigs)-1))[0])), 0, len(validConfigs)-1)]
		return config[0], config[1], config[2], config[3], nil
	}
	return 0, 0, 0, 0, NoValidConfigFound{inputRes, minOutputRes}
}

func GenerateRandomConv2D(advCfg AdvancedConfig, prevOutput int, imageRes utils.Resolution, layers ...layer.Config) (layer.Conv2D, error) {
	if len(layers) > 0 {
		minOutputRes := (&utils.Resolution{
			Width:  advCfg.MinResolutionWidth,
			Height: advCfg.MinResolutionHeight,
		}).CalculateMinRequiredBefore(layers)
		width, height, pad, stride, err := getValidRandomConfig(advCfg, imageRes, minOutputRes)
		if err != nil {
			return layer.Conv2D{}, err
		}
		return layer.Conv2D{
			Input:      prevOutput,
			Output:     1 + rand.Intn(advCfg.MaxConvOutput),
			Height:     height,
			Width:      width,
			Activation: activationFns[rand.Intn(len(activationFns))].Clone(),
			Pad:        SquareShapeSlice(pad),
			Stride:     SquareShapeSlice(stride),
		}, nil
	}
	return layer.Conv2D{
		Input:      prevOutput,
		Output:     1 + rand.Intn(advCfg.MaxConvOutput),
		Height:     2 + rand.Intn(tensor.MinInt(advCfg.MaxConvKernelSize-1, imageRes.Height-2)),
		Width:      2 + rand.Intn(tensor.MinInt(advCfg.MaxConvKernelSize-1, imageRes.Width-2)),
		Activation: activationFns[rand.Intn(len(activationFns))].Clone(),
		Pad:        SquareShapeSlice(rand.Intn(advCfg.MaxConvPad + 1)),
		Stride: SquareShapeSlice(1 + rand.Intn(tensor.MinInt(
			advCfg.MaxConvStride,
			tensor.MinInt(imageRes.Width, imageRes.Height)-1,
		))),
	}, nil
}

func GenerateRandomMaxPooling2D(advCfg AdvancedConfig, imageRes utils.Resolution, layers ...layer.Config) (layer.MaxPooling2D, error) {
	if len(layers) > 0 {
		minOutputRes := (&utils.Resolution{
			Width:  advCfg.MinResolutionWidth,
			Height: advCfg.MinResolutionHeight,
		}).CalculateMinRequiredBefore(layers)
		width, height, pad, stride, err := getValidRandomConfig(advCfg, imageRes, minOutputRes)
		if err != nil {
			return layer.MaxPooling2D{}, err
		}
		return layer.MaxPooling2D{
			Kernel: []int{
				height,
				width,
			},
			Pad:    SquareShapeSlice(pad),
			Stride: SquareShapeSlice(stride),
		}, nil
	}

	return layer.MaxPooling2D{
		Kernel: []int{
			2 + rand.Intn(int(math.Min(float64(advCfg.MaxPoolKernelSize-1), float64(imageRes.Height-2)))), // todo use int min
			2 + rand.Intn(int(math.Min(float64(advCfg.MaxPoolKernelSize-1), float64(imageRes.Width-2)))),  // see gorgonia's nn.go:255 todo use int min
		},
		Pad: SquareShapeSlice(rand.Intn(advCfg.MaxPoolPad + 1)),
		Stride: SquareShapeSlice(1 + rand.Intn(int(math.Min(
			float64(advCfg.MaxPoolStride), // todo use int min
			math.Min(float64(imageRes.Width), float64(imageRes.Height))-1,
		)))),
	}, nil
}

func generateRandomFC(advCfg AdvancedConfig, prevOutput int) layer.FC {
	return layer.FC{
		Input:      prevOutput,
		Output:     1 + rand.Intn(advCfg.MaxDenseSize),
		Activation: activationFns[rand.Intn(len(activationFns))].Clone(),
	}
}

func GenerateRandomStructure(advCfg AdvancedConfig, inputWidth, inputHeight, numClasses int, grayscale bool) (layers []layer.Config) {
	// append some Conv2D-MaxPooling2D pairs with random parameters
	numConvMaxPoolingPairs := 1 + rand.Intn(advCfg.MaxConvMaxPoolingPairs)
	var prevOutput int
	if grayscale {
		prevOutput = 1
	} else {
		prevOutput = 3
	}
	res := utils.Resolution{Width: inputWidth, Height: inputHeight}
	var newRes utils.Resolution
	for i := 0; i < numConvMaxPoolingPairs; i++ {
		conv2D, _ := GenerateRandomConv2D(advCfg, prevOutput, res)
		if newRes = res.After(conv2D); !newRes.Validate(advCfg.MinResolutionWidth, advCfg.MinResolutionHeight) {
			break
		}
		fmt.Println(conv2D)
		layers = append(layers, conv2D)
		prevOutput = conv2D.Output
		res = newRes

		maxPooling2D, _ := GenerateRandomMaxPooling2D(advCfg, newRes)
		if newRes = newRes.After(maxPooling2D); !newRes.Validate(advCfg.MinResolutionWidth, advCfg.MinResolutionHeight) {
			break
		}
		fmt.Println(maxPooling2D)
		layers = append(layers, maxPooling2D)
		res = newRes
	}

	// flatten
	layers = append(layers, layer.Flatten{})

	// append some dense layers
	numDenseLayers := 1 + rand.Intn(advCfg.MaxDenseLayers)
	prevOutput = prevOutput * res.Width * res.Height
	for i := 0; i < numDenseLayers; i++ {
		fc := generateRandomFC(advCfg, prevOutput)
		fmt.Printf("%v -> %v\n", prevOutput, fc.Output)
		layers = append(layers, fc)
		prevOutput = fc.Output
	}
	fmt.Printf("%v -> %v\n", prevOutput, numClasses)

	// append last dense layer of size = num classes
	layers = append(layers,
		layer.FC{
			Input:      prevOutput,
			Output:     numClasses,
			Activation: activationFns[rand.Intn(len(activationFns))].Clone(),
		},
	)

	return
}
