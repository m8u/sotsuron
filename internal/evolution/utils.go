package evolution

import "github.com/m8u/goro/pkg/v1/layer"

func SquareShapeSlice(side int) []int {
	return []int{side, side}
}

type Progress struct {
	Generation int
	Individual int
	ETASeconds float64
}

type AllChartData struct {
	Name     string
	Accuracy float32
}

func activationFnToString(activationFn layer.ActivationFn) string {
	switch activationFn.(type) {
	case *layer.LinearActivation:
		return "Linear"
	case *layer.SigmoidActivation:
		return "Sigmoid"
	case *layer.SoftmaxActivation:
		return "Softmax"
	case *layer.TanhActivation:
		return "Tanh"
	case *layer.ReLUActivation:
		return "ReLU"
	case *layer.LeakyReLUActivation:
		return "LeakyReLU"
	default:
		return "Unknown"
	}
}

type simpleLayerConfig interface{}

type simpleConv2D struct {
	Type       string // "Conv2D"
	Input      int
	Output     int
	Height     int
	Width      int
	Activation string
	Pad        int
	Stride     int
}
type simpleMaxPooling2D struct {
	Type   string // "MaxPooling2D"
	Height int
	Width  int
	Pad    int
	Stride int
}
type simpleFC struct {
	Type       string // "FC"
	Input      int
	Output     int
	Activation string
}

func SimplifyLayers(layers []layer.Config) (simplified []simpleLayerConfig) {
	for i, _ := range layers {
		switch layers[i].(type) {
		case layer.Conv2D:
			conv2D := layers[i].(layer.Conv2D)
			simplified = append(simplified, simpleConv2D{
				Type:       "Conv2D",
				Input:      conv2D.Input,
				Output:     conv2D.Output,
				Height:     conv2D.Height,
				Width:      conv2D.Width,
				Activation: activationFnToString(conv2D.Activation),
				Pad:        conv2D.Pad[0],
				Stride:     conv2D.Stride[0],
			})
		case layer.MaxPooling2D:
			maxPooling2D := layers[i].(layer.MaxPooling2D)
			simplified = append(simplified, simpleMaxPooling2D{
				Type:   "MaxPooling2D",
				Height: maxPooling2D.Kernel[0],
				Width:  maxPooling2D.Kernel[1],
				Pad:    maxPooling2D.Pad[0],
				Stride: maxPooling2D.Stride[0],
			})
		case layer.FC:
			fc := layers[i].(layer.FC)
			simplified = append(simplified, simpleFC{
				Type:       "FC",
				Input:      fc.Input,
				Output:     fc.Output,
				Activation: activationFnToString(fc.Activation),
			})
		}
	}
	return simplified
}
