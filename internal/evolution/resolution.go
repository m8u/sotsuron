package evolution

import (
	"github.com/aunum/goro/pkg/v1/layer"
	"gorgonia.org/tensor"
)

type resolution struct {
	width, height int
}

func (res *resolution) validate() bool {
	return res.width >= minResolutionWidth && res.height >= minResolutionHeight
}

func (res *resolution) after(config layer.Config) (resAfter resolution) {
	if conv2D, ok := config.(layer.Conv2D); ok {
		resAfter = resolution{
			(res.width-conv2D.Width+2*conv2D.Pad[1])/conv2D.Stride[1] + 1,
			(res.height-conv2D.Height+2*conv2D.Pad[0])/conv2D.Stride[0] + 1,
		}
	} else if maxPooling2D, ok := config.(layer.MaxPooling2D); ok {
		resAfter = resolution{
			(res.width-maxPooling2D.Kernel[1]+2*maxPooling2D.Pad[1])/maxPooling2D.Stride[1] + 1,
			(res.height-maxPooling2D.Kernel[0]+2*maxPooling2D.Pad[0])/maxPooling2D.Stride[0] + 1,
		}
	} else {
		resAfter = *res
	}
	return
}

func (res *resolution) afterMany(layers []layer.Config) (resAfter resolution) {
	resAfter = *res
	for i := 0; i < len(layers); i++ {
		resAfter = resAfter.after(layers[i])
	}
	return
}

func (res *resolution) before(config layer.Config) (resBefore resolution) {
	if conv2D, ok := config.(layer.Conv2D); ok {
		resBefore = resolution{
			(res.width-1)*conv2D.Stride[1] - 2*conv2D.Pad[1] + conv2D.Width,
			(res.height-1)*conv2D.Stride[0] - 2*conv2D.Pad[0] + conv2D.Height,
		}
	} else if maxPooling2D, ok := config.(layer.MaxPooling2D); ok {
		resBefore = resolution{
			(res.width-1)*maxPooling2D.Stride[1] - 2*maxPooling2D.Pad[1] + maxPooling2D.Kernel[1],
			(res.height-1)*maxPooling2D.Stride[0] - 2*maxPooling2D.Pad[0] + maxPooling2D.Kernel[0],
		}
	} else {
		resBefore = *res
	}
	return
}

func (res *resolution) calculateMinOutputResolution(layers []layer.Config) (minResolution resolution) {
	minResolution = *res
	for i := len(layers) - 1; i >= 0; i-- {
		minResolution = minResolution.before(layers[i])
		minResolution.width = tensor.MaxInt(minResolution.width, minResolutionWidth)
		minResolution.height = tensor.MaxInt(minResolution.height, minResolutionHeight)
	}
	return
}
