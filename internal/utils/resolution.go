package utils

import (
	"fmt"
	"github.com/m8u/goro/pkg/v1/layer"
	"gorgonia.org/tensor"
)

type Resolution struct {
	Width, Height int
}

func (res *Resolution) String() string {
	return fmt.Sprintf("%dx%d", res.Width, res.Height)
}

func (res *Resolution) Validate(minResWidth, minResHeight int) bool {
	return res.Width >= minResWidth && res.Height >= minResHeight
}

func (res *Resolution) After(config layer.Config) (resAfter Resolution) {
	if conv2D, ok := config.(layer.Conv2D); ok {
		resAfter = Resolution{
			(res.Width-conv2D.Width+2*conv2D.Pad[1])/conv2D.Stride[1] + 1,
			(res.Height-conv2D.Height+2*conv2D.Pad[0])/conv2D.Stride[0] + 1,
		}
	} else if maxPooling2D, ok := config.(layer.MaxPooling2D); ok {
		resAfter = Resolution{
			(res.Width-maxPooling2D.Kernel[1]+2*maxPooling2D.Pad[1])/maxPooling2D.Stride[1] + 1,
			(res.Height-maxPooling2D.Kernel[0]+2*maxPooling2D.Pad[0])/maxPooling2D.Stride[0] + 1,
		}
	} else {
		resAfter = *res
	}
	return
}

func (res *Resolution) AfterMany(layers []layer.Config) (resAfter Resolution) {
	resAfter = *res
	for i := 0; i < len(layers); i++ {
		resAfter = resAfter.After(layers[i])
	}
	return
}

func (res *Resolution) before(config layer.Config) (resBefore Resolution) {
	if conv2D, ok := config.(layer.Conv2D); ok {
		resBefore = Resolution{
			(res.Width-1)*conv2D.Stride[1] - 2*conv2D.Pad[1] + conv2D.Width,
			(res.Height-1)*conv2D.Stride[0] - 2*conv2D.Pad[0] + conv2D.Height,
		}
	} else if maxPooling2D, ok := config.(layer.MaxPooling2D); ok {
		resBefore = Resolution{
			(res.Width-1)*maxPooling2D.Stride[1] - 2*maxPooling2D.Pad[1] + maxPooling2D.Kernel[1],
			(res.Height-1)*maxPooling2D.Stride[0] - 2*maxPooling2D.Pad[0] + maxPooling2D.Kernel[0],
		}
	} else {
		resBefore = *res
	}
	return
}

func (res *Resolution) CalculateMinRequiredBefore(layers []layer.Config) (minRes Resolution) {
	minRes = *res
	for i := len(layers) - 1; i >= 0; i-- {
		minRes = minRes.before(layers[i])
		minRes.Width = tensor.MaxInt(minRes.Width, res.Width)
		minRes.Height = tensor.MaxInt(minRes.Height, res.Height)
	}
	return
}
