package evolution

import "github.com/aunum/goro/pkg/v1/layer"

type resolution struct {
	width, height int
}

func (res *resolution) validate() bool {
	return res.width >= minResolutionWidth && res.height >= minResolutionHeight
}

func (res *resolution) after(config layer.Config) (newRes resolution) {
	if conv2D, ok := config.(layer.Conv2D); ok {
		newRes = resolution{
			(res.width-conv2D.Width+2*conv2D.Pad[1])/conv2D.Stride[1] + 1,
			(res.height-conv2D.Height+2*conv2D.Pad[0])/conv2D.Stride[0] + 1,
		}
	} else if maxPooling2D, ok := config.(layer.MaxPooling2D); ok {
		newRes = resolution{
			(res.width-maxPooling2D.Kernel[1]+2*maxPooling2D.Pad[1])/maxPooling2D.Stride[1] + 1,
			(res.height-maxPooling2D.Kernel[0]+2*maxPooling2D.Pad[0])/maxPooling2D.Stride[0] + 1,
		}
	}
	return
}
