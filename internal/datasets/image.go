package datasets

import (
	"gorgonia.org/tensor"
	"image"
	"os"
)

func LoadImage(path string, grayscale bool) (tensor.Tensor, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	img, _, err := image.Decode(file)
	if err != nil {
		return nil, err
	}
	width, height := img.Bounds().Dx(), img.Bounds().Dy()
	var channels int
	if grayscale {
		channels = 1
	} else {
		channels = 3
	}
	var backing []float32
	for y := 0; y < height; y++ {
		for x := 0; x < width; x++ {
			r, g, b, _ := img.At(x, y).RGBA()
			if grayscale {
				backing = append(backing, float32(r)/0xffff*0.3+float32(g)/0xffff*0.59+float32(b)/0xffff*0.11)
			} else {
				backing = append(backing, float32(r)/0xffff, float32(g)/0xffff, float32(b)/0xffff)
			}
		}
	}

	return tensor.New(tensor.WithShape(1, channels, width, height), tensor.WithBacking(backing)), nil
}
