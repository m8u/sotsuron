package datasets

import (
	"gorgonia.org/tensor"
	"image"
	"os"
)

func LoadImage(path string) (tensor.Tensor, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	img, _, err := image.Decode(file)
	if err != nil {
		return nil, err
	}
	width, height := img.Bounds().Dx(), img.Bounds().Dy()
	channels := 3
	var backing []float32
	for y := 0; y < height; y++ {
		for x := 0; x < width; x++ {
			r, g, b, _ := img.At(x, y).RGBA()
			backing = append(backing, float32(r)/0xffff, float32(g)/0xffff, float32(b)/0xffff)
		}
	}

	return tensor.New(tensor.WithShape(1, channels, width, height), tensor.WithBacking(backing)), nil
}
