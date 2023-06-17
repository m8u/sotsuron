package evolution

type AdvancedConfig struct {
	Epochs    int
	BatchSize int

	MutationMultiplier float32

	MaxConvMaxPoolingPairs int
	MaxConvOutput          int
	MaxConvKernelSize      int
	MaxConvPad             int
	MaxConvStride          int
	MaxPoolKernelSize      int
	MaxPoolPad             int
	MaxPoolStride          int
	MaxDenseLayers         int
	MaxDenseSize           int
	MinResolutionWidth     int
	MinResolutionHeight    int
}

func DefaultAdvancedConfig() AdvancedConfig {
	return AdvancedConfig{
		Epochs:    5,
		BatchSize: 10,

		MutationMultiplier: 1.0,

		MaxConvMaxPoolingPairs: 3,
		MaxConvOutput:          16,
		MaxConvKernelSize:      16,
		MaxConvPad:             2,
		MaxConvStride:          1,
		MaxPoolKernelSize:      16,
		MaxPoolPad:             2,
		MaxPoolStride:          1,
		MaxDenseLayers:         2,
		MaxDenseSize:           512,
		MinResolutionWidth:     3,
		MinResolutionHeight:    3,
	}
}
