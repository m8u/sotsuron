package evolution

import (
	"context"
	"sotsuron/internal/datasets"
	"sotsuron/internal/utils"
	"testing"
)

func TestSpecies_Evolve(t *testing.T) {
	type fields struct {
		numIndividuals, inputWidth, inputHeight, numClasses int
		grayscale                                           bool
	}
	type args struct {
		numGenerations int
	}

	tests := []struct {
		name   string
		fields fields
		args   args
	}{
		{
			name: "",
			fields: fields{
				numIndividuals: 10,
				inputWidth:     28,
				inputHeight:    28,
				numClasses:     10,
				grayscale:      true,
			},
			args: args{
				numGenerations: 5,
			},
		},
	}

	dataset, err := datasets.LoadDataset("/home/m8u/code/datasets/mnist_png_light", tests[0].fields.grayscale)
	utils.MaybeCrash(err)
	xTrain, yTrain, xTest, yTest, err := dataset.SplitTrainTest(0.8)
	utils.MaybeCrash(err)

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			species := NewSpecies(DefaultAdvancedConfig(), tt.fields.numIndividuals, tt.fields.inputWidth, tt.fields.inputHeight, tt.fields.numClasses, tt.fields.grayscale)
			species.Evolve(context.Background(), DefaultAdvancedConfig(), tt.args.numGenerations, xTrain, yTrain, xTest, yTest, nil, nil, nil)
		})
	}
}
