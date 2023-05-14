package evolution

import (
	"gorgonia.org/tensor"
	"sotsuron/internal/datasets"
	"sotsuron/internal/utils"
	"testing"
)

func TestSpecies_Evolve(t *testing.T) {
	type fields struct {
		numIndividuals, inputWidth, inputHeight, numClasses int
	}
	type args struct {
		numGenerations int
		xTrain         tensor.Tensor
		yTrain         tensor.Tensor
		xTest          tensor.Tensor
		yTest          tensor.Tensor
	}

	dataset, err := datasets.LoadDataset("/home/m8u/Downloads/mnist_png_light")
	utils.MaybeCrash(err)
	xTrain, yTrain, xTest, yTest, err := dataset.SplitTrainTest(0.8)
	utils.MaybeCrash(err)

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
			},
			args: args{
				numGenerations: 3,
				xTrain:         xTrain,
				yTrain:         yTrain,
				xTest:          xTest,
				yTest:          yTest,
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			species := NewSpecies(tt.fields.numIndividuals, tt.fields.inputWidth, tt.fields.inputHeight, tt.fields.numClasses)
			species.Evolve(tt.args.numGenerations, tt.args.xTrain, tt.args.yTrain, tt.args.xTest, tt.args.yTest)
		})
	}
}
