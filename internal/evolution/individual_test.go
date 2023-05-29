package evolution

import (
	"errors"
	"fmt"
	"golang.org/x/exp/rand"
	"gorgonia.org/tensor"
	"math"
	"sotsuron/internal/datasets"
	"sotsuron/internal/utils"
	"strings"
	"testing"
	"time"
)

func TestIndividual_Mutate(t *testing.T) {
	rand.Seed(uint64(time.Now().UnixNano()))
	type fields struct {
		inputWidth, inputHeight, numClasses int
	}
	type test struct {
		name   string
		fields fields
	}
	var tests []test
	for i := 0; i < 100; i++ {
		inputWidth := 3 + rand.Intn(100)
		inputHeight := 3 + rand.Intn(100)
		numClasses := 2 + rand.Intn(20)
		tests = append(tests, test{
			name:   fmt.Sprintf("%dx%dx%d", numClasses, inputWidth, inputHeight),
			fields: fields{inputWidth, inputHeight, numClasses},
		})
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			//t.Parallel()
			individual := NewIndividual(DefaultAdvancedConfig(), tt.fields.inputWidth, tt.fields.inputHeight, tt.fields.numClasses, true)
			_, err := individual.Mutate(DefaultAdvancedConfig())
			if errors.As(err, &NoValidConfigFound{}) {
				t.Skipf(err.Error())
			} else if err != nil {
				t.Errorf(err.Error())
			}
		})
	}
}

func TestIndividual_Mutate_repeatedly(t *testing.T) {
	dataset, err := datasets.LoadDataset("/home/m8u/Downloads/mnist_png_ultralight", true)
	utils.MaybeCrash(err)
	xTrain, yTrain, xTest, yTest, err := dataset.SplitTrainTest(0.8)
	utils.MaybeCrash(err)

	individual := NewIndividual(DefaultAdvancedConfig(), 28, 28, 10, true)
	individual.CalculateFitnessBatch(xTrain, yTrain, xTest, yTest)
	individual, _ = individual.Mutate(DefaultAdvancedConfig())
	individual.CalculateFitnessBatch(xTrain, yTrain, xTest, yTest)
	individual, _ = individual.Mutate(DefaultAdvancedConfig())
	individual.CalculateFitnessBatch(xTrain, yTrain, xTest, yTest)
}

func TestIndividual_Crossover(t *testing.T) {
	rand.Seed(uint64(time.Now().UnixNano()))
	type fields struct {
		inputWidth, inputHeight, numClasses int
	}
	type test struct {
		name   string
		fields fields
	}
	var tests []test
	for i := 0; i < 50; i++ {
		inputWidth := 20 + rand.Intn(20)
		inputHeight := 20 + rand.Intn(20)
		numClasses := 2 + rand.Intn(20)
		tests = append(tests, test{
			name:   fmt.Sprintf("%d: %dx%dx%d", i, numClasses, inputWidth, inputHeight),
			fields: fields{inputWidth, inputHeight, numClasses},
		})
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			individual := NewIndividual(DefaultAdvancedConfig(), tt.fields.inputWidth, tt.fields.inputHeight, tt.fields.numClasses, true)
			other := NewIndividual(DefaultAdvancedConfig(), tt.fields.inputWidth, tt.fields.inputHeight, tt.fields.numClasses, true)
			_, _, err1, err2 := individual.Crossover(other)
			if err1 != nil || err2 != nil {
				t.Skipf("could not crossover: child1 error: %v, child2 error: %v", err1, err2)
			}
		})
	}
}

func TestIndividual_CalculateFitness(t *testing.T) {
	rand.Seed(0)
	type fields struct {
		inputWidth, inputHeight, numClasses int
	}
	type args struct {
		xTrain tensor.Tensor
		yTrain tensor.Tensor
		xTest  tensor.Tensor
		yTest  tensor.Tensor
	}
	type test struct {
		name    string
		fields  fields
		args    args
		wantErr bool
	}
	var tests []test
	dataset, err := datasets.LoadDataset("/home/m8u/Downloads/mnist_png_ultralight", true)
	utils.MaybeCrash(err)
	xTrain, yTrain, xTest, yTest, err := dataset.SplitTrainTest(0.8)
	utils.MaybeCrash(err)
	for i := 0; i < 50; i++ {
		tests = append(tests, test{
			name: fmt.Sprintf("%d: %dx%dx%d", i, 10, 28, 28),
			fields: fields{
				inputWidth:  28,
				inputHeight: 28,
				numClasses:  10,
			},
			args: args{
				xTrain: xTrain,
				yTrain: yTrain,
				xTest:  xTest,
				yTest:  yTest,
			},
			wantErr: false,
		})
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			//t.Parallel()
			individual := NewIndividual(DefaultAdvancedConfig(), tt.fields.inputWidth, tt.fields.inputHeight, tt.fields.numClasses, true)
			gotFitness, err := individual.CalculateFitnessBatch(tt.args.xTrain, tt.args.yTrain, tt.args.xTest, tt.args.yTest)
			if (err != nil) != tt.wantErr {
				t.Errorf("CalculateFitnessBatch() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			fmt.Printf("\nCalculateFitnessBatch() gotFitness = %v\n", gotFitness)
			img, err := datasets.LoadImage("/home/m8u/Downloads/two.png", true)
			utils.MaybeCrash(err)
			pred, err := individual.Predict(img)
			utils.MaybeCrash(err)

			progressBars := make([]string, len(pred.Data().([]float32)))
			for i, probability := range pred.Data().([]float32) {
				if probability < 0 || probability > 99999 || math.IsNaN(float64(probability)) {
					probability = 0
				}
				progressBars[i] = fmt.Sprintf("%d: %s", i, strings.Repeat("#", int(probability*100)))
			}
			fmt.Println(strings.Join(progressBars, "\n"))
		})
	}
}

/*
{3 9 3 13  0x1b11680 [0 0] [1 1] [1 1] 0xb73e20}
{(4, 15) [2 2] [1 1] }
{9 2 8 3  0x15ab520 [0 0] [1 1] [1 1] 0xb73e20}
{(5, 2) [2 2] [1 1] }
{}
{280 235  0x1b11680 0xb73bc0 false 0xb73bc0}
{235 151  0x1b11680 0xb73bc0 false 0xb73bc0}
{151 10  0x15ab520 0xb73bc0 false 0xb73bc0}
{7 20}
*/
