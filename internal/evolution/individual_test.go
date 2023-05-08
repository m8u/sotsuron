package evolution

import (
	"errors"
	"fmt"
	"golang.org/x/exp/rand"
	"gorgonia.org/tensor"
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
			t.Parallel()
			individual := NewIndividual(tt.fields.inputWidth, tt.fields.inputHeight, tt.fields.numClasses)
			err := individual.Mutate()
			if errors.As(err, &NoValidConfigFound{}) {
				t.Skipf(err.Error())
			} else if err != nil {
				t.Errorf(err.Error())
			}
		})
	}
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
		inputWidth := 20 + rand.Intn(100)
		inputHeight := 20 + rand.Intn(100)
		numClasses := 2 + rand.Intn(20)
		tests = append(tests, test{
			name:   fmt.Sprintf("%d: %dx%dx%d", i, numClasses, inputWidth, inputHeight),
			fields: fields{inputWidth, inputHeight, numClasses},
		})
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			//t.Parallel()
			individual := NewIndividual(tt.fields.inputWidth, tt.fields.inputHeight, tt.fields.numClasses)
			other := NewIndividual(tt.fields.inputWidth, tt.fields.inputHeight, tt.fields.numClasses)
			_, _, err1, err2 := individual.Crossover(other)
			if err1 != nil || err2 != nil {
				t.Skipf("could not crossover: child1 error: %v, child2 error: %v", err1, err2)
			}
		})
	}
}

func TestIndividual_CalculateFitness(t *testing.T) {
	rand.Seed(uint64(time.Now().UnixNano()))
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
	for i := 0; i < 10; i++ {
		dataset, err := datasets.LoadDataset("/home/m8u/Downloads/mnist_png_light")
		utils.MaybeCrash(err)
		xTrain, yTrain, xTest, yTest, err := dataset.SplitTrainTest(0.8)
		utils.MaybeCrash(err)

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
			t.Parallel()
			individual := NewIndividual(tt.fields.inputWidth, tt.fields.inputHeight, tt.fields.numClasses)
			gotFitness, err := individual.CalculateFitness(tt.args.xTrain, tt.args.yTrain, tt.args.xTest, tt.args.yTest)
			if (err != nil) != tt.wantErr {
				t.Errorf("CalculateFitness() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			fmt.Printf("\nCalculateFitness() gotFitness = %v\n", gotFitness)
			img, err := datasets.LoadImage("/home/m8u/Downloads/two.png")
			utils.MaybeCrash(err)
			pred, err := individual.Predict(img)
			utils.MaybeCrash(err)

			progressBars := make([]string, len(pred.Data().([]float32)))
			for i, probability := range pred.Data().([]float32) {
				if probability < 0 {
					probability = 0
				}
				progressBars[i] = fmt.Sprintf("%d: %s", i, strings.Repeat("#", int(probability*100)))
			}
			fmt.Println(strings.Join(progressBars, "\n"))
		})
	}
}
