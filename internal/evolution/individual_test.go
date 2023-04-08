package evolution

import (
	"github.com/aunum/goro/pkg/v1/layer"
	m "github.com/aunum/goro/pkg/v1/model"
	"reflect"
	"sotsuron/internal/utils"
	"testing"
)

func TestNewIndividual(t *testing.T) {
	tests := []struct {
		name           string
		wantIndividual *Individual
	}{
		// TODO: Add test cases.
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if gotIndividual := NewIndividual(); !reflect.DeepEqual(gotIndividual, tt.wantIndividual) {
				t.Errorf("NewIndividual() = %v, want %v", gotIndividual, tt.wantIndividual)
			}
		})
	}
}

func Test_generateRandomStructure(t *testing.T) {
	type args struct {
		inputWidth  int
		inputHeight int
		numClasses  int
	}
	tests := []struct {
		name       string
		args       args
		wantLayers []layer.Config
	}{
		{
			name: "test1",
			args: args{
				inputWidth:  28,
				inputHeight: 28,
				numClasses:  10,
			},
			wantLayers: nil,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			layers := generateRandomStructure(tt.args.inputWidth, tt.args.inputHeight, tt.args.numClasses)
			model, err := m.NewSequential("")
			utils.MaybeCrash(err)
			model.AddLayers(layers...)
			err = model.Compile(
				m.NewInput("x", []int{1, 3, tt.args.inputWidth, tt.args.inputHeight}),
				m.NewInput("y", []int{1, tt.args.numClasses}),
			)
			if err != nil {
				t.Error(err)
			}
		})
	}
}
