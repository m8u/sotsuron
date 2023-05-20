package evolution

import (
	"fmt"
	"github.com/m8u/goro/pkg/v1/layer"
	m "github.com/m8u/goro/pkg/v1/model"
	"golang.org/x/exp/rand"
	"sotsuron/internal/utils"
	"testing"
	"time"
)

func Test_generateRandomStructure(t *testing.T) {
	rand.Seed(uint64(time.Now().UnixNano()))

	type args struct {
		inputWidth  int
		inputHeight int
		numClasses  int
	}
	type test struct {
		name       string
		args       args
		wantLayers []layer.Config
	}
	var tests []test
	for i := 0; i < 100; i++ {
		inputWidth := 3 + rand.Intn(100)
		inputHeight := 3 + rand.Intn(100)
		numClasses := 2 + rand.Intn(20)
		tests = append(tests,
			test{
				name: fmt.Sprintf("%dx%dx%d", numClasses, inputWidth, inputHeight),
				args: args{
					inputWidth:  inputWidth,
					inputHeight: inputHeight,
					numClasses:  numClasses,
				},
				wantLayers: nil,
			})
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			//t.Parallel()
			layers := generateRandomStructure(tt.args.inputWidth, tt.args.inputHeight, tt.args.numClasses, true)
			model, err := m.NewSequential("")
			utils.MaybeCrash(err)
			model.AddLayers(layers...)
			err = model.Compile(
				m.NewInput("x", []int{1, 1, tt.args.inputHeight, tt.args.inputWidth}),
				m.NewInput("y", []int{1, tt.args.numClasses}),
			)
			if err != nil {
				t.Error(err)
			}
		})
	}
}
