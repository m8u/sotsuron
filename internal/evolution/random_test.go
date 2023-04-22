package evolution

import (
	"fmt"
	"github.com/aunum/goro/pkg/v1/layer"
	m "github.com/aunum/goro/pkg/v1/model"
	"golang.org/x/exp/rand"
	"sotsuron/internal/utils"
	"testing"
	"time"
)

type generateRandomStructureTest struct {
	name       string
	args       generateRandomStructureArgs
	wantLayers []layer.Config
}

type generateRandomStructureArgs struct {
	inputWidth  int
	inputHeight int
	numClasses  int
}

func Test_generateRandomStructure(t *testing.T) {
	rand.Seed(uint64(time.Now().Unix()))
	var tests []generateRandomStructureTest
	for i := 0; i < 100; i++ {
		inputWidth := 3 + rand.Intn(100)
		inputHeight := 3 + rand.Intn(100)
		numClasses := 2 + rand.Intn(20)
		tests = append(tests,
			generateRandomStructureTest{
				name: fmt.Sprintf("%dx%dx%d", numClasses, inputWidth, inputHeight),
				args: generateRandomStructureArgs{
					inputWidth:  inputWidth,
					inputHeight: inputHeight,
					numClasses:  numClasses,
				},
				wantLayers: nil,
			})
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			layers := generateRandomStructure(tt.args.inputWidth, tt.args.inputHeight, tt.args.numClasses)
			model, err := m.NewSequential("")
			utils.MaybeCrash(err)
			model.AddLayers(layers...)
			err = model.Compile(
				m.NewInput("x", []int{1, 3, tt.args.inputHeight, tt.args.inputWidth}),
				m.NewInput("y", []int{1, tt.args.numClasses}),
			)
			if err != nil {
				t.Error(err)
			}
		})
	}
}
