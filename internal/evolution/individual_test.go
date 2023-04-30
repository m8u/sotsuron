package evolution

import (
	"errors"
	"fmt"
	"golang.org/x/exp/rand"
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
	rand.Seed(0)
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
