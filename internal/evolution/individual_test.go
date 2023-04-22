package evolution

import (
	"fmt"
	m "github.com/aunum/goro/pkg/v1/model"
	"golang.org/x/exp/rand"
	"reflect"
	"testing"
)

func TestIndividual_Mutate(t *testing.T) {
	//rand.Seed(uint64(time.Now().UnixNano()))
	rand.Seed(0)
	type fields struct {
		inputWidth, inputHeight, numClasses int
	}
	type test struct {
		name    string
		fields  fields
		wantErr error
	}
	var tests []test
	for i := 0; i < 100; i++ {
		inputWidth := 3 + rand.Intn(100)
		inputHeight := 3 + rand.Intn(100)
		numClasses := 2 + rand.Intn(20)
		tests = append(tests, test{
			name:    fmt.Sprintf("%dx%dx%d", numClasses, inputWidth, inputHeight),
			fields:  fields{inputWidth, inputHeight, numClasses},
			wantErr: nil,
		})
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			individual := &Individual{
				Sequential: NewIndividual(tt.fields.inputWidth, tt.fields.inputHeight, tt.fields.numClasses).Sequential,
			}
			err := individual.Mutate()
			if !reflect.DeepEqual(err, tt.wantErr) {
				t.Errorf("Mutate() error = %v, wantErr %v", err, tt.wantErr)
			}
		})
	}
}

func TestIndividual_Crossover(t *testing.T) {
	type fields struct {
		Sequential *m.Sequential
	}
	type args struct {
		other *Individual
	}
	tests := []struct {
		name   string
		fields fields
		args   args
	}{
		// TODO: Add test cases.
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			individual := &Individual{
				Sequential: tt.fields.Sequential,
			}
			individual.Crossover(tt.args.other)
		})
	}
}
