package evolution

import (
	m "github.com/aunum/goro/pkg/v1/model"
	"testing"
)

func TestIndividual_Mutate(t *testing.T) {
	type fields struct {
		Sequential *m.Sequential
	}
	tests := []struct {
		name   string
		fields fields
	}{
		{
			name: "test1",
			fields: fields{
				Sequential: NewIndividual(28, 28, 10).Sequential,
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			individual := &Individual{
				Sequential: tt.fields.Sequential,
			}
			individual.Mutate()
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
