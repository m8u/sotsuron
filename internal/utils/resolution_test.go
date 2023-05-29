package utils

import (
	"fmt"
	"github.com/m8u/goro/pkg/v1/layer"
	m "github.com/m8u/goro/pkg/v1/model"
	"golang.org/x/exp/rand"
	"reflect"
	"sotsuron/internal/evolution"
	"testing"
	"time"
)

func Test_resolution_after(t *testing.T) {
	type fields struct {
		width  int
		height int
	}
	type args struct {
		config layer.Config
	}
	tests := []struct {
		name         string
		fields       fields
		args         args
		wantResAfter Resolution
	}{
		{
			name: "conv2d_1",
			fields: fields{
				width:  28,
				height: 28,
			},
			args: args{
				config: layer.Conv2D{
					Width:  3,
					Height: 3,
					Stride: []int{1, 1},
					Pad:    []int{1, 1},
				},
			},
			wantResAfter: Resolution{
				Width:  28,
				Height: 28,
			},
		},
		{
			name: "conv2d_2",
			fields: fields{
				width:  34,
				height: 42,
			},
			args: args{
				config: layer.Conv2D{
					Width:  2,
					Height: 14,
					Stride: []int{1, 1},
					Pad:    []int{2, 2},
				},
			},
			wantResAfter: Resolution{
				Width:  37,
				Height: 33,
			},
		},
		{
			name: "maxpooling2d_1",
			fields: fields{
				width:  28,
				height: 28,
			},
			args: args{
				config: layer.MaxPooling2D{
					Kernel: []int{2, 4},
					Pad:    []int{2, 2},
					Stride: []int{1, 1},
				},
			},
			wantResAfter: Resolution{
				Width:  29,
				Height: 31,
			},
		},
		{
			name: "maxpooling2d_2",
			fields: fields{
				width:  34,
				height: 42,
			},
			args: args{
				config: layer.MaxPooling2D{
					Kernel: []int{2, 4},
					Pad:    []int{2, 2},
					Stride: []int{1, 1},
				},
			},
			wantResAfter: Resolution{
				Width:  35,
				Height: 45,
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			res := &Resolution{
				Width:  tt.fields.width,
				Height: tt.fields.height,
			}
			if gotResAfter := res.After(tt.args.config); !reflect.DeepEqual(gotResAfter, tt.wantResAfter) {
				t.Errorf("after() = %v, want %v", gotResAfter, tt.wantResAfter)
			}
		})
	}
}

func Test_resolution_before(t *testing.T) {
	rand.Seed(uint64(time.Now().UnixNano()))

	type args struct {
		config layer.Config
	}
	type test struct {
		name          string
		resAfter      Resolution
		args          args
		wantResBefore Resolution
	}
	var tests []test
	for i := 0; i < 50; i++ {
		// use resolution.after to test resolution.before
		res := Resolution{
			Width:  3 + rand.Intn(100),
			Height: 3 + rand.Intn(100),
		}
		conv2D, _ := evolution.GenerateRandomConv2D(evolution.DefaultAdvancedConfig(), 10, res)
		resAfter := res.After(conv2D)
		tests = append(tests, test{
			name:          "conv2d",
			resAfter:      resAfter,
			args:          args{config: conv2D},
			wantResBefore: res,
		})

		res = Resolution{
			Width:  3 + rand.Intn(100),
			Height: 3 + rand.Intn(100),
		}
		maxPooling2D, _ := evolution.GenerateRandomMaxPooling2D(evolution.DefaultAdvancedConfig(), res)
		resAfter = res.After(maxPooling2D)
		tests = append(tests, test{
			name:          "maxpooling2d",
			resAfter:      resAfter,
			args:          args{config: maxPooling2D},
			wantResBefore: res,
		})
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if gotResBefore := tt.resAfter.before(tt.args.config); !reflect.DeepEqual(gotResBefore, tt.wantResBefore) {
				t.Errorf("before() = %v, want %v", gotResBefore, tt.wantResBefore)
			}
		})
	}
}

func Test_resolution_beforeMany(t *testing.T) {
	rand.Seed(uint64(time.Now().UnixNano()))

	type args struct {
		layers []layer.Config
	}
	type test struct {
		name string
		args args
	}
	var tests []test

	for i := 0; i < 100; i++ {
		inputWidth := 3 + rand.Intn(100)
		inputHeight := 3 + rand.Intn(100)
		var layers []layer.Config
	generateUntilHasConv2D:
		for {
			layers = evolution.GenerateRandomStructure(evolution.DefaultAdvancedConfig(), inputWidth, inputHeight, 2, true)
			for _, l := range layers {
				if _, ok := l.(layer.Conv2D); ok {
					break generateUntilHasConv2D
				}
			}
		}
		tests = append(tests, test{
			name: fmt.Sprintf("%dx%dx%d", 2, inputWidth, inputHeight),
			args: args{layers},
		})
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			for _, l := range tt.args.layers {
				fmt.Println(l)
			}
			fmt.Println("...")
			minResolution := (&Resolution{3, 3}).CalculateMinRequiredBefore(tt.args.layers)
			fmt.Println(minResolution)

			for i, l := range tt.args.layers {
				if fc, ok := l.(layer.FC); ok {
					newRes := minResolution.AfterMany(tt.args.layers[:i])
					fc.Input = tt.args.layers[i-3].(layer.Conv2D).Output * newRes.Width * newRes.Height
					tt.args.layers[i] = fc
					break
				}
			}

			// try to compile a model with input of min resolution
			model, _ := m.NewSequential("")
			model.AddLayers(tt.args.layers...)
			err := model.Compile(
				m.NewInput("x", []int{1, 1, minResolution.Height, minResolution.Width}),
				m.NewInput("y", []int{1, 2}),
			)
			if err != nil {
				t.Errorf("error compiling model: %v", err)
			}

			// try to compile with input of min resolution - 1
			model, _ = m.NewSequential("")
			model.AddLayers(tt.args.layers...)
			err = model.Compile(
				m.NewInput("x", []int{1, 1, minResolution.Height - 3, minResolution.Width - 3}),
				m.NewInput("y", []int{1, 2}),
			)
			if err == nil {
				t.Errorf("expected error compiling model with input of min resolution - 1")
			}
		})
	}
}
