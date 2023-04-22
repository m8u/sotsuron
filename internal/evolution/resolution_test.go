package evolution

import (
	"github.com/aunum/goro/pkg/v1/layer"
	"reflect"
	"testing"
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
		name       string
		fields     fields
		args       args
		wantNewRes resolution
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
			wantNewRes: resolution{
				width:  28,
				height: 28,
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
			wantNewRes: resolution{
				width:  37,
				height: 33,
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
			wantNewRes: resolution{
				width:  29,
				height: 31,
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
			wantNewRes: resolution{
				width:  35,
				height: 45,
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			res := &resolution{
				width:  tt.fields.width,
				height: tt.fields.height,
			}
			if gotNewRes := res.after(tt.args.config); !reflect.DeepEqual(gotNewRes, tt.wantNewRes) {
				t.Errorf("after() = %v, want %v", gotNewRes, tt.wantNewRes)
			}
		})
	}
}
