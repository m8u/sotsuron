package datasets

import (
	"gorgonia.org/tensor"
	"reflect"
	"testing"
)

func TestDataset_SplitTrainTest(t *testing.T) {
	type fields struct {
		x          tensor.Tensor
		y          tensor.Tensor
		classNames []string
	}
	type args struct {
		ratio float32
	}
	tests := []struct {
		name       string
		fields     fields
		args       args
		wantXTrain tensor.Tensor
		wantYTrain tensor.Tensor
		wantXTest  tensor.Tensor
		wantYTest  tensor.Tensor
		wantErr    bool
	}{
		{
			name: "test1",
			fields: fields{
				x: tensor.New(tensor.WithShape(4, 1*2*2), tensor.WithBacking([]int{
					1, 2, 3, 4,
					5, 6, 7, 8,
					9, 10, 11, 12,
					13, 14, 15, 16,
				})),
				y: tensor.New(tensor.WithShape(4, 4), tensor.WithBacking([]int{
					1, 0, 0, 0,
					0, 1, 0, 0,
					0, 0, 1, 0,
					0, 0, 0, 1,
				})),
			},
			args: args{0.8},
			wantXTrain: tensor.New(tensor.WithShape(3, 1*2*2), tensor.WithBacking([]int{
				1, 2, 3, 4,
				5, 6, 7, 8,
				9, 10, 11, 12,
			})),
			wantYTrain: tensor.New(tensor.WithShape(3, 4), tensor.WithBacking([]int{
				1, 0, 0, 0,
				0, 1, 0, 0,
				0, 0, 1, 0,
			})),
			wantXTest: tensor.New(tensor.WithShape(1, 1*2*2), tensor.WithBacking([]int{
				13, 14, 15, 16,
			})),
			wantYTest: tensor.New(tensor.WithShape(1, 4), tensor.WithBacking([]int{
				0, 0, 0, 1,
			})),
			wantErr: false,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dataset := &Dataset{
				x:          tt.fields.x,
				y:          tt.fields.y,
				classNames: tt.fields.classNames,
			}
			gotXTrain, gotYTrain, gotXTest, gotYTest, err := dataset.SplitTrainTest(tt.args.ratio)
			if (err != nil) != tt.wantErr {
				t.Errorf("SplitTrainTest() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if !reflect.DeepEqual(gotXTrain.Data(), tt.wantXTrain.Data()) {
				t.Errorf("SplitTrainTest() gotXTrain = %v, want %v", gotXTrain.Data(), tt.wantXTrain.Data())
			}
			if !reflect.DeepEqual(gotYTrain.Data(), tt.wantYTrain.Data()) {
				t.Errorf("SplitTrainTest() gotYTrain = %v, want %v", gotYTrain.Data(), tt.wantYTrain.Data())
			}
			if !reflect.DeepEqual(gotXTest.Data(), tt.wantXTest.Data()) {
				t.Errorf("SplitTrainTest() gotXTest = %v, want %v", gotXTest.Data(), tt.wantXTest.Data())
			}
			if !reflect.DeepEqual(gotYTest.Data(), tt.wantYTest.Data()) {
				t.Errorf("SplitTrainTest() gotYTest = %v, want %v", gotYTest.Data(), tt.wantYTest.Data())
			}
		})
	}
}
