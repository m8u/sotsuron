package datasets

import (
	"fmt"
	"github.com/m8u/gold/pkg/v1/dense"
	"golang.org/x/exp/rand"
	"gorgonia.org/tensor"
	"image"
	_ "image/jpeg"
	_ "image/png"
	"log"
	"os"
	"runtime"
	"sotsuron/internal/utils"
	"strings"
	"time"
)

type Dataset struct {
	path       string
	x, y       tensor.Tensor
	classNames []string
}

func LoadDataset(path string, grayscale bool) (dataset *Dataset, err error) { // TODO: asyncify
	dataset = &Dataset{path: strings.TrimRight(path, "/\\")}
	classes, err := os.ReadDir(path)
	if err != nil {
		return nil, err
	}
	var images []image.Image
	var labels []int
	var sampleWidth, sampleHeight int
	var channels int
	if grayscale {
		channels = 1
	} else {
		channels = 3
	}
	rows := 0
	start := time.Now()

	// take sample width & height
	classPath := fmt.Sprintf("%s/%s", path, classes[0].Name())
	entities, err := os.ReadDir(classPath)
	if err != nil {
		return nil, err
	}
	sampleFile, err := os.Open(fmt.Sprintf("%s/%s", classPath, entities[0].Name()))
	if err != nil {
		return nil, err
	}
	sampleImg, _, err := image.Decode(sampleFile)
	if err != nil {
		return nil, err
	}
	sampleWidth, sampleHeight = sampleImg.Bounds().Dx(), sampleImg.Bounds().Dy()

	// load images and labels
	for i, class := range classes {
		className := class.Name()
		//log.Println(className)
		classPath := fmt.Sprintf("%s/%s", path, className)
		entities, err := os.ReadDir(classPath)
		if err != nil {
			return nil, err
		}
		rows += len(entities)
		dataset.classNames = append(dataset.classNames, className)
		for _, entity := range entities {
			file, err := os.Open(fmt.Sprintf("%s/%s", classPath, entity.Name()))
			if err != nil {
				return nil, err
			}
			img, _, err := image.Decode(file)
			if err != nil {
				return nil, err
			}
			images = append(images, img)
			labels = append(labels, i)
		}
	}

	// shuffle
	rand.Shuffle(len(images), func(i, j int) {
		images[i], images[j] = images[j], images[i]
		labels[i], labels[j] = labels[j], labels[i]
	})

	// prepare buffers
	var xBacking, yBacking []float32
	for i := 0; i < len(images); i++ { // TODO: we can use LoadImage and concat resulting tensors into one
		img := images[i]
		for y := 0; y < sampleHeight; y++ {
			for x := 0; x < sampleWidth; x++ {
				r, g, b, _ := img.At(x, y).RGBA()
				if grayscale {
					xBacking = append(xBacking, float32(r)/0xffff*0.3+float32(g)/0xffff*0.59+float32(b)/0xffff*0.11)
				} else {
					xBacking = append(xBacking, float32(r)/0xffff, float32(g)/0xffff, float32(b)/0xffff)
				}
			}
		}
		for j := 0; j < len(classes); j++ {
			if j == labels[i] {
				yBacking = append(yBacking, 0.9) // TODO: does it make a difference if we use 0.1 & 0.9? Also 0.001...0.999 for Xs
			} else {
				yBacking = append(yBacking, 0.1)
			}
		}
	}

	// create tensors from buffers
	dataset.x = tensor.New(tensor.WithShape(rows, channels, sampleHeight, sampleWidth), tensor.WithBacking(xBacking))
	dataset.y = tensor.New(tensor.WithShape(rows, len(classes)), tensor.WithBacking(yBacking))

	log.Println("load complete!", time.Since(start))
	return
}

func (dataset *Dataset) SplitTrainTest(ratio float32) (xTrain, yTrain, xTest, yTest tensor.Tensor, err error) {
	rows := dataset.x.Shape()[0]
	//fmt.Printf(
	//	"xTrain: (%v, %v)\nyTrain: (%v, %v)\nxTest: (%v, %v)\nyTest: (%v, %v)\n",
	//	0, int(float32(rows)*ratio),
	//	0, int(float32(rows)*ratio),
	//	int(float32(rows)*ratio), rows,
	//	int(float32(rows)*ratio), rows,
	//)
	xTrain, err = dataset.x.Slice(dense.MakeRangedSlice(0, int(float32(rows)*ratio)))
	if err != nil {
		return
	}
	yTrain, err = dataset.y.Slice(dense.MakeRangedSlice(0, int(float32(rows)*ratio)))
	if err != nil {
		return
	}
	xTest, err = dataset.x.Slice(dense.MakeRangedSlice(int(float32(rows)*ratio), rows))
	if err != nil {
		return
	}
	yTest, err = dataset.y.Slice(dense.MakeRangedSlice(int(float32(rows)*ratio), rows))
	if err != nil {
		return
	}
	return
}

type DatasetInfo struct {
	Name                  string
	NumImages, NumClasses int
	Resolution            utils.Resolution
	Grayscale             bool
}

func (dataset *Dataset) GetInfo() *DatasetInfo {
	var split []string
	if runtime.GOOS == "windows" {
		split = strings.Split(dataset.path, "\\")
	} else {
		split = strings.Split(dataset.path, "/")
	}
	return &DatasetInfo{
		Name:       split[len(split)-1],
		NumImages:  dataset.x.Shape()[0],
		NumClasses: dataset.y.Shape()[1],
		Resolution: utils.Resolution{Width: dataset.x.Shape()[3], Height: dataset.x.Shape()[2]},
		Grayscale:  dataset.x.Shape()[1] == 1,
	}
}
