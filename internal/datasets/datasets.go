package datasets

import (
	"fmt"
	"github.com/aunum/gold/pkg/v1/common/require"
	"golang.org/x/exp/rand"
	"gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
	"image"
	_ "image/jpeg"
	_ "image/png"
	"log"
	"os"
	"time"
)

type Dataset struct {
	x, y       tensor.Tensor
	classNames []string
}

func LoadDataset(path string) (dataset *Dataset, err error) { // TODO: asyncify
	dataset = &Dataset{}
	classes, err := os.ReadDir(path)
	if err != nil {
		return nil, err
	}
	var images []image.Image
	var labels []int
	var sampleWidth, sampleHeight int
	channels := 3
	rows := 0
	start := time.Now()

	// take sample width & height
	classPath := fmt.Sprintf("%s/%s", path, classes[0].Name())
	entities, err := os.ReadDir(classPath)
	require.NoError(err)
	sampleFile, err := os.Open(fmt.Sprintf("%s/%s", classPath, entities[0].Name()))
	require.NoError(err)
	sampleImg, _, err := image.Decode(sampleFile)
	require.NoError(err)
	sampleWidth, sampleHeight = sampleImg.Bounds().Dx(), sampleImg.Bounds().Dy()

	// load images and labels
	for i, class := range classes {
		className := class.Name()
		//log.Println(className)
		classPath := fmt.Sprintf("%s/%s", path, className)
		entities, err := os.ReadDir(classPath)
		require.NoError(err)
		rows += len(entities)
		dataset.classNames = append(dataset.classNames, className)
		for _, entity := range entities {
			file, err := os.Open(fmt.Sprintf("%s/%s", classPath, entity.Name()))
			require.NoError(err)
			img, _, err := image.Decode(file)
			require.NoError(err)
			images = append(images, img)
			labels = append(labels, i)
		}
	}

	// shuffle
	rand.Seed(uint64(time.Now().UnixNano()))
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
				xBacking = append(xBacking, float32(r)/0xffff, float32(g)/0xffff, float32(b)/0xffff)
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
	xTrain, err = dataset.x.Slice(gorgonia.S(0, int(float32(rows)*ratio)))
	if err != nil {
		return
	}
	yTrain, err = dataset.y.Slice(gorgonia.S(0, int(float32(rows)*ratio)))
	if err != nil {
		return
	}
	xTest, err = dataset.x.Slice(gorgonia.S(int(float32(rows)*ratio), rows))
	if err != nil {
		return
	}
	yTest, err = dataset.y.Slice(gorgonia.S(int(float32(rows)*ratio), rows))
	if err != nil {
		return
	}
	return
}

func (dataset *Dataset) NumClasses() int {
	return len(dataset.classNames)
}
