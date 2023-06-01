package evolution

func SquareShapeSlice(side int) []int {
	return []int{side, side}
}

type Progress struct {
	Generation int
	Individual int
	ETASeconds float64
}

type AllChartData struct {
	Name     string
	Accuracy float32
}
