package evolution

import (
	"gorgonia.org/tensor"
	"sotsuron/internal/utils"
)

type Species struct {
	individuals []*Individual
}

func NewSpecies(numIndividuals, inputWidth, inputHeight, numClasses int) *Species {
	kind := &Species{}
	for i := 0; i < numIndividuals; i++ {
		kind.individuals = append(kind.individuals, NewIndividual(inputWidth, inputHeight, numClasses))
	}
	return kind
}

func (species *Species) Evolve(numGenerations int, xTrain, yTrain, xTest, yTest tensor.Tensor) {
	for i := 0; i < numGenerations; i++ {
		for _, individual := range species.individuals {
			fitness, err := individual.CalculateFitness(xTrain, yTrain, xTest, yTest)
			utils.MaybeCrash(err)
			individual.fitness = fitness
		}
	}
}
