package evolution

import (
	"fmt"
	"gorgonia.org/tensor"
	"log"
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

func (species *Species) SelectTwoBest() (*Individual, *Individual) {
	bestFitnesses := make([]float32, 2)
	bestIndividuals := make([]*Individual, 2)
	for _, individual := range species.individuals {
		fitness := <-individual.fitness
		if fitness > bestFitnesses[0] {
			bestFitnesses[0], bestIndividuals[0] = fitness, individual
		} else if fitness > bestFitnesses[1] {
			bestFitnesses[1], bestIndividuals[1] = fitness, individual
		}
	}
	// print best fitnesses
	fmt.Printf("Best fitnesses: %v\n", bestFitnesses)
	return bestIndividuals[0], bestIndividuals[1]
}

func (species *Species) Evolve(numGenerations int, xTrain, yTrain, xTest, yTest tensor.Tensor) {
	var err error
	for i := 0; i < numGenerations; i++ {
		fmt.Printf("===================================== Generation %d =====================================\n", i)
		// calculate fitness for each individual
		for _, individual := range species.individuals {
			//fmt.Printf("Calculating fitness for %v\n", individual.name)
			// print individual's structure
			//for _, layer := range individual.Chain.Layers {
			//	fmt.Printf("%v\n", layer)
			//}
			individual := individual
			go func() {
				fitness, err := individual.CalculateFitnessBatch(xTrain, yTrain, xTest, yTest)
				if err != nil {
					fmt.Println("WARNING WARNING WARNING WARNING WARNING WARNING WARNING WARNING WARNING ", err.Error())
				}
				individual.fitness <- fitness
			}()
		}
		// find 2 best individuals
		parent1, parent2 := species.SelectTwoBest()
		// create new generation
		var newGeneration []*Individual
		// crossover
		child1, child2, err1, err2 := parent1.Crossover(parent2)
		if err1 != nil && err2 != nil {
			// if crossover fails, just copy the best 2 individuals
			child1, err = parent1.Mutate()
			utils.MaybeCrash(err)
			child2, err = parent2.Mutate()
			utils.MaybeCrash(err)
		}
		newGeneration = append(newGeneration, child1, child2)
		// mutate N times to fill the rest of new generation
		var mutated *Individual
		for i := 0; len(newGeneration) < len(species.individuals) && i < len(species.individuals)*2; i++ {
			if i%2 == 0 {
				mutated, err = child1.Mutate()
			} else {
				mutated, err = child2.Mutate()
			}
			if err == nil {
				newGeneration = append(newGeneration, mutated)
			}
		}
		if len(newGeneration) < len(species.individuals) {
			log.Fatalln("Not enough individuals for a new generation")
		}
		species.individuals = newGeneration
	}
}
