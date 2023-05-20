package evolution

import (
	"fmt"
	"gorgonia.org/tensor"
	"gorgonia.org/vecf32"
	"log"
	"sotsuron/internal/datasets"
	"sotsuron/internal/utils"
)

type Species struct {
	individuals []*Individual
}

func NewSpecies(numIndividuals, inputWidth, inputHeight, numClasses int, grayscale bool) *Species {
	kind := &Species{}
	for i := 0; i < numIndividuals; i++ {
		kind.individuals = append(kind.individuals, NewIndividual(inputWidth, inputHeight, numClasses, grayscale))
	}
	return kind
}

func (species *Species) selectTwoBest() (*Individual, *Individual) {
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
	fmt.Printf("Best fitnesses: %v\n", bestFitnesses)
	return bestIndividuals[0], bestIndividuals[1]
}

func (species *Species) Evolve(numGenerations int, xTrain, yTrain, xTest, yTest tensor.Tensor) (best *Individual) {
	var err error

	toyTestImages := make([]tensor.Tensor, 10)
	toyTestImages[0], err = datasets.LoadImage("/home/m8u/Downloads/mnist_toy_test/zero.png", species.individuals[0].isGrayscale)
	toyTestImages[1], err = datasets.LoadImage("/home/m8u/Downloads/mnist_toy_test/one.png", species.individuals[0].isGrayscale)
	toyTestImages[2], err = datasets.LoadImage("/home/m8u/Downloads/mnist_toy_test/two.png", species.individuals[0].isGrayscale)
	toyTestImages[3], err = datasets.LoadImage("/home/m8u/Downloads/mnist_toy_test/three.png", species.individuals[0].isGrayscale)
	toyTestImages[4], err = datasets.LoadImage("/home/m8u/Downloads/mnist_toy_test/four.png", species.individuals[0].isGrayscale)
	toyTestImages[5], err = datasets.LoadImage("/home/m8u/Downloads/mnist_toy_test/five.png", species.individuals[0].isGrayscale)
	toyTestImages[6], err = datasets.LoadImage("/home/m8u/Downloads/mnist_toy_test/six.png", species.individuals[0].isGrayscale)
	toyTestImages[7], err = datasets.LoadImage("/home/m8u/Downloads/mnist_toy_test/seven.png", species.individuals[0].isGrayscale)
	toyTestImages[8], err = datasets.LoadImage("/home/m8u/Downloads/mnist_toy_test/eight.png", species.individuals[0].isGrayscale)
	toyTestImages[9], err = datasets.LoadImage("/home/m8u/Downloads/mnist_toy_test/nine.png", species.individuals[0].isGrayscale)
	utils.MaybeCrash(err)

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
					fmt.Println("WARNING:", err.Error())
				}
				individual.fitness <- fitness
			}()
		}
		// find 2 best individuals
		parent1, parent2 := species.selectTwoBest()
		//=======================================================================================
		// print layers of the best individual
		for _, layer := range parent1.Chain.Layers {
			fmt.Printf("%+v\n", layer)
		}
		for i, img := range toyTestImages {
			pred, _ := parent1.Predict(img)
			maxIndex := vecf32.Argmax(pred.Data().([]float32))
			fmt.Printf("Predicted: %d, Actual: %d\n", maxIndex, i)
		}
		//=======================================================================================

		if i == numGenerations-1 {
			return parent1
		}
		// create new generation, starting with the best individual from previous
		newGeneration := []*Individual{parent1}
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
	return
}
