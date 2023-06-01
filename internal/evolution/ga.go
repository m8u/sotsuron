package evolution

import (
	"context"
	"fmt"
	"gorgonia.org/tensor"
	"log"
	"sort"
	"sotsuron/internal/utils"
	"sync"
	"time"
)

type Species struct {
	individuals          []*Individual
	targetNumIndividuals int
}

func NewSpecies(config AdvancedConfig, numIndividuals, inputWidth, inputHeight, numClasses int, grayscale bool) *Species {
	species := &Species{targetNumIndividuals: numIndividuals}
	species.individuals = make([]*Individual, numIndividuals)
	var wg sync.WaitGroup
	for i := 0; i < numIndividuals; i++ {
		i := i
		wg.Add(1)
		go func() {
			species.individuals[i] = NewIndividual(config, inputWidth, inputHeight, numClasses, grayscale)
			wg.Done()
		}()
	}
	wg.Wait()
	return species
}

func (species *Species) Evolve(
	ctx context.Context, advCfg AdvancedConfig, numGenerations int,
	xTrain, yTrain, xTest, yTest tensor.Tensor,
	progressChan chan Progress, allChartChan chan AllChartData, bestChartChan chan float32) {
	var err error
	var mu sync.Mutex
	progress := Progress{}

	//toyTestImages := make([]tensor.Tensor, 10)
	//toyTestImages[0], err = datasets.LoadImage("/home/m8u/Downloads/datasets/cifar10_10k/airplane/1000.png", species.individuals[0].isGrayscale)
	//toyTestImages[1], err = datasets.LoadImage("/home/m8u/Downloads/datasets/cifar10_10k/automobile/1000.png", species.individuals[0].isGrayscale)
	//toyTestImages[2], err = datasets.LoadImage("/home/m8u/Downloads/datasets/cifar10_10k/bird/1000.png", species.individuals[0].isGrayscale)
	//toyTestImages[3], err = datasets.LoadImage("/home/m8u/Downloads/datasets/cifar10_10k/cat/1000.png", species.individuals[0].isGrayscale)
	//toyTestImages[4], err = datasets.LoadImage("/home/m8u/Downloads/datasets/cifar10_10k/deer/1000.png", species.individuals[0].isGrayscale)
	//toyTestImages[5], err = datasets.LoadImage("/home/m8u/Downloads/datasets/cifar10_10k/dog/1000.png", species.individuals[0].isGrayscale)
	//toyTestImages[6], err = datasets.LoadImage("/home/m8u/Downloads/datasets/cifar10_10k/frog/1000.png", species.individuals[0].isGrayscale)
	//toyTestImages[7], err = datasets.LoadImage("/home/m8u/Downloads/datasets/cifar10_10k/horse/1000.png", species.individuals[0].isGrayscale)
	//toyTestImages[8], err = datasets.LoadImage("/home/m8u/Downloads/datasets/cifar10_10k/ship/1000.png", species.individuals[0].isGrayscale)
	//toyTestImages[9], err = datasets.LoadImage("/home/m8u/Downloads/datasets/cifar10_10k/truck/1000.png", species.individuals[0].isGrayscale)
	//toyTestClasses := []string{"airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"}
	//unfamiliar, err := datasets.LoadDataset("/home/m8u/Downloads/datasets/mnist_png", species.individuals[0].isGrayscale)
	//unfamiliarX, unfamiliarY, _, _, err := unfamiliar.SplitTrainTest(0.99)
	utils.MaybeCrash(err)

	start := time.Now()
	for i := 0; i < numGenerations; i++ {
		fmt.Printf("===================================== Generation %d =====================================\n", i)
		// calculate fitness for each individual
		var wg sync.WaitGroup
		for _, individual := range species.individuals {
			//fmt.Printf("Calculating fitness for %v\n", individual.name)
			// print individual's structure
			//for _, layer := range individual.Chain.Layers {
			//	fmt.Printf("%v\n", layer)
			//}
			if individual.trained {
				fmt.Printf("Skipping %v\n", individual.name)
				progress.Individual++
				continue
			}
			individual := individual
			wg.Add(1)
			go func() {
				fitness, err := individual.CalculateFitnessBatch(ctx, allChartChan, advCfg, xTrain, yTrain, xTest, yTest)
				if err != nil {
					//fmt.Println("WARNING:", err.Error())
					fmt.Println(individual.name, "has died")
				} else {
					individual.trained = true
					individual.fitness = fitness
				}
				mu.Lock()
				progress.Individual++
				select {
				case progressChan <- progress:
				default:
				}
				mu.Unlock()
				wg.Done()
			}()
		}
		wg.Wait()

		select {
		case <-ctx.Done():
			fmt.Println("ABORTING")
			return
		default:
		}

		// eliminate individuals that have failed to train
		for i := 0; i < len(species.individuals); i++ {
			if !species.individuals[i].trained {
				species.individuals = append(species.individuals[:i], species.individuals[i+1:]...)
				i--
			}
		}
		// if there are no at least 2 individuals, then crash
		if len(species.individuals) < 2 {
			log.Fatalln("There are no at least 2 individuals left")
		}
		// find 2 best individuals
		sort.Slice(species.individuals, func(i, j int) bool {
			return species.individuals[i].fitness > species.individuals[j].fitness
		})
		parent1, parent2 := species.individuals[0], species.individuals[1]
		bestChartChan <- parent1.fitness
		//=======================================================================================
		fmt.Println("_______")
		fmt.Printf("Best fitness: %v (%v)\n", parent1.fitness, parent1.name)
		// print layers of the best individual
		for _, layer := range parent1.Chain.Layers {
			fmt.Printf("%+v\n", layer)
		}
		fmt.Println("_______")
		//for i, img := range toyTestImages {
		//	pred, err := parent1.Predict(img)
		//	utils.MaybeCrash(err)
		//	maxIndex := vecf32.Argmax(pred.Data().([]float32))
		//	fmt.Printf("Predicted: %s, Actual: %s", toyTestClasses[maxIndex], toyTestClasses[i])
		//	if maxIndex == i {
		//		fmt.Println(" +")
		//	} else {
		//		fmt.Println()
		//	}
		//}
		//unfamiliarAccuracy, _, err := parent1.evaluateBatch(ctx, unfamiliarX, unfamiliarY, advCfg.BatchSize)
		//fmt.Printf(" AYO  AYO  AYO  AYO  AYO  AYO  AYO  AYO  AYO  AYO  AYO Unfamiliar accuracy: %v\n", unfamiliarAccuracy)
		if err != nil {
			fmt.Println("WARNING:", err.Error())
		}
		fmt.Println("********")
		//=======================================================================================

		if i == numGenerations-1 {
			progress.Generation = -1
			progressChan <- progress
			return
		}
		// crossover
		child1, child2, err1, err2 := parent1.Crossover(advCfg, parent2)
		if err1 != nil && err2 != nil {
			// if crossover fails, just copy the best 2 individuals
			child1, err = parent1.Mutate(advCfg, advCfg.MutationChance)
			utils.MaybeCrash(err)
			child2, err = parent2.Mutate(advCfg, advCfg.MutationChance)
			utils.MaybeCrash(err)
		}
		// create a new generation
		var newGeneration []*Individual
		//if parent1.lives > 0 {
		newGeneration = append(newGeneration, parent1)
		//parent1.lives--
		//}
		newGeneration = append(newGeneration, child1, child2, NewIndividual(advCfg, child1.inputRes.Width, child1.inputRes.Height, child1.numClasses, child1.isGrayscale))
		// mutate N times to fill the rest of new generation
		mutationChance := 1 - (parent1.fitness+parent2.fitness)/2
		fmt.Printf(">>>>>> Mutation chance: %v\n", mutationChance)
		var mutated *Individual
		for i := 0; len(newGeneration) < species.targetNumIndividuals && i < species.targetNumIndividuals*3; i++ {
			if i%2 == 0 {
				mutated, err = child1.Mutate(advCfg, mutationChance)
			} else {
				mutated, err = child2.Mutate(advCfg, mutationChance)
			}
			if err == nil {
				newGeneration = append(newGeneration, mutated)
			}
		}
		if len(newGeneration) < len(species.individuals) {
			log.Fatalln("Not enough individuals for a new generation")
		}
		species.individuals = newGeneration
		progress.Generation++
		progress.ETASeconds = time.Since(start).Seconds() / float64(i+1) * float64(numGenerations-i-1)
		select {
		case progressChan <- progress:
		default:
		}
	}
	return
}

// Best returns the best individual
// it is assumed that the individuals are sorted by fitness after Evolve() is called
func (species *Species) Best() *Individual {
	return species.individuals[0]
}
