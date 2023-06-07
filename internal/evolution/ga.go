package evolution

import (
	"context"
	"fmt"
	"github.com/m8u/goro/pkg/v1/layer"
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
	progressChan chan Progress, allChartChan chan AllChartData, bestChartChan chan float32, bestLayersChan chan []layer.Config) {

	var err error
	var mu sync.Mutex
	var wg sync.WaitGroup
	progress := Progress{}

	start := time.Now()
	for i := 0; i < numGenerations; i++ {
		fmt.Printf("===================================== Generation %d =====================================\n", i)
		// calculate fitness for each individual
		for _, individual := range species.individuals {
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
			progress.Generation = -1
			progressChan <- progress
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
		select {
		case bestChartChan <- parent1.fitness:
		default:
		}
		select {
		case bestLayersChan <- parent1.Chain.Layers:
		default:
		}

		if i == numGenerations-1 {
			if progressChan != nil {
				progress.Generation = -1
				progressChan <- progress
			}
			return
		}

		// dispose VMs of obsolete individuals because for an unknown reason GC won't do it
		for _, individual := range species.individuals[1:] {
			individual.DisposeVMs()
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

		newGeneration := make([]*Individual, species.targetNumIndividuals)
		newGeneration[0] = NewIndividual(advCfg, child1.inputRes.Width, child1.inputRes.Height, child1.numClasses, child1.isGrayscale)
		newGeneration[1] = parent1
		newGeneration[2] = child1
		newGeneration[3] = child2
		// mutate N times to fill the rest of new generation
		mutationChance := (1 - (parent1.fitness+parent2.fitness)/2) * 2
		fmt.Printf(">>>>>> Mutation chance: %v\n", mutationChance)
		for i := 4; i < len(newGeneration); i++ {
			wg.Add(1)
			i := i
			go func() {
				if i%2 == 0 {
					newGeneration[i], err = child1.Mutate(advCfg, mutationChance)
					utils.MaybeCrash(err)
				} else {
					newGeneration[i], err = child2.Mutate(advCfg, mutationChance)
					utils.MaybeCrash(err)
				}
				wg.Done()
			}()
		}
		wg.Wait()
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
func (species *Species) Best() *Individual {
	return species.individuals[0]
}

// TODO старые графики в allChart не удаляются (вроде как только если через браузер открывать)
// TODO аперитивка все равно заполняется чето сильно
// TODO есть минимум размера поколения
