#pragma once

#include <vector>

#include "init.h"
#include "Network.h"
#include "utils.h"

namespace gmlp
{

inline void crossover(std::vector<float>& w1,
                      std::vector<float>& w2,
                      const float ratio,
                      init::RandomEngine& random_engine)
{
    assert(w1.size() == w2.size());
    assert(ratio > 0.0f);
    assert(ratio < 1.0f);
    std::uniform_real_distribution<float> uniform;
    for (std::size_t i = 0; i < w1.size(); ++i)
    {
        if (uniform(random_engine) < ratio)
        {
            std::swap(w1[i], w2[i]);
        }
    }
}

inline void mutate(std::vector<float>& w,
                   const float ratio,
                   const float sigma,
                   init::RandomEngine& random_engine)
{
    assert(ratio > 0.0f);
    assert(ratio < 1.0f);
    std::uniform_real_distribution<float> uniform;
    std::normal_distribution<float> normal(0.0f, sigma);
    for (std::size_t i = 0; i < w.size(); ++i)
    {
        if (uniform(random_engine) < ratio)
        {
            w[i] += w[i] * normal(random_engine);
        }
    }
}

struct Model
{
    float loss;
    Network net;
};

inline std::vector<Model> make_population(const std::size_t population_size,
                                          const TargetType target_type,
                                          const std::vector<std::size_t>& layers,
                                          init::RandomEngine& random_engine)
{
    std::vector<Model> population;
    for (std::size_t p = 0; p < population_size; ++p)
    {
        gmlp::Network net{target_type, layers, random_engine};
        population.push_back({-1.0f, std::move(net)});
    }
    return population;
}

inline void select_fittest(std::vector<Model>& population,
                           const std::size_t n_fittest,
                           const std::vector<std::vector<float>>& X,
                           const std::vector<std::vector<float>>& y)
{
    for (auto& model : population)
    {
        std::vector<std::vector<float>> pred;
        for (const auto& row : X)
        {
            pred.emplace_back(model.net.predict(row));
        }
        model.loss = gmlp::mae(y, pred);
    }
    std::sort(population.begin(), population.end(), [](const auto& x, const auto& y)
    {
        return x.loss < y.loss;
    });
    population.erase(population.begin() + n_fittest, population.end());
}

inline void reproduce(std::vector<Model>& population,
                      const float crossover_ratio,
                      const float mutate_ratio,
                      const float mutate_sigma,
                      init::RandomEngine& random_engine)
{
    auto size = population.size();
    if (size % 2 != 0)
    {
        size -= 1;
    }
    for (std::size_t i = 0; i < size; ++i)
    {
        auto child1 = population[i].net.clone();
        auto child2 = population[++i].net.clone();
        crossover(child1.get_weights(),
                  child2.get_weights(),
                  crossover_ratio,
                  random_engine);
        mutate(child1.get_weights(),
               mutate_ratio,
               mutate_sigma,
               random_engine);
        mutate(child2.get_weights(),
               mutate_ratio,
               mutate_sigma,
               random_engine);
        population.push_back({-1.0f, std::move(child1)});
        population.push_back({-1.0f, std::move(child2)});
    }
}

inline std::vector<Model> ga_optimize(const std::size_t n_generations,
                                      const std::size_t population_size,
                                      const float crossover_ratio,
                                      const float mutate_ratio,
                                      const float mutate_sigma,
                                      const TargetType target_type,
                                      const std::vector<size_t>& layers,
                                      const std::vector<std::vector<float>>& X,
                                      const std::vector<std::vector<float>>& y,
                                      init::RandomEngine& random_engine)
{
    const auto n_fittest = population_size / 2;
    auto population = gmlp::make_population(n_fittest, target_type, layers, random_engine);
    for (std::size_t g = 0; g < n_generations; ++g)
    {
        gmlp::reproduce(population, crossover_ratio, mutate_ratio, mutate_sigma, random_engine);
        std::cout << "generation: " << g << std::endl;
        std::cout << "population size: " << population.size() << std::endl;
        gmlp::select_fittest(population, n_fittest, X, y);
        std::cout << "lowest loss: " << population.front().loss << std::endl;
    }
    return population;
}

}
