#include <fstream>
#include <sstream>

#include "Network.h"
#include "genetic.h"
#include "utils.h"

int main()
{
    std::ifstream f{"/home/cblume/workspace/cbnn/data/boston.csv"};
    std::vector<std::vector<float>> X;
    std::vector<std::vector<float>> y;
    while (!f.eof())
    {
        X.emplace_back(13);
        y.emplace_back(1);
        for (std::size_t i = 0; i < 13; ++i)
        {
            f >> X.back()[i];
        }
        f >> y.back()[0];
    }

    gmlp::init::DefaultRandomEngine engine{42};

    const auto split = gmlp::split_train_test(X, y, 0.3f, engine);
    std::cout << "training with " << split.X_train.size() << " samples" << std::endl;

    const auto target_type = gmlp::Regression;
    const std::vector<std::size_t> layers = {13, 13, 1};
    const std::size_t n_gens = 300;
    const std::size_t population_size = 100;
    const float crossover_ratio = 0.5f;
    const float mutate_ratio = 0.05f;
    const float mutate_sigma = 2.0f;

    if (target_type == gmlp::Classification)
    {
        for (auto& value : y)
        {
            if (value[0] > 22)
            {
                value[0] = 1;
            }
            else
            {
                value[0] = 0;
            }
        }
    }

    const auto population = gmlp::ga_optimize(n_gens, population_size, crossover_ratio,
                                              mutate_ratio, mutate_sigma, target_type,
                                              layers, split.X_train, split.y_train, engine);

    std::cout << "prediction" << std::endl;
    std::vector<std::vector<float>> pred;
    for (const auto& row : split.X_test)
    {
        pred.push_back(population.front().net.predict(row));
    }

    if (target_type == gmlp::Classification)
    {
        for (auto& value : pred)
        {
            value[0] = value[0] > 0.5f ? 1.0f : 0.0f;
        }
    }

    std::cout << "MAE=" << gmlp::mae(split.y_test, pred) << std::endl;
}
