#pragma once

#include <vector>

#include "init.h"

namespace gmlp
{

struct Split
{
    std::vector<std::vector<float>> X_train;
    std::vector<std::vector<float>> X_test;
    std::vector<std::vector<float>> y_train;
    std::vector<std::vector<float>> y_test;
};

inline Split split_train_test(const std::vector<std::vector<float>>& X,
                              const std::vector<std::vector<float>>& y,
                              const float test_ratio,
                              init::RandomEngine& random_engine)
{
    std::uniform_real_distribution<float> uniform;
    Split split;
    for (std::size_t i = 0; i < X.size(); ++i)
    {
        if (uniform(random_engine) < test_ratio)
        {
            split.X_test.push_back(X[i]);
            split.y_test.push_back(y[i]);
        }
        else
        {
            split.X_train.push_back(X[i]);
            split.y_train.push_back(y[i]);
        }
    }
    return split;
}

inline float mae(const std::vector<std::vector<float>>& truth,
                 const std::vector<std::vector<float>>& pred)
{
    assert(truth.size() == pred.size());
    float sum = 0.0f;
    for (std::size_t i = 0; i < truth.size(); ++i)
    {
        float sub = 0.0f;
        assert(truth[i].size() == pred[i].size());
        for (std::size_t j = 0; j < truth[i].size(); ++j)
        {
            sub += std::abs(truth[i][j] - pred[i][j]);
        }
        sub /= static_cast<float>(truth[i].size());
        sum += sub;
    }
    return sum / static_cast<float>(truth.size());
}

inline float mse(const std::vector<std::vector<float>>& truth,
                 const std::vector<std::vector<float>>& pred)
{
    assert(truth.size() == pred.size());
    float sum = 0.0f;
    for (std::size_t i = 0; i < truth.size(); ++i)
    {
        float sub = 0.0f;
        assert(truth[i].size() == pred[i].size());
        for (std::size_t j = 0; j < truth[i].size(); ++j)
        {
            const float diff = truth[i][j] - pred[i][j];
            sub += diff * diff;
        }
        sub /= static_cast<float>(truth[i].size());
        sum += sub;
    }
    return sum / static_cast<float>(truth.size());
}

}
