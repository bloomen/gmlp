#pragma once

#include <cassert>
#include <cstddef>
#include <memory>
#include <random>
#include <vector>

#include "init.h"
#include "transfer.h"

namespace gmlp
{

class Neuron
{
public:

    explicit
    Neuron(const std::size_t weight_offset)
        : weight_offset_{weight_offset}
    {}

    float get_delta(const std::vector<float>& weights,
                    const std::size_t index) const
    {
        return weights[weight_offset_ + index] * data_.back();
    }

    float predict(const std::vector<float>& weights,
                  const std::vector<float>& inputs,
                  const transfer::Transfer& transfer) const
    {
        float output = 0.0f;
        for (std::size_t i = 0; i < inputs.size(); ++i)
        {
            output += weights[weight_offset_ + i] * inputs[i];
        }
        output += weights[weight_offset_ + inputs.size()]; // bias
        output = transfer.call(output);
        return output;
    }

    float forward(const std::vector<float>& weights,
                  const std::vector<float>& inputs,
                  const transfer::Transfer& transfer)
    {
        const auto output = predict(weights, inputs, transfer);
        if (data_.empty())
        {
            data_.resize(inputs.size() + input_offset_);
        }
        std::copy(inputs.begin(), inputs.end(), data_.begin());
        data_[inputs.size()] = output;
        return output;
    }

    void backward(const float delta,
                  const transfer::Transfer& transfer)
    {
        const auto n_inputs = data_.size() - input_offset_;
        data_.back() = transfer.call_deriv(data_[n_inputs]) * delta;
    }

    void update(std::vector<float>& weights,
                const float learning_rate)
    {
        const auto n_inputs = data_.size() - input_offset_;
        const auto update = -learning_rate * data_.back(); // SGD
        for (std::size_t i = 0; i < n_inputs; ++i)
        {
            weights[weight_offset_ + i] += update * data_[i];
        }
        weights[weight_offset_ + n_inputs] += update; // bias
    }

private:
    // data_ contains neuron inputs except for the last input_offset_ which
    // are 'output' and 'delta' (for cache friendlyness)
    static constexpr std::size_t input_offset_ = 2;
    std::size_t weight_offset_;
    std::vector<float> data_;
};

}
