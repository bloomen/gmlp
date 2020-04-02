#pragma once

#include <iostream>
#include <memory>

#include "loss.h"
#include "Neuron.h"

namespace gmlp
{

enum TargetType : std::uint8_t
{
    Classification,
    Regression,
};

class Network
{
public:
    Network(const TargetType target_type,
            const std::vector<std::size_t>& layers,
            init::RandomEngine& random_engine)
        : target_type_{target_type}
    {
        assert(layers.size() > 0);
        layers_.resize(layers.size());
        std::size_t current_layer = 0;

        if (layers.size() > 1)
        {
            // input layer
            assert(layers[current_layer] > 0);
            for (std::size_t i = 0; i < layers[current_layer]; ++i)
            {
                layers_[current_layer].transfer = std::make_unique<transfer::Sigmoid>();
                add_neuron(layers_[current_layer], random_engine, layers[current_layer]);
            }
            ++current_layer;

            // hidden layers
            for (; current_layer + 1 < layers.size(); ++current_layer)
            {
                assert(layers[current_layer] > 0);
                for (std::size_t i = 0; i < layers[current_layer]; ++i)
                {
                    layers_[current_layer].transfer = std::make_unique<transfer::Sigmoid>();
                    add_neuron(layers_[current_layer], random_engine, layers[current_layer - 1]);
                }
            }
        }

        // output layer
        assert(current_layer + 1 == layers.size());
        assert(layers[current_layer] > 0);
        for (std::size_t i = 0; i < layers[current_layer]; ++i)
        {
            switch (target_type)
            {
                case TargetType::Classification:
                {
                    layers_[current_layer].transfer = std::make_unique<transfer::Sigmoid>();
                    break;
                }
                case TargetType::Regression:
                {
                    layers_[current_layer].transfer = std::make_unique<transfer::Linear>();
                    break;
                }
            }
            add_neuron(layers_[current_layer], random_engine, layers[current_layer - 1]);
        }

        // loss
        assert(layers.back() > 0);
        switch (target_type)
        {
            case TargetType::Classification:
            {
                if (layers.back() > 1)
                {
                    loss_ = std::make_unique<loss::CE>();
                }
                else
                {
                    loss_ = std::make_unique<loss::SE>();
                }
                break;
            }
            case TargetType::Regression:
            {
                loss_ = std::make_unique<loss::SE>();
                break;
            }
        }
    }

    void print() const
    {
        std::cout << "Network arch (" << layers_.size() << "): ";
        for (const Layer& layer : layers_)
        {
            std::cout << layer.neurons.size();
            if (&layer != &layers_.back())
            {
                std::cout << " -> ";
            }
        }
        std::cout << std::endl;
    }

    TargetType get_target_type() const
    {
        return target_type_;
    }

    std::vector<std::size_t> get_layers() const
    {
        std::vector<std::size_t> layers;
        for (const Layer& layer : layers_)
        {
            layers.push_back(layer.neurons.size());
        }
        return layers;
    }

    const std::vector<float>& get_weights() const
    {
        return weights_;
    }

    std::vector<float>& get_weights()
    {
        return weights_;
    }

    void set_weights(std::vector<float> weights)
    {
        assert(weights_.size() == weights.size());
        weights_ = std::move(weights);
    }

    void save(std::ostream& os)
    {
        write(os, target_type_);
        write(os, layers_.size());
        for (const Layer& layer : layers_)
        {
            write(os, layer.neurons.size());
        }
        write(os, weights_.size());
        for (const auto& value : weights_)
        {
            write(os, value);
        }
    }

    static Network load(std::istream& is)
    {
        TargetType target_type;
        read(is, target_type);
        std::size_t layer_count;
        read(is, layer_count);
        std::vector<std::size_t> layers(layer_count);
        for (auto& value : layers)
        {
            read(is, value);
        }
        std::size_t weight_count;
        read(is, weight_count);
        std::vector<float> weights(weight_count);
        for (auto& value : weights)
        {
            read(is, value);
        }
        init::DefaultRandomEngine random_engine{1};
        Network net{static_cast<TargetType>(target_type), layers, random_engine};
        net.set_weights(weights);
        return net;
    }

    Network clone() const
    {
        init::DefaultRandomEngine random_engine{1};
        Network cloned{get_target_type(), get_layers(), random_engine};
        cloned.set_weights(get_weights());
        return cloned;
    }

    float train(const std::vector<std::vector<float>>& X,
                const std::vector<std::vector<float>>& y,
                const float learning_rate)
    {
        assert(X.size() == y.size());
        float loss = 0;
        for (std::size_t i = 0; i < X.size(); ++i)
        {
            const auto output = forward(X[i]);
            std::vector<float> deltas;
            loss += loss_multi_output(deltas, y[i], output);
            backward(std::move(deltas));
            update(learning_rate);
        }
        loss_->transform_error(loss);
        return loss;
    }

    std::vector<float> predict(const std::vector<float>& input) const
    {
        std::vector<float> output = input;
        for (const Layer& layer : layers_)
        {
            std::vector<float> new_input;
            for (const Neuron& neuron : layer.neurons)
            {
                new_input.push_back(neuron.predict(weights_, output, *layer.transfer));
            }
            output = std::move(new_input);
        }
        loss_->transform_output(output.data(), output.size());
        return output;
    }

private:

    struct Layer
    {
        std::unique_ptr<transfer::Transfer> transfer;
        std::vector<Neuron> neurons;
    };

    template<typename T>
    static void write(std::ostream& os, const T& value)
    {
        os.write(reinterpret_cast<const char*>(&value), sizeof(T));
    }

    template<typename T>
    static void read(std::istream& is, T& value)
    {
        is.read(reinterpret_cast<char*>(&value), sizeof(T));
    }

    void add_neuron(Layer& layer,
                    init::RandomEngine& random_engine,
                    const std::size_t n_inputs)
    {
        layer.neurons.emplace_back(weights_.size());
        const auto weight_count = n_inputs + 1; // weights + bias
        weights_.resize(weights_.size() + weight_count);
        init::xavier(random_engine, weights_.data() + weights_.size() - weight_count, weight_count);
    }

    std::vector<float> forward(const std::vector<float>& input)
    {
        std::vector<float> output = input;
        for (Layer& layer : layers_)
        {
            std::vector<float> new_input;
            for (Neuron& neuron : layer.neurons)
            {
                new_input.push_back(neuron.forward(weights_, output, *layer.transfer));
            }
            output = std::move(new_input);
        }
        return output;
    }

    void backward(std::vector<float> deltas)
    {
        for (std::size_t i = layers_.size(); i--;)
        {
            Layer& layer = layers_[i];
            if (i + 1 < layers_.size())
            {
                deltas.clear();
                for (std::size_t j = 0; j < layer.neurons.size(); ++j)
                {
                    float delta = 0;
                    for (Neuron& neuron : layers_[i + 1].neurons)
                    {
                        delta += neuron.get_delta(weights_, j);
                    }
                    deltas.push_back(delta);
                }
            }
            for (std::size_t j = 0; j < layer.neurons.size(); ++j)
            {
                Neuron& neuron = layer.neurons[j];
                neuron.backward(deltas[j], *layer.transfer);
            }
        }
    }

    void update(const float learning_rate)
    {
        for (Layer& layer : layers_)
        {
            for (Neuron& neuron : layer.neurons)
            {
                neuron.update(weights_, learning_rate);
            }
        }
    }

    float loss_multi_output(std::vector<float>& deltas,
                            const std::vector<float>& truth,
                            const std::vector<float>& pred)
    {
        assert(pred.size() == truth.size());
        auto transformed_pred = pred;
        loss_->transform_output(transformed_pred.data(), transformed_pred.size());
        float loss = 0;
        for (std::size_t i = 0; i < truth.size(); ++i)
        {
            loss += loss_->call(truth[i], transformed_pred[i]);
            deltas.push_back(loss_->call_deriv(truth[i], pred[i]));
        }
        return loss;
    }

    TargetType target_type_;
    std::unique_ptr<loss::Loss> loss_;
    std::vector<Layer> layers_;
    std::vector<float> weights_;
};

}
