#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>

namespace gmlp
{

namespace loss
{

class Loss
{
public:
    virtual ~Loss() = default;
    virtual float call(float truth, float pred) const = 0;
    virtual float call_deriv(float truth, float pred) const = 0;
    virtual void transform_output(float* output, std::size_t n) const = 0;
    virtual void transform_error(float& error) const = 0;
};

class SE : public Loss
{
public:
    float call(const float truth, const float pred) const override
    {
        const auto diff = truth - pred;
        return diff * diff;
    }

    float call_deriv(const float truth, const float pred) const override
    {
        return pred - truth;
    }

    void transform_output(float*, std::size_t) const override
    {
        // nothing to do
    }

    void transform_error(float& error) const override
    {
        error *= 0.5f;
    }
};

namespace detail
{

inline void softmax(float* output, const std::size_t n)
{
    const auto D = *std::max_element(output, output + n);
    float denom = 1e-6f;
    for (std::size_t i = 0; i < n; ++i)
    {
        output[i] -= D; // for numerical stability
        denom += std::exp(output[i]);
    }
    for (std::size_t i = 0; i < n; ++i)
    {
        output[i] = std::exp(output[i]) / denom;
    }
}

}

class CE : public Loss
{
public:
    float call(const float truth, const float pred) const override
    {
        return -truth * std::log(pred + 1e-6f);
    }

    float call_deriv(const float truth, const float pred) const override
    {
        return pred - truth;
    }

    void transform_output(float* output, const std::size_t n) const
    {
        detail::softmax(output, n);
    }

    void transform_error(float&) const override
    {
        // nothing to do
    }
};

}

}
