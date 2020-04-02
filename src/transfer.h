#pragma once

namespace gmlp
{

namespace transfer
{

class Transfer
{
public:
    virtual ~Transfer() = default;
    virtual float call(float x) const = 0;
    virtual float call_deriv(float x) const = 0;
};

class Linear : public Transfer
{
public:
    float call(float x) const override
    {
        return x;
    }

    float call_deriv(float) const override
    {
        return 1.0f;
    }
};

class Sigmoid : public Transfer
{
public:
    float call(float x) const override
    {
        return 1.0f / (1.0f + std::exp(-x));
    }

    float call_deriv(float x_sigmoid) const override
    {
        return x_sigmoid * (1.0f - x_sigmoid);
    }
};

class Tanh : public Transfer
{
public:
    float call(float x) const override
    {
        return std::tanh(x);
    }

    float call_deriv(float x_tanh) const override
    {
        return 1 - x_tanh * x_tanh;
    }
};

class Relu : public Transfer
{
public:
    float call(float x) const override
    {
        return x > 0.0f ? x : 0.0f;
    }

    float call_deriv(float x) const override
    {
        return x > 0.0f ? 1.0f : 0.0f;
    }
};

}

}
