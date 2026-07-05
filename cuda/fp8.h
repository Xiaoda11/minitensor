#ifndef FP8_H
#define FP8_H

#include <cmath>
#include <cstdint>
#include <cstdio>

inline uint8_t float_to_e4m3(float val, float scale) {
    if (scale == 0.0f) scale = 1.0f;

    float v = val / scale;
    uint8_t sign = std::signbit(v) ? 1 : 0;
    float av = std::fabs(v);

    if (av == 0.0f) return sign << 7;
    if (!std::isfinite(av) || av >= 448.0f)
        return (sign << 7) | (15 << 3) | 6;

    int exponent = (int)std::floor(std::log2(av));
    int exp_bits = exponent + 7;
    if (exp_bits <= 0) return sign << 7;
    if (exp_bits >= 15) return (sign << 7) | (15 << 3) | 6;

    float base = std::ldexp(1.0f, exponent);
    float normalized = av / base;
    int mantissa = (int)std::floor((normalized - 1.0f) * 8.0f + 0.5f);

    if (mantissa == 8) {
        mantissa = 0;
        exp_bits++;
        if (exp_bits >= 15)
            return (sign << 7) | (15 << 3) | 6;
    }

    return (sign << 7) | ((uint8_t)exp_bits << 3) | (uint8_t)mantissa;
}

inline float e4m3_to_float(uint8_t bits, float scale) {
    uint8_t sign = bits >> 7;
    uint8_t exp_bits = (bits >> 3) & 0x0f;
    uint8_t mantissa = bits & 0x07;

    if (exp_bits == 0) return 0.0f;

    float value = std::ldexp(1.0f + (float)mantissa / 8.0f, (int)exp_bits - 7);
    if (sign) value = -value;
    return value * scale;
}

inline float compute_fp8_scale(const float *data, int n) {
    float max_abs = 0.0f;
    for (int i = 0; i < n; ++i) {
        float av = std::fabs(data[i]);
        if (av > max_abs) max_abs = av;
    }
    if (max_abs == 0.0f) return 1.0f;
    return max_abs / 448.0f;
}

#ifdef FP8_UNIT_TEST
int main() {
    float values[] = {
        0.0f, 1.0f, -1.0f, 2.0f, 0.5f,
        448.0f, -448.0f, 0.001f, 1000.0f
    };
    float scale = 1.0f;

    for (int i = 0; i < (int)(sizeof(values) / sizeof(values[0])); ++i) {
        uint8_t encoded = float_to_e4m3(values[i], scale);
        float decoded = e4m3_to_float(encoded, scale);
        float error = std::fabs(values[i] - decoded);
        std::printf("input=%f encoded=0x%02x decoded=%f abs_error=%f\n",
                    values[i], encoded, decoded, error);
    }

    float data[] = {0.0f, -2.0f, 4.0f, 8.0f};
    std::printf("scale=%f\n", compute_fp8_scale(data, 4));
    return 0;
}
#endif

#endif
