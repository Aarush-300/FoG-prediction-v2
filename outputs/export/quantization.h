// TFLite affine quantization: q = round(real / scale) + zero_point.
constexpr float kInputScale = 0.132359624f;
constexpr int kInputZeroPoint = -5;
constexpr float kOutputScale = 0.00390625f;
constexpr int kOutputZeroPoint = -128;
