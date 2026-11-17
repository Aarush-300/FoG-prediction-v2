// Tune on validation only; apply to INT8 model output probability.
constexpr float kFogThreshold = 0.36328125f;
constexpr int kRequireConsecutive = 2;
constexpr float kEmaAlpha = 0.7f;
constexpr float kThresholdOff = 0.36328125f;
constexpr int kOffConsecutive = 2;
