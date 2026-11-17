// Per-channel z-score. Input channel order: acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z, sqrt(ax*ax+ay*ay+az*az), sqrt(gx*gx+gy*gy+gz*gz).
// For 8ch, append raw-unit magnitudes BEFORE normalization; then apply (x-mean)/std.
constexpr int kNumChannels = 8;
constexpr float kMean[8] = {-1.00720751e+00f, 1.31087065e-01f, 1.12557568e-01f, 2.82904124e+00f, 8.48814368e-01f, -6.53396606e-01f, 1.07770348e+00f, 5.38501472e+01f};
constexpr float kStd[8] = {1.98050335e-01f, 3.15653056e-01f, 1.93598330e-01f, 4.76803818e+01f, 2.37632942e+01f, 6.62965775e+01f, 2.38938376e-01f, 6.59333954e+01f};
