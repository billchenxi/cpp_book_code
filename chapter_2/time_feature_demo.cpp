#include <cmath>
#include <iostream>

struct TimeFeatures {
    int hour;
    int dayOfWeek;
    int month;
    double hourSin;
    double hourCos;
};

TimeFeatures extractTimeFeatures(int hour, int dayOfWeek, int month) {
    constexpr double kPi = 3.14159265358979323846;

    TimeFeatures f{};
    f.hour = hour;
    f.dayOfWeek = dayOfWeek;
    f.month = month;
    f.hourSin = std::sin(2.0 * kPi * hour / 24.0);
    f.hourCos = std::cos(2.0 * kPi * hour / 24.0);
    return f;
}

int main() {
    const auto f = extractTimeFeatures(/*hour=*/15, /*dayOfWeek=*/1, /*month=*/3);

    std::cout << "hour=" << f.hour
              << " dayOfWeek=" << f.dayOfWeek
              << " month=" << f.month
              << " hourSin=" << f.hourSin
              << " hourCos=" << f.hourCos
              << '\n';
    return 0;
}

/*
Build from the repo root:
g++ -std=c++17 -O2 chapter_2/time_feature_demo.cpp -o chapter_2/time_feature_demo

Run:
./chapter_2/time_feature_demo
*/
