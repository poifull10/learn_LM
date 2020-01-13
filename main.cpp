#include <opencv2/opencv.hpp>

#include <cmath>
#include <iostream>
#include <random>
#include <utility>
#include <vector>

cv::Point2f noise(float stdv = 1.0F) {
  std::random_device seed_gen;
  std::default_random_engine engine(seed_gen());
  std::normal_distribution dist(0.0F, stdv);
  return cv::Point2f(dist(engine), dist(engine));
}

cv::Point2f model(float R, float u, float v, float theta) {
  auto x = R * std::cos(theta) + u;
  auto y = R * std::sin(theta) + v;
  return cv::Point2f(x, y);
}

std::pair<std::vector<cv::Point2f>, std::vector<float>> generateData(size_t N,
                                                                     float R,
                                                                     float u,
                                                                     float v) {
  std::vector<cv::Point2f> ret;
  std::vector<float> thetas;
  std::random_device seed_gen;
  std::default_random_engine engine(seed_gen());
  std::uniform_real_distribution dist(0.0, 3.141592653 * 2);
  for (size_t i = 0; i < N; i++) {
    const auto rndv = dist(engine);
    thetas.push_back(static_cast<float>(rndv));
    ret.push_back(model(R, u, v, rndv));
  }
  return {ret, thetas};
}

class CircleEstimator {
 public:
  CircleEstimator(const std::vector<cv::Point2f>& points)
      : observedPoints(points), N(points.size()) {}

  void setInitialGuess(float r_, float u_, float v_,
                       const std::vector<float>& thetas_) {
    r = r_;
    u = u_;
    v = v_;
    thetas = thetas_;
  }

  float energy() const { return std::pow(cv::norm(residual()), 2); }

  cv::Point2f residual() const {
    cv::Point2f e;
    for (size_t i = 0; i < N; i++) {
      e += observedPoints[i] - model(std::exp(r), u, v, thetas[i]);
    }
    return e;
  }

  float gaussianNewton(size_t iter) {
    for (size_t i = 0; i < iter; i++) {
      const auto J = jac();
      const auto H = J.t() * J;
      const auto res = residual();
      cv::Mat e = cv::Mat::zeros(2, 1, CV_32F);
      e.at<float>(cv::Point(0, 0)) = res.x;
      e.at<float>(cv::Point(0, 1)) = res.y;
      cv::Mat a = J.t() * e;
      cv::Mat dx = H.inv() * a;

      r += dx.at<float>(cv::Point(0, 0));
      u += dx.at<float>(cv::Point(1, 0));
      v += dx.at<float>(cv::Point(2, 0));
      for (size_t j = 0; j < N; j++) {
        thetas[j] += dx.at<float>(cv::Point(3 + j, 0));
      }
      std::cout << energy() << std::endl;
      showParameters();
    }
    return 0;
  }

  void showParameters() {
    std::cout << "==========" << std::endl;
    std::cout << "R = " << std::exp(r) << std::endl;
    std::cout << "u = " << u << std::endl;
    std::cout << "v = " << v << std::endl;
    // std::cout << "thetas = " << std::endl;
    // for (const auto t : thetas) {
    //   std::cout << t << std::endl;
    // }
  }

  cv::Mat jac() {
    cv::Mat J = cv::Mat::zeros(2, N + 3, CV_32F);
    for (size_t i = 0; i < N; i++) {
      J.at<float>(cv::Point(0, 0)) += std::exp(r) * std::cos(thetas[i]);
      J.at<float>(cv::Point(0, 1)) += std::exp(r) * std::sin(thetas[i]);
      J.at<float>(cv::Point(1, 0)) += 1;
      J.at<float>(cv::Point(2, 1)) += 1;
      J.at<float>(cv::Point(3 + i, 0)) = -std::exp(r) * std::sin(thetas[i]);
      J.at<float>(cv::Point(3 + i, 1)) = std::exp(r) * std::cos(thetas[i]);
    }
    return J;
  }

 private:
  std::vector<cv::Point2f> observedPoints;
  std::vector<float> thetas;
  size_t N;
  float r;
  float u;
  float v;
};

int main(int argc, char** argv) {
  float n = 100;
  float r = 300;
  float u = 512;
  float v = 512;

  cv::Mat img = cv::Mat::zeros(cv::Size(u * 2, v * 2), CV_8UC3);

  const auto [gtPoints, thetas] = generateData(100, r, u, v);
  std::vector<cv::Point2f> observedPoints;

  for (const auto& p : gtPoints) {
    const auto noisyP = p + noise(4);
    observedPoints.push_back(noisyP);
    auto& v = img.at<cv::Vec3b>(noisyP);
    v[1] = 255;
  }

  CircleEstimator ce(gtPoints);
  std::cout << std::log(r) << std::endl;
  ce.setInitialGuess(std::log(r) + 0.1, u + 0.2, v - 0.3, thetas);
  std::cout << ce.energy() << std::endl;

  ce.gaussianNewton(10);

  cv::imwrite("observed.png", img);

  return 0;
}
