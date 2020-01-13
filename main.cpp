#include <opencv2/opencv.hpp>

#include <cmath>
#include <iostream>
#include <random>
#include <tuple>
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
    r = std::log(r_);
    u = u_;
    v = v_;
    thetas = thetas_;
  }

  std::vector<cv::Point2f> estimatedPoints() const {
    std::vector<cv::Point2f> points;
    for (const auto& theta : thetas) {
      points.push_back(model(std::exp(r), u, v, theta));
    }
    return points;
  }

  float energy() const { return std::pow(cv::norm(residual()), 2); }

  cv::Point2f residual() const {
    cv::Point2f e;
    for (size_t i = 0; i < N; i++) {
      const auto sub = observedPoints[i] - model(std::exp(r), u, v, thetas[i]);
      e.x += sub.x * sub.x;
      e.y += sub.y * sub.y;
    }
    return e;
  }

  void gaussianNewton(size_t iter) {
    for (size_t i = 0; i < iter; i++) {
      const auto J = jac();
      const auto H = J.t() * J;
      const auto res = residual();
      cv::Mat e = cv::Mat::zeros(2, 1, CV_32F);
      e.at<float>(cv::Point(0, 0)) = res.x;
      e.at<float>(cv::Point(0, 1)) = res.y;
      cv::Mat a = -J.t() * e;
      cv::Mat dx = H.inv() * a;
      update(dx);
      showParameters();
    }
  }

  void LM(size_t iter, float lambda_ = 1e-10) {
    lambda = lambda_;
    cv::Mat I = cv::Mat::eye(N + 3, N + 3, CV_32F);
    for (size_t i = 0; i < iter; i++) {
      const auto J = jac();
      const auto H = J.t() * J;
      const auto res = residual();
      cv::Mat e = cv::Mat::zeros(2, 1, CV_32F);
      e.at<float>(cv::Point(0, 0)) = res.x;
      e.at<float>(cv::Point(0, 1)) = res.y;
      cv::Mat a = -J.t() * e;
      cv::Mat dx = (H + lambda * I).inv() * a;
      float beforeEnergy = energy();
      update(dx);
      float afterEnergy = energy();
      std::cout << "Lambda = " << lambda << std::endl;
      if (afterEnergy < beforeEnergy) {
        lambda *= 0.1;
        showParameters();
      } else {
        lambda *= 10;
        update(-dx);
        i--;
      }
    }
  }

  void update(const cv::Mat& dx) {
    r += dx.at<float>(cv::Point(0, 0));
    u += dx.at<float>(cv::Point(1, 0));
    v += dx.at<float>(cv::Point(2, 0));
    for (size_t j = 0; j < N; j++) {
      thetas[j] += dx.at<float>(cv::Point(3 + j, 0));
    }
  }

  void showParameters() const {
    std::cout << "==========" << std::endl;
    const auto [R, U, V] = getParams();
    std::cout << "R = " << R << std::endl;
    std::cout << "u = " << U << std::endl;
    std::cout << "v = " << V << std::endl;
    std::cout << "E = " << energy() << std::endl;
  }

  cv::Mat jac() const {
    cv::Mat J = cv::Mat::zeros(2, N + 3, CV_32F);
    for (size_t i = 0; i < N; i++) {
      const auto sub_u =
          observedPoints[i].x - (std::exp(r) * std::cos(thetas[i]) + u);
      const auto sub_v =
          observedPoints[i].y - (std::exp(r) * std::sin(thetas[i]) + v);
      J.at<float>(cv::Point(0, 0)) +=
          -std::exp(r) * std::cos(thetas[i]) * sub_u;
      J.at<float>(cv::Point(0, 1)) +=
          -std::exp(r) * std::sin(thetas[i]) * sub_v;
      J.at<float>(cv::Point(1, 0)) += -sub_u;
      J.at<float>(cv::Point(2, 1)) += -sub_v;
      J.at<float>(cv::Point(3 + i, 0)) =
          std::exp(r) * std::sin(thetas[i]) * sub_u;
      J.at<float>(cv::Point(3 + i, 1)) =
          -std::exp(r) * std::cos(thetas[i]) * sub_v;
    }
    return J;
  }

  std::tuple<float, float, float> getParams() const {
    return {std::exp(r), u, v};
  }

 private:
  std::vector<cv::Point2f> observedPoints;
  float lambda;
  std::vector<float> thetas;
  size_t N;
  float r;
  float u;
  float v;
};

int main(int argc, char** argv) {
  float n = 1000;
  float r = 300;
  float u = 512;
  float v = 512;

  cv::Mat img = cv::Mat::zeros(cv::Size(u * 2, v * 2), CV_8UC3);
  cv::circle(img, cv::Point(u, v), r, cv::Scalar(255, 255, 255));

  const auto [gtPoints, gtThetas] = generateData(n, r, u, v);

  for (const auto& p : gtPoints) {
    auto& v = img.at<cv::Vec3b>(p);
    v[1] = 255;
  }

  CircleEstimator ce(gtPoints);
  auto initR = r + 30;
  auto initU = u + 10;
  auto initV = v -10;
  ce.setInitialGuess(initR, initU, initV, gtThetas);
  ce.showParameters();
  ce.LM(10);

  const auto [R, U, V] = ce.getParams();

  cv::circle(img, cv::Point(initU, initV), initR, cv::Scalar(128, 128, 128));
  cv::circle(img, cv::Point(U, V), R, cv::Scalar(0, 0, 255));

  // for (const auto& p : ce.estimatedPoints()) {
  //   auto& v = img.at<cv::Vec3b>(p);
  //   v[0] = 0;
  //   v[1] = 255;
  //   v[2] = 255;
  // }

  cv::imwrite("result.png", img);

  return 0;
}
