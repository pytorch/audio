#include <gtest/gtest.h>
#include <libtorchaudio/rir/wall.h>

using namespace torchaudio::rir;

using DTYPE = double;

struct CollisionTestParam {
  // Input
  torch::Tensor origin;
  torch::Tensor direction;
  // Expected
  torch::Tensor hit_point;
  int next_wall_index;
  DTYPE hit_distance;
};

CollisionTestParam par(
    torch::ArrayRef<DTYPE> origin,
    torch::ArrayRef<DTYPE> direction,
    torch::ArrayRef<DTYPE> hit_point,
    int next_wall_index,
    DTYPE hit_distance) {
  auto dir = torch::tensor(direction);
  return {
      torch::tensor(origin),
      dir / dir.norm(),
      torch::tensor(hit_point),
      next_wall_index,
      hit_distance};
}

//////////////////////////////////////////////////////////////////////////////
// 3D test
//////////////////////////////////////////////////////////////////////////////

class Simple3DRoomCollisionTest
    : public ::testing::TestWithParam<CollisionTestParam> {};

TEST_P(Simple3DRoomCollisionTest, CollisionTest3D) {
  //  y                       z
  //  ^                       ^
  //  |        3              |      y
  //  |     ______            |     /
  //  |    |      |           |    /
  //  |  0 |      | 1         |  ______
  //  |    |______|           | /     /  4: floor, 5: ceiling
  //  |        2              |/     /
  // -+----------------> x   -+--------------> x
  //
  auto room = torch::tensor({1, 1, 1});

  auto param = GetParam();
  auto [hit_point, next_wall_index, hit_distance] =
      find_collision_wall<DTYPE>(room, param.origin, param.direction);

  EXPECT_EQ(param.next_wall_index, next_wall_index);
  EXPECT_FLOAT_EQ(param.hit_distance, hit_distance);
  EXPECT_NEAR(
      param.hit_point[0].item<DTYPE>(), hit_point[0].item<DTYPE>(), 1e-5);
  EXPECT_NEAR(
      param.hit_point[1].item<DTYPE>(), hit_point[1].item<DTYPE>(), 1e-5);
  EXPECT_NEAR(
      param.hit_point[2].item<DTYPE>(), hit_point[2].item<DTYPE>(), 1e-5);
}

#define ISQRT2 0.70710678118

INSTANTIATE_TEST_CASE_P(
    BasicCollisionTests,
    Simple3DRoomCollisionTest,
    ::testing::Values(
        // From 0
        par({0, .5, .5}, {1.0, 0.0, 0.0}, {1., .5, .5}, 1, 1.0),
        par({0, .5, .5}, {1.0, -1., 0.0}, {.5, .0, .5}, 2, ISQRT2),
        par({0, .5, .5}, {1.0, 1.0, 0.0}, {.5, 1., .5}, 3, ISQRT2),
        par({0, .5, .5}, {1.0, 0.0, -1.}, {.5, .5, .0}, 4, ISQRT2),
        par({0, .5, .5}, {1.0, 0.0, 1.0}, {.5, .5, 1.}, 5, ISQRT2),
        // From 1
        par({1, .5, .5}, {-1., 0.0, 0.0}, {.0, .5, .5}, 0, 1.0),
        par({1, .5, .5}, {-1., -1., 0.0}, {.5, .0, .5}, 2, ISQRT2),
        par({1, .5, .5}, {-1., 1.0, 0.0}, {.5, 1., .5}, 3, ISQRT2),
        par({1, .5, .5}, {-1., 0.0, -1.}, {.5, .5, .0}, 4, ISQRT2),
        par({1, .5, .5}, {-1., 0.0, 1.0}, {.5, .5, 1.}, 5, ISQRT2),
        // From 2
        par({.5, 0, .5}, {-1., 1.0, 0.0}, {.0, .5, .5}, 0, ISQRT2),
        par({.5, 0, .5}, {1.0, 1.0, 0.0}, {1., .5, .5}, 1, ISQRT2),
        par({.5, 0, .5}, {0.0, 1.0, 0.0}, {.5, 1., .5}, 3, 1.0),
        par({.5, 0, .5}, {0.0, 1.0, -1.}, {.5, .5, .0}, 4, ISQRT2),
        par({.5, 0, .5}, {0.0, 1.0, 1.0}, {.5, .5, 1.}, 5, ISQRT2),
        // From 3
        par({.5, 1, .5}, {-1., -1., 0.0}, {.0, .5, .5}, 0, ISQRT2),
        par({.5, 1, .5}, {1.0, -1., 0.0}, {1., .5, .5}, 1, ISQRT2),
        par({.5, 1, .5}, {0.0, -1., 0.0}, {.5, .0, .5}, 2, 1.0),
        par({.5, 1, .5}, {0.0, -1., -1.}, {.5, .5, .0}, 4, ISQRT2),
        par({.5, 1, .5}, {0.0, -1., 1.0}, {.5, .5, 1.}, 5, ISQRT2),
        // From 4
        par({.5, .5, 0}, {-1., 0.0, 1.0}, {.0, .5, .5}, 0, ISQRT2),
        par({.5, .5, 0}, {1.0, 0.0, 1.0}, {1., .5, .5}, 1, ISQRT2),
        par({.5, .5, 0}, {0.0, -1., 1.0}, {.5, .0, .5}, 2, ISQRT2),
        par({.5, .5, 0}, {0.0, 1.0, 1.0}, {.5, 1., .5}, 3, ISQRT2),
        par({.5, .5, 0}, {0.0, 0.0, 1.0}, {.5, .5, 1.}, 5, 1.0),
        // From 5
        par({.5, .5, 1}, {-1., 0.0, -1.}, {.0, .5, .5}, 0, ISQRT2),
        par({.5, .5, 1}, {1.0, 0.0, -1.}, {1., .5, .5}, 1, ISQRT2),
        par({.5, .5, 1}, {0.0, -1., -1.}, {.5, .0, .5}, 2, ISQRT2),
        par({.5, .5, 1}, {0.0, 1.0, -1.}, {.5, 1., .5}, 3, ISQRT2),
        par({.5, .5, 1}, {0.0, 0.0, -1.}, {.5, .5, .0}, 4, 1.0)));

INSTANTIATE_TEST_CASE_P(
    CornerCollisionTest,
    Simple3DRoomCollisionTest,
    ::testing::Values(
        par({1, 1, 0}, {1., 1., 0.}, {1., 1., 0.}, 1, 0.0),
        par({1, 1, 0}, {-1., 1., 0.}, {1., 1., 0.}, 3, 0.0),
        par({1, 1, 1}, {1., 1., 1.}, {1., 1., 1.}, 1, 0.0),
        par({1, 1, 1}, {-1., 1., 1.}, {1., 1., 1.}, 3, 0.0),
        par({1, 1, 1}, {-1., -1., 1.}, {1., 1., 1.}, 5, 0.0)));
