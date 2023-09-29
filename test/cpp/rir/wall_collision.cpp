#include <gtest/gtest.h>
#include <torchaudio/csrc/rir/wall.h>

using namespace torchaudio::rir;

struct CollisionTestParam {
  // Input
  torch::Tensor origin;
  torch::Tensor direction;
  // Expected
  torch::Tensor hit_point;
  int next_wall_index;
  float hit_distance;
};

CollisionTestParam par(
    torch::ArrayRef<float> origin,
    torch::ArrayRef<float> direction,
    torch::ArrayRef<float> hit_point,
    int next_wall_index,
    float hit_distance) {
  auto dir = torch::tensor(direction);
  return {
      torch::tensor(origin),
      dir / dir.norm(),
      torch::tensor(hit_point),
      next_wall_index,
      hit_distance};
}

//////////////////////////////////////////////////////////////////////////////
// 2D test
//////////////////////////////////////////////////////////////////////////////

class Simple2DRoomCollisionTest
    : public ::testing::TestWithParam<CollisionTestParam> {};

TEST_P(Simple2DRoomCollisionTest, CollisionTest2D) {
  //
  //  ^
  //  |        3
  //  |     ______
  //  |    |      |
  //  |  0 |      | 1
  //  |    |______|
  //  |        2
  // -+---------------->
  //
  auto room = torch::tensor({1, 1});

  auto param = GetParam();
  auto [hit_point, next_wall_index, hit_distance] =
      find_collision_wall<float, 2>(room, param.origin, param.direction);

  EXPECT_EQ(param.next_wall_index, next_wall_index);
  EXPECT_FLOAT_EQ(param.hit_distance, hit_distance);
  EXPECT_TRUE(torch::allclose(
      param.hit_point, hit_point, /*rtol*/ 1e-05, /*atol*/ 1e-07));
}

#define ISQRT2 0.70710678118

INSTANTIATE_TEST_CASE_P(
    Collision2DTests,
    Simple2DRoomCollisionTest,
    ::testing::Values(
        // From 0
        par({0.0, 0.5}, {1.0, 0.0}, {1.0, 0.5}, 1, 1.0),
        par({0.0, 0.5}, {1.0, -1.}, {0.5, 0.0}, 2, ISQRT2),
        par({0.0, 0.5}, {1.0, 1.0}, {0.5, 1.0}, 3, ISQRT2),
        // From 1
        par({1.0, 0.5}, {-1., 0.0}, {0.0, 0.5}, 0, 1.0),
        par({1.0, 0.5}, {-1., -1.}, {0.5, 0.0}, 2, ISQRT2),
        par({1.0, 0.5}, {-1., 1.0}, {0.5, 1.0}, 3, ISQRT2),
        // From 2
        par({0.5, 0.0}, {-1., 1.0}, {0.0, 0.5}, 0, ISQRT2),
        par({0.5, 0.0}, {1.0, 1.0}, {1.0, 0.5}, 1, ISQRT2),
        par({0.5, 0.0}, {0.0, 1.0}, {0.5, 1.0}, 3, 1.0),
        // From 3
        par({0.5, 1.0}, {-1., -1.}, {0.0, 0.5}, 0, ISQRT2),
        par({0.5, 1.0}, {1.0, -1.}, {1.0, 0.5}, 1, ISQRT2),
        par({0.5, 1.0}, {0.0, -1.}, {0.5, 0.0}, 2, 1.0)));

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
      find_collision_wall<float, 3>(room, param.origin, param.direction);

  EXPECT_EQ(param.next_wall_index, next_wall_index);
  EXPECT_FLOAT_EQ(param.hit_distance, hit_distance);
  EXPECT_TRUE(torch::allclose(
      param.hit_point, hit_point, /*rtol*/ 1e-05, /*atol*/ 1e-07));
}

INSTANTIATE_TEST_CASE_P(
    Collision3DTests,
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
