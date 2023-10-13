#pragma once
#include <torch/types.h>

#define EPS ((scalar_t)(1e-5))
#define SCALAR(x) ((x).template item<scalar_t>())

namespace torchaudio {
namespace rir {

////////////////////////////////////////////////////////////////////////////////
// Basic Wall implementation
////////////////////////////////////////////////////////////////////////////////

/// Wall helper class. A wall records its own absorption, reflection and
/// scattering coefficient, and exposes a few methods for geometrical operations
/// (e.g. reflection of a ray)
template <typename scalar_t>
struct Wall {
  const torch::Tensor origin;
  const torch::Tensor normal;
  const torch::Tensor scattering;
  const torch::Tensor reflection;

  Wall(
      const torch::ArrayRef<scalar_t>& origin,
      const torch::ArrayRef<scalar_t>& normal,
      const torch::Tensor& absorption,
      const torch::Tensor& scattering)
      : origin(torch::tensor(origin).to(scattering.dtype())),
        normal(torch::tensor(normal).to(scattering.dtype())),
        scattering(scattering),
        reflection(1. - absorption) {}
};

/// Returns the side (-1, 1 or 0) on which a point lies w.r.t. the wall.
template <typename scalar_t>
int side(const Wall<scalar_t>& wall, const torch::Tensor& pos) {
  auto dot = SCALAR((pos - wall.origin).dot(wall.normal));

  if (dot > EPS) {
    return 1;
  } else if (dot < -EPS) {
    return -1;
  } else {
    return 0;
  }
}

/// Reflects a ray (dir) on the wall. Preserves norm of vector.
template <typename scalar_t>
torch::Tensor reflect(const Wall<scalar_t>& wall, const torch::Tensor& dir) {
  return dir - wall.normal * 2 * dir.dot(wall.normal);
}

/// Returns the cosine angle of a ray (dir) with the normal of the wall
template <typename scalar_t>
scalar_t cosine(const Wall<scalar_t>& wall, const torch::Tensor& dir) {
  return SCALAR(dir.dot(wall.normal) / dir.norm());
}

////////////////////////////////////////////////////////////////////////////////
// Room (multiple walls) and interactions
////////////////////////////////////////////////////////////////////////////////

/// Creates a shoebox room consists of multiple walls.
/// Normals are vectors facing *outwards* the room, and origins are arbitrary
/// corners of each wall.
///
/// Note:
/// The wall has to be ordered in the following way:
/// - parallel walls are next (W/E, S/N, and F/C)
/// - The one closer to the origin must come first. (W -> E, S -> N, F -> C)
/// - The order of wall pair must be W/E, S/N, then F/C because
///   `find_collision_wall` will search in the order x, y, z and
///   wall pairs must be distibguishable on these axis.

/// 3D room
template <typename T>
const std::array<Wall<T>, 6> make_room(
    const T& w,
    const T& l,
    const T& h,
    const torch::Tensor& abs,
    const torch::Tensor& scat) {
  using namespace torch::indexing;
#define SLICE(x, i) x.index({Slice(), i})
  return {
      Wall<T>({0, l, 0}, {-1, 0, 0}, SLICE(abs, 0), SLICE(scat, 0)), // West
      Wall<T>({w, 0, 0}, {1, 0, 0}, SLICE(abs, 1), SLICE(scat, 1)), // East
      Wall<T>({0, 0, 0}, {0, -1, 0}, SLICE(abs, 2), SLICE(scat, 2)), // South
      Wall<T>({w, l, 0}, {0, 1, 0}, SLICE(abs, 3), SLICE(scat, 3)), // North
      Wall<T>({w, 0, 0}, {0, 0, -1}, SLICE(abs, 4), SLICE(scat, 4)), // Floor
      Wall<T>({w, 0, h}, {0, 0, 1}, SLICE(abs, 5), SLICE(scat, 5)) // Ceiling
  };
#undef SLICE
}

/// Find a wall that the given ray hits.
/// The room is assumed to be shoebox room and the walls are constructed
/// in the order used in `make_room`.
/// The room is shoebox-shape and the ray travels infinite distance
/// so that it does hit one of the walls.
/// See also:
/// https://github.com/LCAV/pyroomacoustics/blob/df8af24c88a87b5d51c6123087cd3cd2d361286a/pyroomacoustics/libroom_src/room.cpp#L609-L716
template <typename scalar_t>
std::tuple<torch::Tensor, int, scalar_t> find_collision_wall(
    const torch::Tensor& room,
    const torch::Tensor& origin,
    const torch::Tensor& direction // Unit-vector
) {
#define BOOL(x) torch::all(x).template item<bool>()
#define INSIDE(x, y) (BOOL(-EPS < (x)) && BOOL((x) < (y + EPS)))

  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
      3 == room.size(0),
      "Expected room to be 3 dimension, but received ",
      room.sizes());
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
      3 == origin.size(0),
      "Expected origin to be 3 dimension, but received ",
      origin.sizes());
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
      3 == direction.size(0),
      "Expected direction to be 3 dimension, but received ",
      direction.sizes());
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
      BOOL(room > 0), "Room size should be greater than zero. Found: ", room);
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
      INSIDE(origin, room),
      "The origin of ray must be inside the room. Origin: ",
      origin,
      ", room: ",
      room);

  // i is the coordinate in the collision is searched.
  for (unsigned int i = 0; i < 3; ++i) {
    auto dir0 = SCALAR(direction[i]);
    auto abs_dir0 = std::abs(dir0);
    // If the ray is almost parallel to a plane, then we delegate the
    // computation to the other planes.
    if (abs_dir0 < EPS) {
      continue;
    }

    // Check the distance to the facing wall along the coordinate.
    scalar_t distance = (dir0 < 0.)
        ? SCALAR(origin[i]) // Going towards origin
        : SCALAR(room[i] - origin[i]); // Going away from origin
    // sometimes origin is slightly outside of room
    if (distance < 0) {
      distance = 0.;
    }
    auto ratio = distance / abs_dir0;
    int i_increment = dir0 > 0.;

    // Compute the intersection of ray and the wall
    auto intersection = origin + ratio * direction;

    // The intersection can be within the room or outside.
    // If it's inside, the collision point is found.
    //      ^
    //      |           |   Not Good
    //   ---+-----------+---x----
    //      |           |  /
    //      |           | /
    //      |           |/
    //      |           x  Found
    //      |          /|
    //      |         / |
    //      |        o  |
    //      |           |
    //   ---+-----------+-------->
    //     O|           |
    //

    if (INSIDE(intersection, room)) {
      int i_wall = 2 * i + i_increment;
      auto dist = SCALAR((intersection - origin).norm());
      return std::make_tuple(intersection, i_wall, dist);
    }
  }
  // This should not happen
  TORCH_INTERNAL_ASSERT(
      false,
      "Failed to find the intersection. room: ",
      room,
      " origin: ",
      origin,
      " direction: ",
      direction);
#undef INSIDE
#undef BOOL
}
} // namespace rir
} // namespace torchaudio

#undef EPS
#undef SCALAR
