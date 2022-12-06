/*
Copyright (c) 2014-2017 EPFL-LCAV

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

/**
 * Ray tracing implementation. This is heavily based on PyRoomAcoustics:
 * https://github.com/LCAV/pyroomacoustics
 */
#include <torch/script.h>
#include <torch/torch.h>
#include <cmath>

namespace torchaudio {
namespace rir {
namespace {

#define IS_HYBRID_SIM (false) // TODO: remove this once ISM method is supported
#define ISM_ORDER (10) // TODO: remove this once ISM method is supported
#define EPS ((scalar_t)(1e-5))
#define VAL(t) (*((t).template data_ptr<scalar_t>()))

/**
 * Wall helper class. A wall records its own absorption, reflection and
 * scattering coefficient, and exposes a few methods for geometrical operations
 * (e.g. reflection of a ray)
 */
template <typename scalar_t>
class Wall {
 public:
  Wall(
      const torch::Tensor& _absorption,
      const torch::Tensor& _scattering,
      const torch::Tensor& _normal,
      const torch::Tensor& _origin)
      : absorption(_absorption),
        reflection((scalar_t)1. - _absorption),
        scattering(_scattering),
        normal(_normal),
        origin(_origin) {}

  torch::Tensor get_absorption() {
    return absorption;
  }
  torch::Tensor get_reflection() {
    return reflection;
  }
  torch::Tensor get_scattering() {
    return scattering;
  }

  /**
   * Returns the side (-1, 1 or 0) on which a point lies w.r.t. the wall.
   */
  int side(const torch::Tensor& pos) {
    auto dot = VAL((pos - origin).dot(normal));

    if (dot > EPS) {
      return 1;
    } else if (dot < -EPS) {
      return -1;
    } else {
      return 0;
    }
  }

  /**
   * Reflects a ray (dir) on the wall. Preserves norm of vector.
   */
  torch::Tensor reflect(const torch::Tensor& dir) {
    return dir - normal * 2 * dir.dot(normal);
  }

  /**
   * Returns the cosine angle of a ray (dir) with the normal of the wall
   */
  scalar_t cosine(const torch::Tensor& dir) {
    return VAL(dir.dot(normal) / dir.norm());
  }

 private:
  torch::Tensor absorption;
  torch::Tensor reflection; // == 1 - absorption
  torch::Tensor scattering;
  torch::Tensor normal; // The normal to the wall: 2D or 3D vector
  torch::Tensor
      origin; // The origin of the wall: corresponds to an arbitrary corner.
};

/**
 * RayTracer class helper for ray tracing. For attribute description, please see
 * declarations below as well as Python wrapper.
 */
template <typename scalar_t>
class RayTracer {
 public:
  RayTracer(
      const torch::Tensor& _room,
      const torch::Tensor& _source,
      const torch::Tensor& _mic_array,
      int _num_rays,
      const torch::Tensor& _e_absorption,
      const torch::Tensor& _scattering,
      scalar_t _mic_radius,
      scalar_t _sound_speed,
      scalar_t _energy_thres,
      scalar_t _time_thres,
      scalar_t _hist_bin_size)
      : room(_room),
        source(_source),
        mic_array(_mic_array),
        num_rays(_num_rays),
        energy_0(2. / num_rays),
        mic_radius(_mic_radius),
        mic_radius_sq(mic_radius * mic_radius),
        sound_speed(_sound_speed),
        energy_thres(_energy_thres),
        time_thres(_time_thres),
        hist_bin_size(_hist_bin_size),
        max_dist(VAL(room.norm()) + (scalar_t)1.),
        D(room.size(0)),
        do_scattering(VAL(_scattering.max()) > (scalar_t)0.),
        walls(make_walls(_e_absorption, _scattering)) {}

  /**
   * The main (and only) public entry point of this class. The histograms Tensor
   * reference is passed along and modified in the subsequent private method
   * calls. This method spawns num_rays rays in all directions from the source
   * and calls simul_ray() on each of them.
   */
  void compute_histograms(torch::Tensor& histograms) {
    // TODO: the for loop can be parallelized over num_rays by creating
    // `num_threads` histograms and then sum-reducing them into a single
    // histogram.

    if (D == 3) {
      scalar_t offset = 2. / num_rays;
      scalar_t increment = M_PI * (3. - std::sqrt(5.)); // phi increment

      for (auto i = 0; i < num_rays; ++i) {
        auto z = (i * offset - 1) + offset / 2.;
        auto rho = std::sqrt(1. - z * z);

        scalar_t phi = i * increment;

        auto x = cos(phi) * rho;
        auto y = sin(phi) * rho;

        auto azimuth = atan2(y, x);
        auto colatitude = atan2(std::sqrt(x * x + y * y), z);

        simul_ray(histograms, azimuth, colatitude);
      }
    } else if (D == 2) {
      scalar_t offset = 2. * M_PI / num_rays;
      for (int i = 0; i < num_rays; ++i) {
        simul_ray(histograms, i * offset, 0.);
      }
    }
  }

 private:
  const torch::Tensor& room;
  const torch::Tensor& source;
  const torch::Tensor& mic_array;
  int num_rays;
  scalar_t energy_0; // initial energy of each ray
  scalar_t mic_radius;
  double mic_radius_sq;
  scalar_t sound_speed;
  scalar_t energy_thres;
  scalar_t time_thres;
  scalar_t hist_bin_size;
  scalar_t max_dist; // Max distance needed to hit a wall = diagonal of room + 1
  int D; // Dimension of the room
  const bool do_scattering; // Whether scattering is needed (scattering != 0)
  std::vector<Wall<scalar_t>> walls; // The walls of the room

  /**
   * From a ray vector defined by its start and end, returns the next wall hit
   * as a 3-tuple:
   * - the hit point on the wall: 2D or 3D tensor
   * - the index of the wall (as in the .walls vector attribute)
   * - the distance from the start to the wall
   */
  std::tuple<torch::Tensor, int, scalar_t> next_wall_hit(
      const torch::Tensor& start,
      const torch::Tensor& end) {
    const static std::vector<std::vector<int>> shoebox_orders = {
        {0, 1, 2}, {1, 0, 2}, {2, 0, 1}};

    torch::Tensor hit_point = torch::zeros_like(room);
    int next_wall_index = -1;
    auto hit_dist = max_dist;

    torch::Tensor dir = end - start;

    auto dir_a = dir.accessor<scalar_t, 1>();
    auto hit_point_a = hit_point.accessor<scalar_t, 1>();
    auto start_a = start.accessor<scalar_t, 1>();
    auto room_a = room.accessor<scalar_t, 1>();

    for (auto& d : shoebox_orders) {
      if (d[0] >= D) { // Happens for 2D rooms
        continue;
      }
      auto abs_dir0 = std::abs(dir_a[d[0]]);
      if (abs_dir0 < EPS) {
        continue;
      }

      // distance to plane
      auto distance = 0.;

      // this will tell us if the front or back plane is hit
      int ind_inc = 0;

      if (dir_a[d[0]] < 0.) {
        hit_point[d[0]] = 0.;
        distance = start_a[d[0]];
        ind_inc = 0;
      } else {
        hit_point_a[d[0]] = room_a[d[0]];
        distance = room_a[d[0]] - start_a[d[0]];
        ind_inc = 1;
      }

      if (distance < EPS) {
        continue;
      }

      auto ratio = distance / abs_dir0;

      // Now compute the intersection point and verify if intersection happens
      auto intersection_found = true;
      for (auto i = 1; i < D; ++i) {
        hit_point_a[d[i]] = start_a[d[i]] + ratio * dir_a[d[i]];
        // when there is no intersection, we jump to the next plane
        if ((hit_point_a[d[i]] <= -EPS) ||
            (room_a[d[i]] + EPS <= hit_point_a[d[i]])) {
          intersection_found = false;
          break; // check next plane
        }
      }

      if (intersection_found) {
        next_wall_index = 2 * d[0] + ind_inc;
        hit_dist = VAL((hit_point - start).norm());
        break;
      }
    }
    return std::make_tuple(hit_point, next_wall_index, hit_dist);
  }

  /**
   * Add energy level to the output histogram for a given microphone and a given
   * time-bin (computed from travel_dist_at_mic)
   */
  void log_hist(
      torch::Tensor& histograms,
      int mic_idx,
      const torch::Tensor& energy,
      scalar_t travel_dist_at_mic) {
    auto time_at_mic = travel_dist_at_mic / sound_speed;
    auto bin = (int)floor(time_at_mic / hist_bin_size);
    histograms[mic_idx][bin] += energy;
  }

  /**
   * Traces a single ray. phi (horizontal) and theta (vectorical) are the angles
   * of the ray from the source. Theta is 0 for 2D rooms.  When a ray intersects
   * a wall, it is reflected and part of its energy is absorbed. It is also
   * scattered (sent directly to the microphone(s)) according to the scattering
   * coefficient. When a ray is close to the microphone, its current energy is
   * recoreded in the output histogram for that given time slot.
   */
  void simul_ray(torch::Tensor& histograms, scalar_t phi, scalar_t theta) {
    torch::Tensor start = source.clone();

    // the direction of the ray (unit vector)
    torch::Tensor dir;
    if (D == 2) {
      dir = torch::tensor({cos(phi), sin(phi)}, room.scalar_type());
    } else if (D == 3) {
      dir = torch::tensor(
          {sin(theta) * cos(phi), sin(theta) * sin(phi), cos(theta)},
          room.scalar_type());
    }

    int next_wall_index = -1;

    auto num_bands = histograms.size(2);
    auto transmitted = torch::ones({num_bands}) * energy_0;
    auto energy = torch::ones({num_bands});
    auto travel_dist = 0.;

    // To count the number of times the ray bounces on the walls
    // For hybrid generation we add a ray to output only if specular_counter
    // is higher than the ism order.
    int specular_counter = 0;

    // Convert the energy threshold to transmission threshold
    auto e_thres = energy_0 * energy_thres;
    auto distance_thres = time_thres * sound_speed;

    torch::Tensor hit_point = torch::zeros(D, room.scalar_type());

    while (true) {
      // Find the next hit point
      auto hit_distance = 0.;
      std::tie(hit_point, next_wall_index, hit_distance) =
          next_wall_hit(start, start + dir * max_dist);

      // If no wall is hit (rounding errors), stop the ray
      if (next_wall_index == -1) {
        break;
      }

      auto wall = walls[next_wall_index];

      // Check if the specular ray hits any of the microphone
      if (!(IS_HYBRID_SIM && specular_counter < ISM_ORDER)) {
        // Compute the distance between the line defined by (start, hit_point)
        // and the center of the microphone (mic_pos)

        for (auto mic_idx = 0; mic_idx < mic_array.size(0); mic_idx++) {
          torch::Tensor to_mic = mic_array[mic_idx] - start;
          scalar_t impact_distance = VAL(to_mic.dot(dir));

          bool impacts = (-EPS < impact_distance) &&
              (impact_distance < hit_distance + EPS);

          // If yes, we compute the ray's transmitted amplitude at the mic and
          // we continue the ray
          if (impacts &&
              (VAL((to_mic - dir * impact_distance).norm()) <
               mic_radius + EPS)) {
            // The length of this last hop
            auto distance = std::abs(impact_distance);

            auto travel_dist_at_mic = travel_dist + distance;
            double r_sq = travel_dist_at_mic * travel_dist_at_mic;
            auto p_hit =
                ((scalar_t)1. -
                 std::sqrt(
                     (scalar_t)1. -
                     mic_radius_sq / std::max(mic_radius_sq, r_sq)));
            energy = transmitted / (r_sq * p_hit);

            log_hist(histograms, mic_idx, energy, travel_dist_at_mic);
          }
        }
      }

      travel_dist += hit_distance;
      transmitted *= wall.get_reflection();

      //   Let's shoot the scattered ray induced by the rebound on the wall
      if (do_scattering) {
        scat_ray(histograms, wall, transmitted, start, hit_point, travel_dist);
        transmitted = transmitted * (1. - wall.get_scattering());
      }

      // Check if we reach the thresholds for this ray
      if (travel_dist > distance_thres || VAL(transmitted.max()) < e_thres) {
        break;
      }

      // set up for next iteration
      specular_counter = specular_counter + 1;
      dir = wall.reflect(dir); // reflect w.r.t normal while conserving length
      start = hit_point;
    }
  }

  /**
   * Scatters a ray towards the microphone(s), i.e. records its scattered energy
   * in the histogram. Called when a ray hits a wall.
   */
  void scat_ray(
      torch::Tensor& histograms,
      Wall<scalar_t>& wall,
      const torch::Tensor& transmitted,
      const torch::Tensor& prev_hit_point,
      const torch::Tensor& hit_point,
      scalar_t travel_dist) {
    auto distance_thres = time_thres * sound_speed;

    for (auto mic_idx = 0; mic_idx < mic_array.size(0); mic_idx++) {
      auto mic_pos = mic_array[mic_idx];
      if (wall.side(mic_pos) != wall.side(prev_hit_point)) {
        continue;
      }

      // As the ray is shot towards the microphone center,
      // the hop dist can be easily computed
      torch::Tensor hit_point_to_mic = mic_pos - hit_point;
      auto hop_dist = VAL(hit_point_to_mic.norm());
      auto travel_dist_at_mic = travel_dist + hop_dist;

      // compute the scattered energy reaching the microphone
      auto h_sq = hop_dist * hop_dist;
      auto p_hit_equal =
          (scalar_t)1. - std::sqrt((scalar_t)1. - mic_radius_sq / h_sq);
      // cosine angle should be positive, but could be negative if normal is
      // facing out of room so we take abs
      auto p_lambert = (scalar_t)2. * std::abs(wall.cosine(hit_point_to_mic));
      auto scat_trans =
          wall.get_scattering() * transmitted * p_hit_equal * p_lambert;

      if (travel_dist_at_mic < distance_thres &&
          VAL(scat_trans.max()) > energy_thres) {
        double r_sq = double(travel_dist_at_mic) * travel_dist_at_mic;
        auto p_hit =
            ((scalar_t)1. -
             std::sqrt(
                 (scalar_t)1. - mic_radius_sq / std::max(mic_radius_sq, r_sq)));
        auto energy = scat_trans / (r_sq * p_hit);
        log_hist(histograms, mic_idx, energy, travel_dist_at_mic);
      }
    }
  }

  /**
   * Creates the walls based on the input to the constructor.
   * Since the room is always a shoebox we can hard-code values.
   * Normals are vectors facing *outwards* the room, and origins are arbitrary
   * corners of each wall.
   */
  std::vector<Wall<scalar_t>> make_walls(
      const torch::Tensor& _e_absorption,
      const torch::Tensor& _scattering) {
    auto room_a = room.accessor<scalar_t, 1>();
    scalar_t zero = 0;
    scalar_t W = room_a[0];
    scalar_t L = room_a[1];

    std::vector<Wall<scalar_t>> walls;

    torch::Tensor normals;
    torch::Tensor origins;

    if (D == 2) {
      normals = torch::tensor(
          {
              {-1, 0}, // West
              {1, 0}, // East
              {0, -1}, // South
              {0, 1}, // North
          },
          room.scalar_type());

      origins = torch::tensor(
          {
              {zero, L}, // West
              {W, zero}, // East
              {zero, zero}, // South
              {W, L}, // North
          },
          room.scalar_type());
    } else {
      scalar_t H = room_a[2];
      normals = torch::tensor(
          {
              {-1, 0, 0}, // West
              {1, 0, 0}, // East
              {0, -1, 0}, // South
              {0, 1, 0}, // North
              {0, 0, -1}, // Floor
              {0, 0, 1} // Ceiling
          },
          room.scalar_type());
      origins = torch::tensor(
          {
              {zero, L, zero}, // West
              {W, zero, zero}, // East
              {zero, zero, zero}, // South
              {W, L, zero}, // North
              {W, zero, zero}, // Floor
              {W, zero, H} // Ceil
          },
          room.scalar_type());
    }

    for (auto i = 0; i < normals.size(0); i++) {
      walls.push_back(Wall<scalar_t>(
          _e_absorption.index({at::indexing::Slice(), i}),
          _scattering.index({at::indexing::Slice(), i}),
          normals[i],
          origins[i]));
    }
    if (D == 2) {
      // For consistency with pyroomacoustics we switch the order of the walls
      // to South East North West
      std::swap(walls[0], walls[3]);
      std::swap(walls[0], walls[2]);
    }
    return walls;
  }
};

/**
 * @brief Compute energy histogram via ray tracing. See Python wrapper for
 * detail about parameters and output.
 */
torch::Tensor ray_tracing(
    const torch::Tensor& room,
    const torch::Tensor& source,
    const torch::Tensor& mic_array,
    int64_t num_rays,
    const torch::Tensor& e_absorption,
    const torch::Tensor& scattering,
    double mic_radius,
    double sound_speed,
    double energy_thres,
    double time_thres,
    double hist_bin_size) {
  auto num_mics = mic_array.size(0);
  auto num_bands = e_absorption.size(0);
  auto num_bins = (int)ceil(time_thres / hist_bin_size);
  // Output shape will actually be (num_mics, num_bands, num_bins). We let
  // num_bands be the last dim during computation to make indexing cleaner in
  // log_hist(), and to optimize for cache line hit.
  auto histograms =
      torch::zeros({num_mics, num_bins, num_bands}, room.options());

  AT_DISPATCH_FLOATING_TYPES(room.scalar_type(), "ray_tracing", [&] {
    RayTracer<scalar_t> rt(
        room,
        source,
        mic_array,
        num_rays,
        e_absorption,
        scattering,
        mic_radius,
        sound_speed,
        energy_thres,
        time_thres,
        hist_bin_size);
    rt.compute_histograms(histograms);
  });
  histograms = histograms.transpose(
      1, 2); // back into shape (num_mics, num_bands, num_bins)
  return histograms;
}

} // Anonymous namespace

TORCH_LIBRARY_IMPL(torchaudio, CPU, m) {
  m.impl("torchaudio::ray_tracing", torchaudio::rir::ray_tracing);
}

} // namespace rir
} // namespace torchaudio

TORCH_LIBRARY_FRAGMENT(torchaudio, m) {
  m.def(
      "torchaudio::ray_tracing(Tensor room, Tensor source, Tensor mic_array, int num_rays, Tensor e_absorption, Tensor scattering, float mic_radius, float sound_speed, float energy_thres, float time_thres, float hist_bin_size) -> Tensor");
}
