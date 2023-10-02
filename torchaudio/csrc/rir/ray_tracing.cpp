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

//
// Ray tracing implementation. This is heavily based on PyRoomAcoustics:
// https://github.com/LCAV/pyroomacoustics
//
#include <torch/script.h>
#include <torch/torch.h>
#include <torchaudio/csrc/rir/wall.h>
#include <cmath>

namespace torchaudio {
namespace rir {
namespace {

// TODO: remove this once hybrid method is supported
const bool IS_HYBRID_SIM = false;

// TODO: remove this once ISM method is supported
const int ISM_ORDER = 10;

#define EPS ((scalar_t)(1e-5))
#define VAL(x) ((x).template item<scalar_t>())
#define NORM(x) (VAL((x).norm()))
#define MAX(x) (VAL((x).max()))
#define IN_RANGE(x, y) ((-EPS < (x)) && ((x) < (y) + EPS))

template <typename scalar_t, unsigned int D>
const std::array<Wall<scalar_t>, D * 2> make_walls(
    const torch::Tensor& room,
    const torch::Tensor& absorption,
    const torch::Tensor& scattering) {
  if constexpr (D == 2) {
    auto w = room.index({0}).item<scalar_t>();
    auto l = room.index({1}).item<scalar_t>();
    return make_room<scalar_t>(w, l, absorption, scattering);
  }
  if constexpr (D == 3) {
    auto w = room.index({0}).item<scalar_t>();
    auto l = room.index({1}).item<scalar_t>();
    auto h = room.index({2}).item<scalar_t>();
    return make_room<scalar_t>(w, l, h, absorption, scattering);
  }
}

inline double get_energy_coeff(
    const double travel_dist,
    const double mic_radius_sq) {
  double sq = travel_dist * travel_dist;
  auto p_hit = 1. - std::sqrt(1. - mic_radius_sq / std::max(mic_radius_sq, sq));
  return sq * p_hit;
}

/// RayTracer class helper for ray tracing.
/// For attribute description, Python wrapper.
template <typename scalar_t, unsigned int D>
class RayTracer {
  // Provided parameters
  const torch::Tensor& room;
  const torch::Tensor& mic_array;
  const double mic_radius;

  // Values derived from the parameters
  const int num_bands;
  const double mic_radius_sq;
  const bool do_scattering; // Whether scattering is needed (scattering != 0)
  const std::array<Wall<scalar_t>, D * 2> walls; // The walls of the room

  // Runtime value caches
  // Updated at the beginning of the simulation
  double sound_speed = 343.0;
  double distance_thres = 10.0 * sound_speed; // upper bound
  double energy_thres = 0.0; // lower bound
  double hist_bin_width = 0.004; // [second]

 public:
  RayTracer(
      const torch::Tensor& room,
      const torch::Tensor& absorption,
      const torch::Tensor& scattering,
      const torch::Tensor& mic_array,
      const double mic_radius)
      : room(room),
        mic_array(mic_array),
        mic_radius(mic_radius),
        num_bands(absorption.size(0)),
        mic_radius_sq(mic_radius * mic_radius),
        do_scattering(MAX(scattering) > 0.),
        walls(make_walls<scalar_t, D>(room, absorption, scattering)) {}

  // The main (and only) public entry point of this class. The histograms Tensor
  // reference is passed along and modified in the subsequent private method
  // calls. This method spawns num_rays rays in all directions from the source
  // and calls simul_ray() on each of them.
  torch::Tensor compute_histograms(
      const torch::Tensor& origin,
      int num_rays,
      double time_thres,
      double energy_thres_ratio,
      double sound_speed_,
      int num_bins) {
    scalar_t energy_0 = 2. / num_rays;
    auto energies = torch::full({num_bands}, energy_0, room.options());

    auto histograms =
        torch::zeros({mic_array.size(0), num_bins, num_bands}, room.options());

    // Cache runtime parameters
    sound_speed = sound_speed_;
    energy_thres = energy_0 * energy_thres_ratio;
    distance_thres = time_thres * sound_speed;
    hist_bin_width = time_thres / num_bins;

    // TODO: the for loop can be parallelized over num_rays by creating
    // `num_threads` histograms and then sum-reducing them into a single
    // histogram.
    static_assert(D == 2 || D == 3, "Only 2D and 3D are supported.");
    if constexpr (D == 2) {
      scalar_t delta = 2. * M_PI / num_rays;
      for (int i = 0; i < num_rays; ++i) {
        scalar_t phi = i * delta;
        auto dir = torch::tensor({cos(phi), sin(phi)}, room.scalar_type());
        simul_ray(energies, origin, dir, histograms);
      }
    } else {
      scalar_t delta = 2. / num_rays;
      scalar_t increment = M_PI * (3. - std::sqrt(5.)); // phi increment

      for (auto i = 0; i < num_rays; ++i) {
        auto z = (i * delta - 1) + delta / 2.;
        auto rho = std::sqrt(1. - z * z);

        scalar_t phi = i * increment;

        auto x = cos(phi) * rho;
        auto y = sin(phi) * rho;

        auto azimuth = atan2(y, x);
        auto colatitude = atan2(std::sqrt(x * x + y * y), z);

        auto dir = torch::tensor(
            {sin(colatitude) * cos(azimuth),
             sin(colatitude) * sin(azimuth),
             cos(colatitude)},
            room.scalar_type());

        simul_ray(energies, origin, dir, histograms);
      }
    }
    return histograms.transpose(1, 2); // (num_mics, num_bands, num_bins)
  }

 private:
  /// Get the bin index from the distance traveled to a mic.
  inline int get_bin_idx(scalar_t travel_dist_at_mic) {
    auto time_at_mic = travel_dist_at_mic / sound_speed;
    return (int)floor(time_at_mic / hist_bin_width);
  }

  ///
  /// Traces a single ray. phi (horizontal) and theta (vectorical) are the
  /// angles of the ray from the source. Theta is 0 for 2D rooms.  When a ray
  /// intersects a wall, it is reflected and part of its energy is absorbed. It
  /// is also scattered (sent directly to the microphone(s)) according to the
  /// scattering coefficient. When a ray is close to the microphone, its current
  /// energy is recoreded in the output histogram for that given time slot.
  ///
  /// See also:
  /// https://github.com/LCAV/pyroomacoustics/blob/df8af24c88a87b5d51c6123087cd3cd2d361286a/pyroomacoustics/libroom_src/room.cpp#L855-L986
  void simul_ray(
      torch::Tensor& energies,
      torch::Tensor origin,
      torch::Tensor dir,
      torch::Tensor& histograms) {
    auto travel_dist = 0.;
    // To count the number of times the ray bounces on the walls
    // For hybrid generation we add a ray to output only if specular_counter
    // is higher than the ism order.
    int specular_counter = 0;
    while (true) {
      // Find the next hit point
      auto [hit_point, next_wall_index, hit_distance] =
          find_collision_wall<scalar_t, D>(room, origin, dir);

      auto& wall = walls[next_wall_index];

      // Check if the specular ray hits any of the microphone
      if (!(IS_HYBRID_SIM && specular_counter < ISM_ORDER)) {
        // Compute the distance between the line defined by (origin, hit_point)
        // and the center of the microphone (mic_pos)

        for (auto mic_idx = 0; mic_idx < mic_array.size(0); mic_idx++) {
          //
          //                 _  o microphone
          //         to_mic  / |   ^
          //               /       |              wall
          //             /         | mic radious  | |
          //   origin  /           |              | |
          //         /             v              | |
          //       x ---------------------------> |x| collision
          //
          //       | <--------> |
          //       impact_distance
          //       | <--------------------------> |
          //               hit_distance
          //
          torch::Tensor to_mic = mic_array[mic_idx] - origin;
          scalar_t impact_distance = VAL(to_mic.dot(dir));

          // mic is further than the collision point.
          // So microphone did not pick up the sound.
          if (!IN_RANGE(impact_distance, hit_distance)) {
            continue;
          }

          // If the ray hit the coverage of the mic, compute the energy
          if (NORM(to_mic - dir * impact_distance) < mic_radius + EPS) {
            // The length of this last hop
            auto travel_dist_at_mic = travel_dist + std::abs(impact_distance);
            auto coeff = get_energy_coeff(travel_dist_at_mic, mic_radius_sq);
            auto energy = energies / coeff;
            histograms[mic_idx][get_bin_idx(travel_dist_at_mic)] += energy;
          }
        }
      }

      travel_dist += hit_distance;
      energies *= wall.reflection;

      //   Let's shoot the scattered ray induced by the rebound on the wall
      if (do_scattering) {
        scat_ray(histograms, wall, energies, origin, hit_point, travel_dist);
        energies *= (1. - wall.scattering);
      }

      // Check if we reach the thresholds for this ray
      if (travel_dist > distance_thres || VAL(energies.max()) < energy_thres) {
        break;
      }

      // set up for next iteration
      specular_counter += 1;
      dir = reflect(wall, dir);
      origin = hit_point;
    }
  }

  ///
  /// Scatters a ray towards the microphone(s), i.e. records its scattered
  /// energy in the histogram. Called when a ray hits a wall.
  ///
  /// See also:
  /// https://github.com/LCAV/pyroomacoustics/blob/df8af24c88a87b5d51c6123087cd3cd2d361286a/pyroomacoustics/libroom_src/room.cpp#L761-L853
  void scat_ray(
      torch::Tensor& histograms,
      const Wall<scalar_t>& wall,
      const torch::Tensor& energies,
      const torch::Tensor& prev_hit_point,
      const torch::Tensor& hit_point,
      scalar_t travel_dist) {
    for (auto mic_idx = 0; mic_idx < mic_array.size(0); mic_idx++) {
      auto mic_pos = mic_array[mic_idx];
      if (side(wall, mic_pos) != side(wall, prev_hit_point)) {
        continue;
      }

      // As the ray is shot towards the microphone center,
      // the hop dist can be easily computed
      torch::Tensor hit_point_to_mic = mic_pos - hit_point;
      auto hop_dist = NORM(hit_point_to_mic);
      auto travel_dist_at_mic = travel_dist + hop_dist;

      // compute the scattered energy reaching the microphone
      auto h_sq = hop_dist * hop_dist;
      auto p_hit_equal = 1. - std::sqrt(1. - mic_radius_sq / h_sq);
      // cosine angle should be positive, but could be negative if normal is
      // facing out of room so we take abs
      auto p_lambert = (scalar_t)2. * std::abs(cosine(wall, hit_point_to_mic));
      auto scat_trans = wall.scattering * energies * p_hit_equal * p_lambert;

      if (travel_dist_at_mic < distance_thres &&
          MAX(scat_trans) > energy_thres) {
        auto coeff = get_energy_coeff(travel_dist_at_mic, mic_radius_sq);
        auto energy = scat_trans / coeff;
        histograms[mic_idx][get_bin_idx(travel_dist_at_mic)] += energy;
      }
    }
  }
};

///
/// @brief Compute energy histogram via ray tracing. See Python wrapper for
/// detail about parameters and output.
///
torch::Tensor ray_tracing(
    const torch::Tensor& room,
    const torch::Tensor& source,
    const torch::Tensor& mic_array,
    int64_t num_rays,
    const torch::Tensor& absorption,
    const torch::Tensor& scattering,
    double mic_radius,
    double sound_speed,
    double energy_thres,
    double time_thres, // TODO: rename to duration
    double hist_bin_size) {
  // TODO: Raise this to Python layer
  auto num_bins = (int)ceil(time_thres / hist_bin_size);
  switch (room.size(0)) {
    case 2: {
      return AT_DISPATCH_FLOATING_TYPES(
          room.scalar_type(), "ray_tracing_2d", [&] {
            RayTracer<scalar_t, 2> rt(
                room, absorption, scattering, mic_array, mic_radius);
            return rt.compute_histograms(
                source,
                num_rays,
                time_thres,
                energy_thres,
                sound_speed,
                num_bins);
          });
    }
    case 3: {
      return AT_DISPATCH_FLOATING_TYPES(
          room.scalar_type(), "ray_tracing_3d", [&] {
            RayTracer<scalar_t, 3> rt(
                room, absorption, scattering, mic_array, mic_radius);
            return rt.compute_histograms(
                source,
                num_rays,
                time_thres,
                energy_thres,
                sound_speed,
                num_bins);
          });
    }
    default:
      TORCH_CHECK(false, "Only 2D and 3D are supported.");
  }
}

TORCH_LIBRARY_IMPL(torchaudio, CPU, m) {
  m.impl("torchaudio::ray_tracing", torchaudio::rir::ray_tracing);
}

TORCH_LIBRARY_FRAGMENT(torchaudio, m) {
  m.def(
      "torchaudio::ray_tracing(Tensor room, Tensor source, Tensor mic_array, int num_rays, Tensor absorption, Tensor scattering, float mic_radius, float sound_speed, float energy_thres, float time_thres, float hist_bin_size) -> Tensor");
}

} // namespace
} // namespace rir
} // namespace torchaudio
