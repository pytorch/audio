#include <math.h>
#include <torch/script.h> // TODO: remove?
#include <torch/torch.h>

namespace torchaudio {
namespace rir {
namespace {

#define IS_HYBRID_SIM (false) // TODO: remove
#define ISM_ORDER (10) // TODO: remove
#define EPS ((scalar_t)(1e-5))

template <typename scalar_t>
class Wall {
 public:
  Wall(
      torch::Tensor _absorption,
      torch::Tensor _scattering,
      torch::Tensor _normal,
      torch::Tensor _origin)
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

  int side(torch::Tensor pos) {
    auto dot = (pos - origin).dot(normal).item<scalar_t>();

    if (dot > EPS) {
      return 1;
    } else if (dot < -EPS) {
      return -1;
    } else {
      return 0;
    }
  }

  torch::Tensor reflect(torch::Tensor dir) {
    return dir - normal * 2 * dir.dot(normal);
  }

  scalar_t cosine(torch::Tensor p) {
    return p.dot(normal).item<scalar_t>() / p.norm().item<scalar_t>();
  }

 private:
  torch::Tensor absorption;
  torch::Tensor reflection;
  torch::Tensor scattering;
  torch::Tensor normal;
  torch::Tensor origin;
};

template <typename scalar_t>
class RayTracer {
 public:
  RayTracer(
      const torch::Tensor _room,
      const torch::Tensor _source,
      const torch::Tensor _mic_array,
      int _num_rays,
      const torch::Tensor _e_absorption,
      const torch::Tensor _scattering,
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
        // e_absorption(_e_absorption),
        // scattering(_scattering),
        mic_radius(_mic_radius),
        mic_radius_sq(mic_radius * mic_radius),
        sound_speed(_sound_speed),
        energy_thres(_energy_thres),
        time_thres(_time_thres),
        hist_bin_size(_hist_bin_size),
        max_dist(room.norm().item<scalar_t>() + 1.),
        D(room.size(0)),
        do_scattering(_scattering.max().template item<scalar_t>() > 0.),
        walls(make_walls(_e_absorption, _scattering)) {}

  void compute_histograms(torch::Tensor& histograms) {
    // TODO: Could probably parallelize call over num_rays? We would need
    // `num_threads` histograms and then sum-reduce them into a single
    // histogram.

    if (D == 3) {
      scalar_t offset = 2. / num_rays;
      scalar_t increment = M_PI * (3. - sqrt(5.)); // phi increment

      for (auto i = 0; i < num_rays; ++i) {
        auto z = (i * offset - 1) + offset / 2.;
        auto rho = sqrt(1. - z * z);

        scalar_t phi = i * increment;

        auto x = cos(phi) * rho;
        auto y = sin(phi) * rho;

        auto azimuth = atan2(y, x);
        auto colatitude = atan2(sqrt(x * x + y * y), z);

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
  const torch::Tensor room;
  const torch::Tensor source;
  const torch::Tensor mic_array;
  int num_rays;
  scalar_t energy_0;
  //   const torch::Tensor e_absorption;
  //   scalar_t scattering;
  scalar_t mic_radius;
  double mic_radius_sq;
  scalar_t sound_speed;
  scalar_t energy_thres;
  scalar_t time_thres;
  scalar_t hist_bin_size;
  scalar_t max_dist; // Max distance needed to hit a wall = diagonal of room + 1
  int D; // Dimension of the room
  const bool do_scattering;
  std::vector<Wall<scalar_t>> walls;

  std::tuple<torch::Tensor, int, scalar_t> next_wall_hit(
      torch::Tensor start,
      torch::Tensor end) {
    const static std::vector<std::vector<int>> shoebox_orders = {
        {0, 1, 2}, {1, 0, 2}, {2, 0, 1}};

    torch::Tensor hit_point = torch::zeros_like(room);
    int next_wall_index = -1;
    auto hit_dist = max_dist;

    torch::Tensor dir = end - start;

    for (auto& d : shoebox_orders) {
      if (d[0] >= D) { // Happens for 2D rooms
        continue;
      }
      auto abs_dir0 = std::abs(dir[d[0]].item<scalar_t>());
      if (abs_dir0 < EPS) {
        continue;
      }

      // distance to plane
      auto distance = 0.;

      // this will tell us if the front or back plane is hit
      int ind_inc = 0;

      if (dir[d[0]].item<scalar_t>() < 0.) {
        hit_point[d[0]] = 0.;
        distance = start[d[0]].item<scalar_t>();
        ind_inc = 0;
      } else {
        hit_point[d[0]] = room[d[0]];
        distance = (room[d[0]] - start[d[0]]).item<scalar_t>();
        ind_inc = 1;
      }

      if (distance < EPS) {
        continue;
      }

      auto ratio = distance / abs_dir0;

      // Now compute the intersection point and verify if intersection happens
      for (auto i = 1; i < D; ++i) {
        hit_point[d[i]] = start[d[i]] + ratio * dir[d[i]];
        // when there is no intersection, we jump to the next plane
        if ((hit_point[d[i]] <= -EPS).template item<bool>() ||
            (room[d[i]] + EPS <= hit_point[d[i]]).template item<bool>())
          goto next_plane;
      }

      // if we get here, there is intersection with this wall
      next_wall_index = 2 * d[0] + ind_inc;

      hit_dist = (hit_point - start).norm().item<scalar_t>();

      break;

    next_plane:
      (void)0; // no op
    }
    return std::make_tuple(hit_point, next_wall_index, hit_dist);
  }

  void log_hist(
      torch::Tensor& histograms,
      int mic_idx,
      torch::Tensor energy,
      scalar_t travel_dist_at_mic) {
    auto time_at_mic = travel_dist_at_mic / sound_speed;
    auto bin = (int)floor(time_at_mic / hist_bin_size);
    auto curr_value = histograms.index({mic_idx, at::indexing::Slice(), bin});
    histograms.index_put_(
        {mic_idx, at::indexing::Slice(), bin}, curr_value + energy);
  }

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

    int next_wall_index;

    auto num_bands = histograms.size(1);
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

    torch::Tensor hit_point = torch::zeros(D, torch::kFloat);

    while (true) {
      // Find the next hit point
      auto hit_distance = 0.;
      std::tie(hit_point, next_wall_index, hit_distance) =
          next_wall_hit(start, start + dir * max_dist);

      // If no wall is hit (rounding errors), stop the ray
      if (next_wall_index == -1)
        break;

      auto wall = walls[next_wall_index];

      // Check if the specular ray hits any of the microphone
      if (!(IS_HYBRID_SIM && specular_counter < ISM_ORDER)) {
        // Compute the distance between the line defined by (start, hit_point)
        // and the center of the microphone (mic_pos)

        for (auto mic_idx = 0; mic_idx < mic_array.size(0); mic_idx++) {
          torch::Tensor to_mic = mic_array[mic_idx] - start;
          auto impact_distance = to_mic.dot(dir).item<scalar_t>();

          bool impacts = (-EPS < impact_distance) &&
              (impact_distance < hit_distance + EPS);

          // If yes, we compute the ray's transmitted amplitude at the mic and
          // we continue the ray

          if (impacts &&
              ((to_mic - dir * impact_distance)
                   .norm()
                   .template item<scalar_t>() < mic_radius + EPS)) {
            // The length of this last hop
            auto distance = std::abs(impact_distance);

            // Updating travel_time and transmitted amplitude for this ray We
            // DON'T want to modify the variables transmitted amplitude and
            // travel_dist because the ray will continue its way
            auto travel_dist_at_mic = travel_dist + distance;
            double r_sq = travel_dist_at_mic * travel_dist_at_mic;
            auto p_hit =
                (1 - sqrt(1 - mic_radius_sq / std::max(mic_radius_sq, r_sq)));
            energy = transmitted / (r_sq * p_hit);

            log_hist(histograms, mic_idx, energy, travel_dist_at_mic);
          }
        }
      }

      // Update the characteristics
      travel_dist = travel_dist + hit_distance;
      transmitted = transmitted * wall.get_reflection();

      //   Let's shoot the scattered ray induced by the rebound on the wall
      if (do_scattering) {
        scat_ray(histograms, wall, transmitted, start, hit_point, travel_dist);
        transmitted = transmitted * (1. - wall.get_scattering());
      }

      // Check if we reach the thresholds for this ray
      if (travel_dist > distance_thres ||
          transmitted.max().template item<scalar_t>() < e_thres) {
        break;
      }

      // set up for next iteration
      specular_counter = specular_counter + 1;
      dir = wall.reflect(dir); // reflect w.r.t normal while conserving length
      start = hit_point;
    }
  }

  void scat_ray(
      torch::Tensor& histograms,
      Wall<scalar_t>& wall,
      torch::Tensor transmitted,
      torch::Tensor prev_hit_point,
      torch::Tensor hit_point,
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
      auto hop_dist = hit_point_to_mic.norm().item<scalar_t>();
      auto travel_dist_at_mic = travel_dist + hop_dist;

      // compute the scattered energy reaching the microphone
      auto h_sq = hop_dist * hop_dist;
      auto p_hit_equal = 1. - sqrt(1. - mic_radius_sq / h_sq);
      // cosine angle should be positive, but could be negative if normal is
      // facing out of room so we take abs
      auto p_lambert = 2 * std::abs(wall.cosine(hit_point_to_mic));
      auto scat_trans =
          wall.get_scattering() * transmitted * p_hit_equal * p_lambert;

      if (travel_dist_at_mic < distance_thres &&
          scat_trans.max().template item<scalar_t>() > energy_thres) {
        double r_sq = double(travel_dist_at_mic) * travel_dist_at_mic;
        auto p_hit =
            (1 - sqrt(1 - mic_radius_sq / std::max(mic_radius_sq, r_sq)));
        auto energy = scat_trans / (r_sq * p_hit);
        log_hist(histograms, mic_idx, energy, travel_dist_at_mic);
      }
    }
  }

  std::vector<Wall<scalar_t>> make_walls(
      const torch::Tensor _e_absorption,
      const torch::Tensor _scattering) {
    scalar_t zero = 0;
    scalar_t W = room[0].template item<scalar_t>();
    scalar_t L = room[1].template item<scalar_t>();

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
      scalar_t H = room[2].template item<scalar_t>();
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
      // For consistency with pyroomacoustits we switch the order of the walls
      // to South East North West
      std::swap(walls[0], walls[3]);
      std::swap(walls[0], walls[2]);
    }
    return walls;
  }
};

torch::Tensor ray_tracing(
    const torch::Tensor room,
    const torch::Tensor source,
    const torch::Tensor mic_array,
    int64_t num_rays,
    const torch::Tensor e_absorption,
    const torch::Tensor scattering,
    double mic_radius,
    double sound_speed,
    double energy_thres,
    double time_thres,
    double hist_bin_size) {
  // TODO: Maybe hist_size should also be bounded from output of ISM, and from
  // eq (4) from https://reuk.github.io/wayverb/ray_tracer.html?
  auto num_mics = mic_array.size(0);
  auto num_bands = e_absorption.size(0);
  auto num_bins = (int)ceil(time_thres / hist_bin_size);
  auto histograms =
      torch::zeros({num_mics, num_bands, num_bins}, room.options());
  //   histograms.requires_grad_(true); // TODO is this needed?

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
  return histograms;
}

TORCH_LIBRARY_FRAGMENT(torchaudio, m) {
  m.def(
      "torchaudio::ray_tracing(Tensor room, Tensor source, Tensor mic_array, int num_rays, Tensor e_absorption, Tensor scattering, float mic_radius, float sound_speed, float energy_thres, float time_thres, float hist_bin_size) -> Tensor",
      &torchaudio::rir::ray_tracing);
}

} // Anonymous namespace
} // namespace rir
} // namespace torchaudio
