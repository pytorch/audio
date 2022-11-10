#include <math.h>
#include <stdio.h>
#include <torch/script.h> // TODO: remove?
#include <torch/torch.h>
using namespace torch::indexing; // TODO: remove?

namespace torchaudio {
namespace rir {

namespace {

// TODO: use double verywhere instead of float?
// TODO: use const where relevant?
// TODO: get rid of all the .item().toDouble() and toBool() calls ?
// TODO: is shoebox_size the same as room size?????

#define IS_HYBRID_SIM (false) // TODO: remove
#define ISM_ORDER (10) // TODO: remove
#define MAX_DIST \
  (50.) // TODO:remove. Note: this is the max distance it takes to hit a wall,
        // not the max distance that a ray can go over
#define EPS (1e-5) // epsilon is set to 0.1 millimeter (100 um)
#define MIC_RADIUS (0.15) // TODO: Make this a parameter?
#define MIC_RADIUS_SQ (0.15 * 0.15) // TODO: Remove
#define SCATTER (0.1) // TODO: Remove

std::tuple<torch::Tensor, int, float> next_wall_hit(
    torch::Tensor room,
    torch::Tensor start,
    torch::Tensor end,
    bool scattered_ray // TODO: probably remove
) {
  int D = room.size(0);
  const static std::vector<std::vector<int>> shoebox_orders = {
      {0, 1, 2}, {1, 0, 2}, {2, 0, 1}};

  torch::Tensor result = torch::zeros(D, torch::kFloat);
  int next_wall_index = -1;
  double hit_dist = MAX_DIST;

  // There are no obstructing walls in shoebox rooms
  if (scattered_ray)
    return std::make_tuple(result, -1, 0.);

  // The direction vector
  torch::Tensor dir = end - start;

  for (auto& d : shoebox_orders) {
    double abs_dir0 = std::abs(dir[d[0]].item().toDouble());
    if (abs_dir0 < EPS) {
      continue;
    }

    // distance to plane
    double distance = 0.;

    // this wil tell us if the front or back plane is hit
    int ind_inc = 0;

    if (dir[d[0]].item().toDouble() < 0.) {
      result[d[0]] = 0.;
      distance = start[d[0]].item().toDouble();
      ind_inc = 0;
    } else {
      result[d[0]] = room[d[0]];
      distance = (room[d[0]] - start[d[0]]).item().toDouble();
      ind_inc = 1;
    }

    if (distance < EPS)
      continue;

    float ratio = distance / abs_dir0;

    // Now compute the intersection point and verify if intersection happens
    for (auto i = 1; i < D; ++i) {
      result[d[i]] = start[d[i]] + ratio * dir[d[i]];
      // when there is no intersection, we jump to the next plane
      if ((result[d[i]] <= -EPS).item().toBool() ||
          (room[d[i]] + EPS <= result[d[i]]).item().toBool())
        goto next_plane; // TODO: avoid goto??
    }

    // if we get here, there is intersection with this wall
    next_wall_index = 2 * d[0] + ind_inc;

    hit_dist = (result - start).norm().item().toDouble();

    break;

  next_plane:
    (void)0; // no op
  }
  return std::make_tuple(result, next_wall_index, hit_dist);
}

void simul_ray(
    torch::Tensor& hist,
    torch::Tensor room,
    torch::Tensor source,
    torch::Tensor mic_array,
    double e_absorption,
    double sound_speed,
    double energy_thres,
    double time_thres,
    double hist_bin_size,
    float phi,
    float theta,
    double energy_0) {
  // TODO: requires_grad of tensors????

  int D = room.size(0);

  torch::Tensor start = source.clone();

  // the direction of the ray (unit vector)
  torch::Tensor dir = torch::zeros(D, torch::kFloat);
  // TODO: There must be better way to initialize the tensors??
  if (D == 2) {
    //   dir.head(2) = Eigen::Vector2f(cos(phi), sin(phi));
    dir[0] = cos(phi);
    dir[1] = sin(phi);
  } else if (D == 3) {
    //   dir.head(3) = Eigen::Vector3f(sin(theta) * cos(phi), sin(theta) *
    //   sin(phi), cos(theta));
    dir[0] = sin(theta) * cos(phi);
    dir[1] = sin(theta) * sin(phi);
    dir[2] = cos(theta);
  }

  // The following initializations are arbitrary and does not count since
  // we set the boolean to false
  int next_wall_index = 0;

  // The ray's characteristics
  //   Eigen::ArrayXf transmitted = Eigen::ArrayXf::Ones(n_bands) * energy_0;
  //   Eigen::ArrayXf energy = Eigen::ArrayXf::Ones(n_bands);
  // TODO: handle n_bands
  auto transmitted = energy_0;
  double energy = 1.;
  double travel_dist = 0.;

  // To count the number of times the ray bounces on the walls
  // For hybrid generation we add a ray to output only if specular_counter
  // is higher than the ism order.
  int specular_counter = 0;

  // Convert the energy threshold to transmission threshold
  double e_thres = energy_0 * energy_thres;
  double distance_thres = time_thres * sound_speed;

  torch::Tensor hit_point = torch::zeros(D, torch::kFloat);

  while (true) {
    // Find the next hit point
    double hit_distance = 0;
    std::tie(hit_point, next_wall_index, hit_distance) =
        next_wall_hit(room, start, start + dir * MAX_DIST, false);

    // If no wall is hit (rounding errors), stop the ray
    if (next_wall_index == -1)
      break;

    // TODO: Do we need this wall variable??
    // Intersected wall
    // Wall<D> &wall = walls[next_wall_index];

    // Check if the specular ray hits any of the microphone
    if (!(IS_HYBRID_SIM && specular_counter < ISM_ORDER)) {
      // Compute the distance between the line defined by (start, hit_point)
      // and the center of the microphone (mic_pos)

      torch::Tensor to_mic = mic_array - start;
      double impact_distance = to_mic.dot(dir).item().toDouble();

      bool impacts =
          (-EPS < impact_distance) && (impact_distance < hit_distance + EPS);

      // If yes, we compute the ray's transmitted amplitude at the mic and we
      // continue the ray
      if (impacts &&
          ((to_mic - dir * impact_distance).norm().item().toDouble() <
           MIC_RADIUS + EPS)) {
        // The length of this last hop
        double distance = std::abs(impact_distance);

        // Updating travel_time and transmitted amplitude for this ray We DON'T
        // want to modify the variables transmitted amplitude and travel_dist
        // because the ray will continue its way
        double travel_dist_at_mic = travel_dist + distance;
        double r_sq = travel_dist_at_mic * travel_dist_at_mic;
        auto p_hit =
            (1 - sqrt(1 - MIC_RADIUS_SQ / std::max(MIC_RADIUS_SQ, r_sq)));
        energy = transmitted / (r_sq * p_hit);

        // microphones[k].log_histogram(travel_dist_at_mic, energy, start);
        double time_at_mic = travel_dist_at_mic / sound_speed;
        auto bin = floor(time_at_mic / hist_bin_size);
        hist[bin] += energy; // TODO: Also log counts??
      }
    }

    // Update the characteristics
    travel_dist += hit_distance;
    // transmitted *= wall.get_energy_reflection();
    transmitted *= (1. - e_absorption);

    // Let's shoot the scattered ray induced by the rebound on the wall
    if (SCATTER > 0) {
      // Shoot the scattered ray

      // TODOODODODODODODODODODO
      //   scat_ray(transmitted, wall, start, hit_point, travel_dist);

      // The overall ray's energy gets decreased by the total
      // amount of scattered energy
      transmitted *= (1. - SCATTER);
    }

    // Check if we reach the thresholds for this ray
    if (travel_dist > distance_thres || transmitted < e_thres) {
      break;
    }

    // // set up for next iteration
    // specular_counter += 1;
    // dir = wall.normal_reflect(dir); // conserves length
    // start = hit_point;

    break; // TODO: remove!!!!!
  }
}

void ray_tracing_impl(
    torch::Tensor& hist,
    torch::Tensor room,
    torch::Tensor source,
    torch::Tensor mic_array,
    int64_t num_rays,
    double e_absorption,
    double sound_speed,
    double energy_thres,
    double time_thres,
    double hist_bin_size) {
  double energy_0 = 2. / num_rays; // TODO: just move that into simul_ray??
  int D = room.size(0);

  // TODO: Could probably parallelize call over num_rays? We would need
  // `num_threads` histograms and then sum-reduce them into a single histogram.

  if (D == 3) {
    auto offset = 2. / num_rays;
    auto increment = M_PI * (3. - sqrt(5.)); // phi increment

    for (auto i = 0; i < num_rays; ++i) {
      auto z = (i * offset - 1) + offset / 2.;
      auto rho = sqrt(1. - z * z);

      float phi = i * increment;

      auto x = cos(phi) * rho;
      auto y = sin(phi) * rho;

      auto azimuth = atan2(y, x);
      auto colatitude = atan2(sqrt(x * x + y * y), z);

      //   simul_ray(azimuth, colatitude, source_pos, energy_0);
    }
  } else if (D == 2) {
    float offset = 2. * M_PI / num_rays;
    for (auto i = 0; i < num_rays; ++i) {
      simul_ray(
          hist,
          room,
          source,
          mic_array,
          e_absorption,
          sound_speed,
          energy_thres,
          time_thres,
          hist_bin_size,
          i * offset,
          0.,
          energy_0);
    }
  }
}

torch::Tensor ray_tracing(
    torch::Tensor room,
    torch::Tensor source,
    torch::Tensor mic_array,
    int64_t num_rays,
    double e_absorption,
    double sound_speed,
    double energy_thres,
    double time_thres,
    double hist_bin_size) {
  // TODO: Maybe hist_size should also be bounded from output of ISM, and from
  // eq (4) from https://reuk.github.io/wayverb/ray_tracer.html
  auto hist_size = ceil(time_thres / hist_bin_size);
  auto hist = torch::zeros(hist_size, torch::kFloat);
  AT_DISPATCH_FLOATING_TYPES(
      at::ScalarType::Float, "ray_tracing", [&] { // TODO: AND_HALF???
        ray_tracing_impl(
            hist,
            room,
            source,
            mic_array,
            num_rays,
            e_absorption,
            sound_speed,
            energy_thres,
            time_thres,
            hist_bin_size);
      });
  return hist;
}

TORCH_LIBRARY_FRAGMENT(torchaudio, m) {
  m.def(
      "torchaudio::ray_tracing(Tensor room, Tensor source, Tensor mic_array, int num_rays, float e_absorption, float sound_speed, float energy_thres, float time_thres, float hist_bin_size) -> Tensor",
      &torchaudio::rir::ray_tracing);
}

} // Anonymous namespace
} // namespace rir
} // namespace torchaudio
