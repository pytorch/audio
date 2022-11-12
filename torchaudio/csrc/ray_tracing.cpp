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
// TODO: get rid of all the .item().toFloat() and toBool() calls ?
// TODO: is shoebox_size the same as room size?????

#define IS_HYBRID_SIM (false) // TODO: remove
#define ISM_ORDER (10) // TODO: remove
#define EPS (1e-5) // epsilon is set to 0.1 millimeter (100 um)
#define MIC_RADIUS (0.5) // TODO: Make this a parameter?
#define MIC_RADIUS_SQ (MIC_RADIUS * MIC_RADIUS) // TODO: Remove

std::tuple<torch::Tensor, int, float> next_wall_hit(
    torch::Tensor room,
    torch::Tensor start,
    torch::Tensor end,
    float max_dist) {
  int D = room.size(0);
  const static std::vector<std::vector<int>> shoebox_orders = {
      {0, 1, 2}, {1, 0, 2}, {2, 0, 1}};

  torch::Tensor result = torch::zeros(D, torch::kFloat);
  int next_wall_index = -1;
  float hit_dist = max_dist;

  // The direction vector
  torch::Tensor dir = end - start;

  for (auto& d : shoebox_orders) {
    if (d[0] >= D) { // Happens for 2D rooms
      continue;
    }
    float abs_dir0 = std::abs(dir[d[0]].item().toFloat());
    if (abs_dir0 < EPS) {
      continue;
    }

    // distance to plane
    float distance = 0.;

    // this will tell us if the front or back plane is hit
    int ind_inc = 0;

    if (dir[d[0]].item().toFloat() < 0.) {
      result[d[0]] = 0.;
      distance = start[d[0]].item().toFloat();
      ind_inc = 0;
    } else {
      result[d[0]] = room[d[0]];
      distance = (room[d[0]] - start[d[0]]).item().toFloat();
      ind_inc = 1;
    }

    if (distance < EPS) {
      continue;
    }

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

    hit_dist = (result - start).norm().item().toFloat();

    break;

  next_plane:
    (void)0; // no op
  }
  return std::make_tuple(result, next_wall_index, hit_dist);
}

void log_hist(
    torch::Tensor& hist,
    float energy,
    float travel_dist_at_mic,
    float sound_speed,
    float hist_bin_size) {
  // TODO: Also log counts??
  float time_at_mic = travel_dist_at_mic / sound_speed;
  auto bin = floor(time_at_mic / hist_bin_size);
  hist[bin] += energy;
}

int side(torch::Tensor normal, torch::Tensor origin, torch::Tensor pos) {
  auto dot = (pos - origin).dot(normal).item().toFloat();

  if (dot > EPS) {
    return 1;
  } else if (dot < -EPS) {
    return -1;
  } else {
    return 0;
  }
}

void scat_ray(
    torch::Tensor& hist,
    torch::Tensor mic_array,
    float scatter,
    float sound_speed,
    float energy_thres,
    float time_thres,
    float hist_bin_size,
    float transmitted,
    torch::Tensor normal,
    torch::Tensor hit_point,
    float travel_dist) {
  // Convert the energy threshold to transmission threshold (make this more
  // efficient at some point)
  float distance_thres = time_thres * sound_speed;

  // As the ray is shot towards the microphone center,
  // the hop dist can be easily computed
  torch::Tensor hit_point_to_mic = mic_array - hit_point;
  float hop_dist = hit_point_to_mic.norm().item().toFloat();
  float travel_dist_at_mic = travel_dist + hop_dist;

  // compute the scattered energy reaching the microphone
  float h_sq = hop_dist * hop_dist;
  float p_hit_equal = 1.f - sqrt(1.f - MIC_RADIUS_SQ / h_sq);
  float cosine = hit_point_to_mic.dot(normal).item().toFloat() /
      hit_point_to_mic.norm().item().toFloat();
  // cosine angle should be positive, but could be negative if normal is
  // facing out of room so we take abs
  float p_lambert = 2 * std::abs(cosine);
  float scat_trans = scatter * transmitted * p_hit_equal * p_lambert;

  if (travel_dist_at_mic < distance_thres && scat_trans > energy_thres) {
    double r_sq = double(travel_dist_at_mic) * travel_dist_at_mic;
    auto p_hit = (1 - sqrt(1 - MIC_RADIUS_SQ / std::max(MIC_RADIUS_SQ, r_sq)));
    float energy = scat_trans / (r_sq * p_hit);
    log_hist(hist, energy, travel_dist_at_mic, sound_speed, hist_bin_size);
  }
}

void simul_ray(
    torch::Tensor& hist,
    torch::Tensor normals,
    torch::Tensor origins,
    torch::Tensor room,
    torch::Tensor source,
    torch::Tensor mic_array,
    float e_absorption,
    float scatter,
    float sound_speed,
    float energy_thres,
    float time_thres,
    float hist_bin_size,
    float phi,
    float theta,
    float energy_0,
    float max_dist) {
  // TODO: requires_grad of tensors????

  int D = room.size(0);

  torch::Tensor start = source.clone();

  // the direction of the ray (unit vector)
  torch::Tensor dir;
  if (D == 2) {
    dir = torch::tensor({cos(phi), sin(phi)}, torch::kFloat);
  } else if (D == 3) {
    dir = torch::tensor(
        {sin(theta) * cos(phi), sin(theta) * sin(phi), cos(theta)},
        torch::kFloat);
  }

  // The following initializations are arbitrary and does not count since
  // we set the boolean to false
  int next_wall_index = 0;

  // The ray's characteristics
  //   Eigen::ArrayXf transmitted = Eigen::ArrayXf::Ones(n_bands) * energy_0;
  //   Eigen::ArrayXf energy = Eigen::ArrayXf::Ones(n_bands);
  // TODO: handle n_bands
  auto transmitted = energy_0;
  float energy = 1.;
  float travel_dist = 0.;

  // To count the number of times the ray bounces on the walls
  // For hybrid generation we add a ray to output only if specular_counter
  // is higher than the ism order.
  int specular_counter = 0;

  // Convert the energy threshold to transmission threshold
  float e_thres = energy_0 * energy_thres;
  float distance_thres = time_thres * sound_speed;

  torch::Tensor hit_point = torch::zeros(D, torch::kFloat);

  while (true) {
    // Find the next hit point
    float hit_distance = 0;
    std::tie(hit_point, next_wall_index, hit_distance) =
        next_wall_hit(room, start, start + dir * max_dist, max_dist);

    // If no wall is hit (rounding errors), stop the ray
    if (next_wall_index == -1)
      break;

    // Check if the specular ray hits any of the microphone
    // if (!(IS_HYBRID_SIM && specular_counter < ISM_ORDER)) {
    if (true) { // TODO: remove
      // Compute the distance between the line defined by (start, hit_point)
      // and the center of the microphone (mic_pos)

      torch::Tensor to_mic = mic_array - start;
      float impact_distance = to_mic.dot(dir).item().toFloat();

      bool impacts =
          (-EPS < impact_distance) && (impact_distance < hit_distance + EPS);

      // If yes, we compute the ray's transmitted amplitude at the mic and we
      // continue the ray
      if (impacts &&
          ((to_mic - dir * impact_distance).norm().item().toFloat() <
           MIC_RADIUS + EPS)) {
        // The length of this last hop
        float distance = std::abs(impact_distance);

        // Updating travel_time and transmitted amplitude for this ray We DON'T
        // want to modify the variables transmitted amplitude and travel_dist
        // because the ray will continue its way
        float travel_dist_at_mic = travel_dist + distance;
        double r_sq = travel_dist_at_mic * travel_dist_at_mic;
        auto p_hit =
            (1 - sqrt(1 - MIC_RADIUS_SQ / std::max(MIC_RADIUS_SQ, r_sq)));
        energy = transmitted / (r_sq * p_hit);

        log_hist(hist, energy, travel_dist_at_mic, sound_speed, hist_bin_size);
      }
    }

    // Update the characteristics
    travel_dist += hit_distance;
    transmitted *= (1. - e_absorption);

    auto normal = normals[next_wall_index];
    auto origin = origins[next_wall_index];

    // Let's shoot the scattered ray induced by the rebound on the wall
    if ((scatter > 0) &&
        (side(normal, origin, mic_array) == side(normal, origin, start))) {
      scat_ray(
          hist,
          mic_array,
          scatter,
          sound_speed,
          energy_thres,
          time_thres,
          hist_bin_size,
          transmitted,
          normal,
          hit_point,
          travel_dist);
      transmitted *= (1. - scatter);
    }

    // Check if we reach the thresholds for this ray
    if (travel_dist > distance_thres || transmitted < e_thres) {
      break;
    }

    // set up for next iteration
    specular_counter += 1;
    // reflect w.r.t normal while conserving length
    dir = dir - normal * 2 * dir.dot(normal);
    start = hit_point;
    // if (specular_counter == 2) {
    //   break;
    // }
  }
}

// TODO: Handle 3D rooms
torch::Tensor make_normals(torch::Tensor room) {
  // Note that the normals are facing outwards the room:
  //
  //                  ^
  //                  |
  //                  |
  //          -------------------
  //          |       N         |
  //   <----- |W               E| ----->
  //          |       S         |
  //          -------------------
  //                  |
  //                  |
  //                  v

  return torch::tensor(
      {
          {0.f, -1.f}, // South
          {1.f, 0.f}, // East
          {0.f, 1.f}, // North
          {-1.f, 0.f} // West
      },
      torch::kFloat);
}

// See make_normals
torch::Tensor make_origins(torch::Tensor room) {
  return torch::tensor(
      {
          {0.f, 0.f}, // South
          {room[0].item().toFloat(), 0.f}, // East
          {room[0].item().toFloat(), room[1].item().toFloat()}, // North
          {0.f, room[1].item().toFloat()} // West
      },
      torch::kFloat);
}

void ray_tracing_impl(
    torch::Tensor& hist,
    torch::Tensor room,
    torch::Tensor source,
    torch::Tensor mic_array,
    int64_t num_rays,
    float e_absorption,
    float scatter,
    float sound_speed,
    float energy_thres,
    float time_thres,
    float hist_bin_size) {
  float energy_0 = 2. / num_rays;
  int D = room.size(0);

  // Max distance needed to hit a wall = diagonal of room + 1
  float max_dist = room.norm().item().toFloat() + 1.;

  torch::Tensor normals = make_normals(room);
  torch::Tensor origins = make_origins(room);
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

      simul_ray(
          hist,
          normals,
          origins,
          room,
          source,
          mic_array,
          e_absorption,
          scatter,
          sound_speed,
          energy_thres,
          time_thres,
          hist_bin_size,
          azimuth,
          colatitude,
          energy_0,
          max_dist);
    }
  } else if (D == 2) {
    float offset = 2. * M_PI / num_rays;
    for (int i = 0; i < num_rays; ++i) {
      simul_ray(
          hist,
          normals,
          origins,
          room,
          source,
          mic_array,
          e_absorption,
          scatter,
          sound_speed,
          energy_thres,
          time_thres,
          hist_bin_size,
          i * offset,
          0.,
          energy_0,
          max_dist);
    }
  }
}

torch::Tensor ray_tracing(
    torch::Tensor room,
    torch::Tensor source,
    torch::Tensor mic_array,
    int64_t num_rays,
    double e_absorption,
    double scatter,
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
            (float)e_absorption,
            (float)scatter,
            (float)sound_speed,
            (float)energy_thres,
            (float)time_thres,
            (float)hist_bin_size);
      });
  return hist;
}

TORCH_LIBRARY_FRAGMENT(torchaudio, m) {
  m.def(
      "torchaudio::ray_tracing(Tensor room, Tensor source, Tensor mic_array, int num_rays, float e_absorption, float scatter, float sound_speed, float energy_thres, float time_thres, float hist_bin_size) -> Tensor",
      &torchaudio::rir::ray_tracing);
}

} // Anonymous namespace
} // namespace rir
} // namespace torchaudio
