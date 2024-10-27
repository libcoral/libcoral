use ndarray::{prelude::*, Data};

use crate::metricdata::MetricData;

/// computes the squared norms of the given two dimensional array
pub fn compute_sq_norms<S: Data<Elem = f32>>(data: &ArrayBase<S, Ix2>) -> Array<f32, Ix1> {
    data.rows().into_iter().map(|row| row.dot(&row)).collect()
}

pub fn eucl<S: Data<Elem = f32>>(
    a: &ArrayBase<S, Ix1>,
    b: &ArrayBase<S, Ix1>,
    sq_norm_a: f32,
    sq_norm_b: f32,
) -> f32 {
    let sq_eucl = sq_norm_a + sq_norm_b - 2.0 * a.dot(b);
    if sq_eucl < 0.0 {
        0.0
    } else {
        sq_eucl.sqrt()
    }
}

fn argmax(v: &[f32]) -> usize {
    let mut i = 0;
    let mut m = v[i];
    for idx in 1..v.len() {
        if v[idx] > m {
            i = idx;
            m = v[idx];
        }
    }
    i
}

/// Returns a tuple of two elements: the centers, the assignment, and the radius.
/// The centers array is a vector of indices into the input data.
/// The assignment is a vector of indices into the centers array,
/// with the same length as there are input rows.
pub fn greedy_minimum_maximum<D: MetricData>(
    data: &D,
    k: usize,
) -> (Array1<usize>, Array1<usize>, Array1<f32>) {
    let n = data.num_points();
    if n <= k {
        // Each point is its own center
        let centers = Array1::<usize>::from_iter(0..n);
        let assignment = Array1::<usize>::from_iter(0..n);
        return (centers, assignment, Array1::<f32>::zeros(n));
    }

    let first_center = 0usize;
    let mut centers: Array1<usize> = Array1::zeros(k);
    centers[0] = first_center;
    let mut distances = vec![f32::INFINITY; n];
    let mut new_distances = vec![f32::INFINITY; n];
    let mut assignment = Array1::<usize>::zeros(n);

    data.all_distances(first_center, &mut distances);

    for idx in 1..k {
        // FIXME: in a multithreaded context this call deadlocks
        // crate::check_signals();
        let farthest = argmax(&distances);
        centers[idx] = farthest;
        data.all_distances(farthest, &mut new_distances);
        for i in 0..n {
            if new_distances[i] < distances[i] {
                assignment[i] = idx;
                distances[i] = new_distances[i];
            }
        }
    }

    let mut radii: Array1<f32> = Array1::zeros(k);

    for i in 0..n {
        radii[assignment[i]] = radii[assignment[i]].max(distances[i]);
    }

    (centers, assignment, radii)
}

pub fn assign_closest<D: MetricData>(
    data: &D,
    centers: &Array1<usize>,
) -> (Array1<usize>, Array1<f32>) {
    let n = data.num_points();
    let n_centers = centers.len();

    let mut distances: Array1<f32> = Array1::from_elem(n, f32::INFINITY);
    let mut assignment = Array1::<usize>::zeros(n);

    for i in 0..n {
        for c in 0..n_centers {
            let idx = centers[c];
            let d = data.distance(idx, i);
            if d < distances[i] {
                assignment[i] = c;
                distances[i] = d;
            }
        }
    }

    let mut radii: Array1<f32> = Array1::zeros(n_centers);

    for i in 0..n {
        radii[assignment[i]] = radii[assignment[i]].max(distances[i]);
    }

    (assignment, radii)
}

#[cfg(test)]
mod test {
    use crate::{metricdata::EuclideanData, test::*};

    use super::greedy_minimum_maximum;

    #[test]
    fn test_anticover() {
        let data = make_blobs(3, 100, 10, 1.0, 10.0);
        let data = EuclideanData::new(data);

        let mut last_radius = f32::INFINITY;

        for k in 1..100 {
            let (_centers, _assignment, radii) = greedy_minimum_maximum(&data, k);
            let radius = radii.into_iter().max_by(f32::total_cmp).unwrap();
            dbg!(radius);
            assert!(radius < last_radius);
            last_radius = radius;
        }
    }
}
