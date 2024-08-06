use ndarray::{prelude::*, Data};

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
    (sq_norm_a + sq_norm_b - 2.0 * a.dot(b)).sqrt()
}

fn argmax<S: Data<Elem = f32>>(v: &ArrayBase<S, Ix1>) -> usize {
    let mut i = 0;
    let mut m = v[i];
    for idx in 1..v.shape()[0] {
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
pub fn greedy_minimum_maximum<S: Data<Elem = f32>>(
    data: &ArrayBase<S, Ix2>,
    k: usize,
) -> (Array1<usize>, Array1<usize>, Array1<f32>) {
    let n = data.shape()[0];
    if n <= k {
        // Each point is its own center
        let centers = Array1::<usize>::from_iter(0..n);
        let assignment = Array1::<usize>::from_iter(0..n);
        return (centers, assignment, Array1::<f32>::zeros(n));
    }

    let sq_norms = compute_sq_norms(data);

    let first_center = 0usize;
    let mut centers: Array1<usize> = Array1::zeros(k);
    centers[0] = first_center;
    let mut distances: Array1<f32> = Array1::from_elem(n, f32::INFINITY);
    let mut assignment = Array1::<usize>::zeros(n);

    for i in 0..n {
        distances[i] = eucl(
            &data.row(first_center),
            &data.row(i),
            sq_norms[first_center],
            sq_norms[i],
        );
    }

    for idx in 1..k {
        // FIXME: in a multithreaded context this call deadlocks
        // crate::check_signals();
        let farthest = argmax(&distances);
        centers[idx] = farthest;
        for i in 0..n {
            let d = eucl(
                &data.row(farthest),
                &data.row(i),
                sq_norms[farthest],
                sq_norms[i],
            );
            if d < distances[i] {
                assignment[i] = idx;
                distances[i] = d;
            }
        }
    }

    let mut radii: Array1<f32> = Array1::zeros(k);

    for i in 0..n {
        radii[assignment[i]] = radii[assignment[i]].max(distances[i]);
    }

    (centers, assignment, radii)
}

pub fn assign_closest<S: Data<Elem = f32>>(
    data: &ArrayBase<S, Ix2>,
    centers: &Array1<usize>,
) -> (Array1<usize>, Array1<f32>) {
    let n = data.nrows();
    let n_centers = centers.len();

    let sq_norms = compute_sq_norms(data);

    let mut distances: Array1<f32> = Array1::from_elem(n, f32::INFINITY);
    let mut assignment = Array1::<usize>::zeros(n);

    for i in 0..n {
        for c in 0..n_centers {
            let idx = centers[c];
            let d = eucl(&data.row(idx), &data.row(i), sq_norms[idx], sq_norms[i]);
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
    use crate::test::*;

    use super::greedy_minimum_maximum;

    #[test]
    fn test_anticover() {
        let data = make_blobs(3, 100, 10, 1.0, 10.0);

        let mut last_radius = f32::INFINITY;

        for k in 1..100 {
            let (_centers, _assignment, radii) = greedy_minimum_maximum(&data, k);
            let radius = radii.into_iter().max_by(f32::total_cmp).unwrap();
            assert!(radius < last_radius);
            last_radius = radius;
        }
    }
}
