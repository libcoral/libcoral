pub mod coreset;
pub mod diversity;
pub mod gmm;

#[cfg(test)]
mod test {
    use ndarray::prelude::*;
    use ndarray_rand::rand::prelude::*;
    use ndarray_rand::rand_distr::{Normal, Uniform};
    use ndarray_rand::RandomExt;

    pub(crate) fn make_blobs(
        dims: usize,
        point_per_blob: usize,
        n_blobs: usize,
        stddev: f32,
        centers_side: f32,
    ) -> Array2<f32> {
        let mut rng = thread_rng();
        let centers_dist = Uniform::new(-centers_side, centers_side);

        let mut points = Array2::<f32>::zeros((0, dims));

        for _ in 0..n_blobs {
            let c_coords = Array1::random_using(dims, centers_dist, &mut rng);
            let distr = Normal::new(0.0, stddev).unwrap();

            let blob = Array2::random_using((point_per_blob, dims), distr, &mut rng);
            let blob = blob + c_coords;

            points.append(Axis(0), blob.view()).unwrap();
        }

        points
    }
}
