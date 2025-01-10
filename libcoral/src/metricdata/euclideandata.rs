use ndarray::{prelude::*, Data, OwnedRepr};

use crate::coreset::NChunks;

use crate::metricdata::{MetricData, Subset};

pub struct EuclideanData<S: Data<Elem = f32>> {
    data: ArrayBase<S, Ix2>,
    squared_norms: Array1<f32>,
}

impl<S: Data<Elem = f32>> EuclideanData<S> {
    pub fn new(data: ArrayBase<S, Ix2>) -> Self {
        let norms = data.rows().into_iter().map(|row| row.dot(&row)).collect();
        Self {
            data,
            squared_norms: norms,
        }
    }
}

impl<S: Data<Elem = f32>> MetricData for EuclideanData<S> {
    fn distance(&self, i: usize, j: usize) -> f32 {
        let sq_eucl = self.squared_norms[i] + self.squared_norms[j]
            - 2.0 * self.data.row(i).dot(&self.data.row(j));
        if sq_eucl < 0.0 {
            0.0
        } else {
            sq_eucl.sqrt()
        }
    }

    fn all_distances(&self, j: usize, out: &mut [f32]) {
        // OPTIMIZE: try using matrix vector product, for instance
        assert_eq!(out.len(), self.data.nrows());
        for (i, oo) in out.iter_mut().enumerate() {
            *oo = self.distance(i, j);
        }
    }

    fn num_points(&self) -> usize {
        self.data.nrows()
    }

    fn dimensions(&self) -> usize {
        self.data.ncols()
    }
}

impl<S: Data<Elem = f32>> Subset for EuclideanData<S> {
    type Out = EuclideanData<OwnedRepr<f32>>;
    fn subset<I: IntoIterator<Item = usize>>(&self, indices: I) -> Self::Out {
        let indices: Vec<usize> = indices.into_iter().collect();
        EuclideanData::new(self.data.select(Axis(0), &indices))
    }
}

impl<S: ndarray::Data<Elem = f32>> NChunks for EuclideanData<S> {
    type Output<'slf> = EuclideanData<ndarray::ViewRepr<&'slf f32>> where S: 'slf;

    fn nchunks(&self, num_chunks: usize) -> impl Iterator<Item = Self::Output<'_>> {
        let points = self.data.nchunks(num_chunks);
        let squared_norms = self.squared_norms.nchunks(num_chunks);
        points.zip(squared_norms).map(|(p, n)| EuclideanData {
            data: p,
            squared_norms: n.to_owned(),
        })
    }

    fn nchunks_size(&self, num_chunks: usize) -> usize {
        (self.num_points() as f64 / num_chunks as f64).ceil() as usize
    }
}
