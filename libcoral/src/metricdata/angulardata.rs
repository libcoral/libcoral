use ndarray::{prelude::*, Data, OwnedRepr, stack};

use std::marker::PhantomData;

use crate::coreset::NChunks;

use crate::metricdata::{MetricData, Subset};

pub struct AngularData<S: Data<Elem=f32>> {
    data: Array2<f32>,
    _marker: PhantomData<S>,
}

impl<S: Data<Elem=f32>> AngularData<S> {
    pub fn new(data: ArrayBase<S, Ix2>) -> Self {
        let normalized_rows: Vec<Array1<f32>> = data
            .axis_iter(Axis(0))
            .map(|row| {
                let norm = row.dot(&row).sqrt();
                row.mapv(|x| x / norm)
            })
            .collect();

        let normalized_data = stack(Axis(0), &normalized_rows.iter().map(Array1::view).collect::<Vec<_>>()).unwrap();

        Self {
            data: normalized_data, 
            _marker: PhantomData,
        }
    }
}

impl<S: Data<Elem = f32>> MetricData for AngularData<S> {
    fn distance(&self, i: usize, j: usize) -> f32 {
        1.0 - self.data.row(i).dot(&self.data.row(j))
    }

    fn all_distances(&self, j: usize, out: &mut [f32]){
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

impl<S: Data<Elem = f32>> Subset for AngularData<S> {
    type Out = AngularData<OwnedRepr<f32>>;
    fn subset<I: IntoIterator<Item = usize>>(&self, indices: I) -> Self::Out {
        let indices: Vec<usize> = indices.into_iter().collect();
        AngularData::new(self.data.select(Axis(0), &indices))
    }
}

impl<S: ndarray::Data<Elem = f32>> NChunks for AngularData<S> {
    type Output<'slf> = AngularData<ndarray::ViewRepr<&'slf f32>> where S: 'slf;

    fn nchunks(&self, num_chunks: usize) -> impl Iterator<Item = Self::Output<'_>> {
        self.data
            .nchunks(num_chunks)
            .map(|p| 
                AngularData { 
                    data: p.to_owned(),
                    _marker: PhantomData
            })
    }

    fn nchunks_size(&self, num_chunks: usize) -> usize {
        (self.num_points() as f64 / num_chunks as f64).ceil() as usize
    }
}