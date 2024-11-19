use ndarray::{concatenate, prelude::*, Data};

use crate::metricdata::MetricData;

pub trait WeightCoresetPoints: Sync + Send {
    fn weight_coreset_points(
        &self,
        n_coreset_points: usize,
        assignment: &Array1<usize>,
    ) -> ArrayD<usize>;
}

impl<F: Fn(usize, &Array1<usize>) -> ArrayD<usize> + Sync + Send> WeightCoresetPoints for F {
    fn weight_coreset_points(
        &self,
        n_coreset_points: usize,
        assignment: &Array1<usize>,
    ) -> ArrayD<usize> {
        self(n_coreset_points, assignment)
    }
}

#[derive(Clone, Copy)]
pub struct WeightByCount;
impl WeightCoresetPoints for WeightByCount {
    fn weight_coreset_points(
        &self,
        n_coreset_points: usize,
        assignment: &Array1<usize>,
    ) -> ArrayD<usize> {
        let mut weights = ArrayD::zeros(IxDyn(&[n_coreset_points]));
        for a in assignment.iter() {
            weights[*a] += 1;
        }
        weights
    }
}

/// "extract" the coreset points from the subset pertaining
/// to the a given center. The function is presented with a slice of
/// indices into the dataset vectors as well as the (optional) ancillary data.
pub trait ExtractCoresetPoints: Sync + Send {
    /// Returns an array of indices into the data pointing to the extracted coreset points.
    fn extract_coreset_points(&self, center_idx: usize, assigned: &[usize]) -> Array1<usize>;
}

impl<F: Fn(usize, &[usize]) -> Array1<usize> + Sync + Send> ExtractCoresetPoints for F {
    fn extract_coreset_points(&self, center_idx: usize, assigned: &[usize]) -> Array1<usize> {
        self(center_idx, assigned)
    }
}

fn weight_by_count(assignment: &Array1<usize>) -> ArrayD<usize> {
    let n_coreset_points = assignment.iter().max().unwrap() + 1;
    let mut weights = ArrayD::zeros(IxDyn(&[n_coreset_points]));
    for a in assignment.iter() {
        weights[*a] += 1;
    }
    weights
}

pub trait Compose {
    fn compose(a: Self, b: Self) -> Self;
}

impl<T: Clone> Compose for Vec<T> {
    fn compose(mut a: Self, b: Self) -> Self {
        a.extend_from_slice(&b);
        a
    }
}

impl<T: Copy> Compose for Array1<T> {
    fn compose(a: Self, b: Self) -> Self {
        concatenate![Axis(0), a, b]
    }
}

impl<T: Copy> Compose for Array2<T> {
    fn compose(a: Self, b: Self) -> Self {
        concatenate![Axis(0), a, b]
    }
}

impl<T: Copy> Compose for ArrayD<T> {
    fn compose(a: Self, b: Self) -> Self {
        concatenate![Axis(0), a, b]
    }
}

#[derive(Clone)]
pub struct FittedCoreset {
    /// the indices _in the original dataset_
    coreset_indices: Array1<usize>,
    radius: Array1<f32>,
    weights: ArrayD<usize>,
    assignment: Array1<usize>,
}

impl FittedCoreset {
    pub fn indices(&self) -> ArrayView1<usize> {
        self.coreset_indices.view()
    }

    pub fn radii(&self) -> ArrayView1<f32> {
        self.radius.view()
    }

    pub fn weights(&self) -> ArrayViewD<usize> {
        self.weights.view()
    }

    pub fn assignment(&self) -> ArrayView1<usize> {
        self.assignment.view()
    }

    /// Maps the given indices into the coreset to indices in the original dataset
    pub fn invert_index<S: Data<Elem = usize>>(
        &self,
        indices: &ArrayBase<S, Ix1>,
    ) -> Array1<usize> {
        indices.map(|i| self.coreset_indices[*i])
    }
}

impl Compose for FittedCoreset {
    fn compose(a: Self, b: Self) -> Self {
        Self {
            coreset_indices: Compose::compose(a.coreset_indices, b.coreset_indices),
            radius: Compose::compose(a.radius, b.radius),
            weights: Compose::compose(a.weights, b.weights),
            assignment: Compose::compose(a.assignment, b.assignment),
        }
    }
}

// #[derive(Clone)]
pub struct CoresetBuilder<'slf> {
    tau: usize,
    threads: usize,
    weighter: Option<Box<dyn WeightCoresetPoints + 'slf>>,
    extractor: Option<Box<dyn ExtractCoresetPoints + 'slf>>,
}

impl<'slf> CoresetBuilder<'slf> {
    pub fn with_tau(tau: usize) -> CoresetBuilder<'slf> {
        Self {
            tau,
            threads: 1,
            weighter: None,
            extractor: None,
        }
    }
}

impl<'slf> CoresetBuilder<'slf> {
    pub fn with_extractor(
        self,
        extractor: Box<dyn ExtractCoresetPoints + 'slf>,
    ) -> CoresetBuilder<'slf> {
        CoresetBuilder {
            tau: self.tau,
            threads: self.threads,
            weighter: None,
            extractor: Some(extractor),
        }
    }

    pub fn with_weighter(
        self,
        weighter: Box<dyn WeightCoresetPoints + 'slf>,
    ) -> CoresetBuilder<'slf> {
        CoresetBuilder {
            tau: self.tau,
            threads: self.threads,
            weighter: Some(weighter),
            extractor: self.extractor,
        }
    }

    pub fn with_threads(self, threads: usize) -> CoresetBuilder<'slf> {
        Self { threads, ..self }
    }

    pub fn fit<'data, C: MetricData + Send, D: MetricData + NChunks<Output<'data> = C>>(
        &self,
        data: &'data D,
    ) -> FittedCoreset {
        if self.threads == 1 {
            self.fit_sequential(data)
        } else {
            self.fit_parallel(data)
        }
    }

    fn fit_sequential<D: MetricData>(&self, data: &D) -> FittedCoreset {
        self.fit_sequential_offset(data, 0)
    }

    fn fit_sequential_offset<D: MetricData>(&self, data: &D, index_offset: usize) -> FittedCoreset {
        use crate::gmm::*;

        let (coreset_points, assignment, radius) = greedy_minimum_maximum(data, self.tau);

        // If we have an extractor, use it to get points for each coreset cluster
        let (coreset_indices, assignment, radius) = if let Some(extractor) = self.extractor.as_ref()
        {
            let mut extended_idxs = Vec::new();
            let mut assigned = Vec::new();
            for c in coreset_points.iter() {
                assigned.clear();
                for (i, a) in assignment.iter().enumerate() {
                    if a == c {
                        assigned.push(i);
                    }
                }
                let extracted = extractor.extract_coreset_points(*c, &assigned);
                extended_idxs.extend_from_slice(extracted.as_slice().unwrap());
            }
            let extended_idxs = Array1::from_vec(extended_idxs);
            let (assignment, radius) = assign_closest(data, &extended_idxs);
            (extended_idxs, assignment, radius)
        } else {
            (coreset_points, assignment, radius)
        };

        let weights = if let Some(weighter) = self.weighter.as_ref() {
            weighter.weight_coreset_points(coreset_indices.len(), &assignment)
        } else {
            weight_by_count(&assignment)
        };

        FittedCoreset {
            // coreset_points: data.subset(&coreset_indices),
            coreset_indices: coreset_indices + index_offset,
            radius,
            weights,
            assignment: assignment + index_offset,
        }
    }

    fn fit_parallel<'data, D: MetricData + NChunks<Output<'data> = C>, C: MetricData + Send>(
        &self,
        data: &'data D,
    ) -> FittedCoreset {
        let n_chunks = self.threads;
        let chunk_size: usize = data.nchunks_size(n_chunks);
        let chunks = data.nchunks(n_chunks);
        let mut out = vec![None; n_chunks];
        std::thread::scope(|scope| {
            let mut handles = Vec::new();
            for (i, (out, chunk)) in out.iter_mut().zip(chunks).enumerate() {
                let offset = i * chunk_size;
                let builder = self;
                let h = scope.spawn(move || {
                    out.replace(builder.fit_sequential_offset(&chunk, offset));
                });
                handles.push(h);
            }

            for h in handles {
                h.join().unwrap();
            }
        });

        // put together the solution
        let mut composed = out[0].clone().unwrap();
        for coreset in &out[1..] {
            composed = Compose::compose(composed, coreset.clone().unwrap());
        }

        composed
    }
}

pub trait NChunks {
    type Output<'slf>
    where
        Self: 'slf;

    /// Returns an iterator of `num_chunks` chunks of `self`.
    fn nchunks(&self, num_chunks: usize) -> impl Iterator<Item = Self::Output<'_>>;

    /// Returns the size of the chunks to get the desired number of chunks
    fn nchunks_size(&self, num_chunks: usize) -> usize;
}

impl NChunks for Array1<usize> {
    type Output<'slf> = ArrayView1<'slf, usize>;

    fn nchunks(&self, num_chunks: usize) -> impl Iterator<Item = Self::Output<'_>> {
        let size = self.nchunks_size(num_chunks);
        self.axis_chunks_iter(Axis(0), size)
    }

    fn nchunks_size(&self, num_chunks: usize) -> usize {
        (self.len() as f64 / num_chunks as f64).ceil() as usize
    }
}

impl NChunks for Array1<f32> {
    type Output<'slf> = ArrayView1<'slf, f32>;

    fn nchunks(&self, num_chunks: usize) -> impl Iterator<Item = Self::Output<'_>> {
        let size = self.nchunks_size(num_chunks);
        self.axis_chunks_iter(Axis(0), size)
    }

    fn nchunks_size(&self, num_chunks: usize) -> usize {
        (self.len() as f64 / num_chunks as f64).ceil() as usize
    }
}

impl<S: ndarray::Data<Elem = f32>> NChunks for ArrayBase<S, Ix2> {
    type Output<'slf> = ArrayView2<'slf, f32> where S: 'slf;

    fn nchunks(&self, num_chunks: usize) -> impl Iterator<Item = Self::Output<'_>> {
        let size = self.nchunks_size(num_chunks);
        self.axis_chunks_iter(Axis(0), size)
    }

    fn nchunks_size(&self, num_chunks: usize) -> usize {
        (self.nrows() as f64 / num_chunks as f64).ceil() as usize
    }
}

impl<S: ndarray::Data<Elem = usize>> NChunks for ArrayBase<S, IxDyn> {
    type Output<'slf> = ArrayViewD<'slf, usize> where S: 'slf;

    fn nchunks(&self, num_chunks: usize) -> impl Iterator<Item = Self::Output<'_>> {
        let size = self.nchunks_size(num_chunks);
        self.axis_chunks_iter(Axis(0), size)
    }

    fn nchunks_size(&self, num_chunks: usize) -> usize {
        (self.shape()[0] as f64 / num_chunks as f64).ceil() as usize
    }
}

impl<T> NChunks for &[T] {
    type Output<'slf> = &'slf [T] where Self: 'slf;

    fn nchunks(&self, num_chunks: usize) -> impl Iterator<Item = Self::Output<'_>> {
        let size = self.nchunks_size(num_chunks);
        self.chunks(size)
    }

    fn nchunks_size(&self, num_chunks: usize) -> usize {
        (self.len() as f64 / num_chunks as f64).ceil() as usize
    }
}

impl<C: NChunks> NChunks for Option<C> {
    type Output<'slf> = Option<C::Output<'slf>> where Self: 'slf;

    fn nchunks<'slf>(&'slf self, num_chunks: usize) -> impl Iterator<Item = Self::Output<'slf>> {
        match self {
            Some(data) => {
                let chunks = data
                    .nchunks(num_chunks)
                    .map(Some)
                    .collect::<Vec<Self::Output<'slf>>>();

                chunks.into_iter()
            }
            None => {
                let mut chunks = Vec::with_capacity(num_chunks);
                for _ in 0..num_chunks {
                    chunks.push(None)
                }
                chunks.into_iter()
            }
        }
    }

    fn nchunks_size(&self, num_chunks: usize) -> usize {
        match self {
            Some(data) => data.nchunks_size(num_chunks),
            None => 0,
        }
    }
}

impl NChunks for () {
    type Output<'slf> = ();

    fn nchunks(&self, num_chunks: usize) -> impl Iterator<Item = Self::Output<'_>> {
        vec![(); num_chunks].into_iter()
    }

    fn nchunks_size(&self, _num_chunks: usize) -> usize {
        0
    }
}

#[cfg(test)]
mod test {
    use crate::{metricdata::EuclideanData, test::make_blobs};

    use super::CoresetBuilder;

    /// check that the assignment indices are within the bounds
    #[test]
    fn test_assigned_range() {
        let data = make_blobs(3, 1000, 100, 1.0, 10.0);
        let data = EuclideanData::new(data);
        let tau = 1000;
        let coreset = CoresetBuilder::with_tau(tau).fit(&data);
        assert!(coreset.assignment().iter().all(|i| i < &tau));
    }
}
