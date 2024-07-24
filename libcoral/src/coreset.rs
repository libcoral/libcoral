use ndarray::{concatenate, prelude::*, Data};

pub trait WeightCoresetPoints {
    fn weight_coreset_points<A>(
        &self,
        n_coreset_points: usize,
        assignment: &Array1<usize>,
        ancillary: Option<&[A]>,
    ) -> ArrayD<usize>;
}
impl WeightCoresetPoints for () {
    fn weight_coreset_points<A>(
        &self,
        _n_coreset_points: usize,
        _assignment: &Array1<usize>,
        _ancillary: Option<&[A]>,
    ) -> ArrayD<usize> {
        unreachable!("just to appease the type checker!")
    }
}

#[derive(Clone, Copy)]
pub struct WeightByCount;
impl WeightCoresetPoints for WeightByCount {
    fn weight_coreset_points<A>(
        &self,
        n_coreset_points: usize,
        assignment: &Array1<usize>,
        _ancillary: Option<&[A]>,
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
pub trait ExtractCoresetPoints {
    /// Returns an array of indices into the data pointing to the extracted coreset points.
    fn extract_coreset_points<S: Data<Elem = f32>, A>(
        &self,
        data: &ArrayBase<S, Ix2>,
        ancillary: Option<&[A]>,
        assigned: &[usize],
    ) -> Array1<usize>;
}
impl ExtractCoresetPoints for () {
    fn extract_coreset_points<S: Data<Elem = f32>, A>(
        &self,
        _data: &ArrayBase<S, Ix2>,
        _ancillary: Option<&[A]>,
        _assigned: &[usize],
    ) -> Array1<usize> {
        unreachable!("just to appease the type checker")
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
pub struct FittedCoreset<A>
where
    A: Clone,
{
    coreset_points: Array2<f32>,
    ancillary: Option<Vec<A>>,
    radius: Array1<f32>,
    weights: ArrayD<usize>,
    assignment: Array1<usize>,
}

impl<A: Clone> FittedCoreset<A> {
    pub fn points(&self) -> ArrayView2<f32> {
        self.coreset_points.view()
    }

    pub fn radii(&self) -> ArrayView1<f32> {
        self.radius.view()
    }

    pub fn weights(&self) -> ArrayViewD<usize> {
        self.weights.view()
    }
}

impl<A: Clone> Compose for FittedCoreset<A> {
    fn compose(a: Self, b: Self) -> Self {
        let ancillary = a
            .ancillary
            .zip(b.ancillary)
            .map(|(a, b)| Compose::compose(a, b));
        Self {
            coreset_points: Compose::compose(a.coreset_points, b.coreset_points),
            ancillary,
            radius: Compose::compose(a.radius, b.radius),
            weights: Compose::compose(a.weights, b.weights),
            assignment: Compose::compose(a.assignment, b.assignment),
        }
    }
}

#[derive(Clone)]
pub struct CoresetBuilder<E: ExtractCoresetPoints + Clone, W: WeightCoresetPoints + Clone> {
    tau: usize,
    threads: usize,
    index_offset: usize,
    weighter: Option<W>,
    extractor: Option<E>,
}

impl CoresetBuilder<(), ()> {
    pub fn with_tau(tau: usize) -> CoresetBuilder<(), ()> {
        Self {
            tau,
            threads: 1,
            index_offset: 0,
            weighter: None,
            extractor: None,
        }
    }
}

impl<
        E: ExtractCoresetPoints + Send + Sync + Clone,
        W: WeightCoresetPoints + Send + Sync + Clone,
    > CoresetBuilder<E, W>
{
    pub fn with_extractor<E2: ExtractCoresetPoints + Clone>(
        self,
        extractor: E2,
    ) -> CoresetBuilder<E2, W> {
        CoresetBuilder {
            tau: self.tau,
            threads: self.threads,
            index_offset: self.index_offset,
            weighter: None,
            extractor: Some(extractor),
        }
    }

    pub fn with_weighter<W2: WeightCoresetPoints + Clone>(
        self,
        weighter: W2,
    ) -> CoresetBuilder<E, W2> {
        CoresetBuilder {
            tau: self.tau,
            threads: self.threads,
            index_offset: self.index_offset,
            weighter: Some(weighter),
            extractor: self.extractor,
        }
    }

    pub fn with_index_offset(self, index_offset: usize) -> CoresetBuilder<E, W> {
        Self {
            index_offset,
            ..self
        }
    }

    pub fn with_threads(self, threads: usize) -> CoresetBuilder<E, W> {
        Self { threads, ..self }
    }

    pub fn fit<S: Data<Elem = f32>, A: Send + Sync + Clone>(
        &self,
        data: &ArrayBase<S, Ix2>,
        ancillary: Option<&[A]>,
    ) -> FittedCoreset<A> {
        if self.threads == 1 {
            self.fit_sequential(data, ancillary)
        } else {
            self.fit_parallel(data, ancillary)
        }
    }

    fn fit_sequential<S: Data<Elem = f32>, A: Clone>(
        &self,
        data: &ArrayBase<S, Ix2>,
        ancillary: Option<&[A]>,
    ) -> FittedCoreset<A> {
        use crate::gmm::*;

        let (coreset_points, assignment, radius) = greedy_minimum_maximum(data, self.tau);

        // If we have an extractor, use it to get points for each coreset cluster
        let (coreset_points, assignment, radius) = if let Some(extractor) = self.extractor.as_ref()
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
                let extracted = extractor.extract_coreset_points(data, ancillary, &assigned);
                extended_idxs.extend_from_slice(extracted.as_slice().unwrap());
            }
            let extended_idxs = Array1::from_vec(extended_idxs);
            let (assignment, radius) = assign_closest(data, &extended_idxs);
            (extended_idxs, assignment, radius)
        } else {
            (coreset_points, assignment, radius)
        };

        let weights = if let Some(weighter) = self.weighter.as_ref() {
            weighter.weight_coreset_points(coreset_points.len(), &assignment, ancillary)
        } else {
            weight_by_count(&assignment)
        };

        let coreset_ancillary = ancillary.map(|ancillary| {
            let mut out = Vec::with_capacity(coreset_points.len());
            for i in coreset_points.iter() {
                out.push(ancillary[*i].clone());
            }
            out
        });

        FittedCoreset {
            coreset_points: data.select(Axis(0), coreset_points.as_slice().unwrap()),
            ancillary: coreset_ancillary,
            radius,
            weights,
            assignment: assignment + self.index_offset,
        }
    }

    fn fit_parallel<S: Data<Elem = f32>, A: Send + Sync + Clone>(
        &self,
        data: &ArrayBase<S, Ix2>,
        ancillary: Option<&[A]>,
    ) -> FittedCoreset<A> {
        let n_chunks = self.threads;
        let chunk_size: usize = data.nchunks_size(n_chunks);
        let chunks = data.nchunks(n_chunks);
        let ancillary_chunks = ancillary.nchunks(n_chunks);
        let mut out = vec![None; n_chunks];
        std::thread::scope(|scope| {
            let mut handles = Vec::new();
            for (i, (out, (chunk, ancillary_chunk))) in
                out.iter_mut().zip(chunks.zip(ancillary_chunks)).enumerate()
            {
                let offset = i * chunk_size;
                let builder = self.clone().with_index_offset(offset);
                let h = scope.spawn(move || {
                    out.replace(builder.fit_sequential(&chunk, ancillary_chunk));
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
