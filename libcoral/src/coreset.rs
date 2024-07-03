use ndarray::{concatenate, prelude::*, Data};
use std::sync::Arc;

trait FnWeight<A>: Fn(&Array1<usize>, &[A]) -> ArrayD<usize> + Send + Sync {}

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
/// indices into the dataset vectors as well as the (optional) ancillary data
pub trait ExtractCoresetPoints {
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

pub struct AncillaryInfo<'data, A> {
    pub ancillary: &'data [A],
    compute_weights_fn: Arc<dyn FnWeight<A>>,
}

impl<'data, A> AncillaryInfo<'data, A> {
    fn compute_weights(&self, assignment: &Array1<usize>) -> ArrayD<usize> {
        assert_eq!(assignment.len(), self.ancillary.len());
        (self.compute_weights_fn)(assignment, self.ancillary)
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
    coreset_points: Array2<f32>,
    radius: Array1<f32>,
    weights: ArrayD<usize>,
    assignment: Array1<usize>,
}

impl FittedCoreset {
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

impl Compose for FittedCoreset {
    fn compose(a: Self, b: Self) -> Self {
        Self {
            coreset_points: Compose::compose(a.coreset_points, b.coreset_points),
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

    pub fn fit<S: Data<Elem = f32>, A: Send + Sync>(
        &self,
        data: &ArrayBase<S, Ix2>,
        ancillary: Option<&[A]>,
    ) -> FittedCoreset {
        if self.threads == 1 {
            self.fit_sequential(data, ancillary)
        } else {
            self.fit_parallel(data, ancillary)
        }
    }

    fn fit_sequential<S: Data<Elem = f32>, A>(
        &self,
        data: &ArrayBase<S, Ix2>,
        ancillary: Option<&[A]>,
    ) -> FittedCoreset {
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
        FittedCoreset {
            coreset_points: data.select(Axis(0), coreset_points.as_slice().unwrap()),
            radius,
            weights,
            assignment: assignment + self.index_offset,
        }
    }

    fn fit_parallel<S: Data<Elem = f32>, A: Send + Sync>(
        &self,
        data: &ArrayBase<S, Ix2>,
        ancillary: Option<&[A]>,
    ) -> FittedCoreset {
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

#[derive(Clone)]
struct CoresetFit {
    coreset_points: Array2<f32>,
    radius: Array1<f32>,
    weights: ArrayD<usize>,
}

impl Compose for CoresetFit {
    fn compose(a: Self, b: Self) -> Self {
        Self {
            coreset_points: Compose::compose(a.coreset_points, b.coreset_points),
            radius: Compose::compose(a.radius, b.radius),
            weights: Compose::compose(a.weights, b.weights),
        }
    }
}

#[derive(Clone)]
pub struct Coreset {
    /// the size of the coreset, i.e. the number of proxy
    /// points to be selected
    tau: usize,
    /// the function to compute the weight of each proxy point
    coreset_fit: Option<CoresetFit>,
}

impl Coreset {
    pub fn new(tau: usize) -> Self {
        Self {
            tau,
            coreset_fit: None,
        }
    }

    /// Compute the coreset points and their weights. Return the array with the assignment of input
    /// data points to the closest coreset point, i.e. the proxy function.
    pub fn fit_predict<S: Data<Elem = f32>, A>(
        &mut self,
        data: &ArrayBase<S, Ix2>,
        ancillary: Option<AncillaryInfo<A>>,
    ) -> Array1<usize> {
        use crate::gmm::*;

        let (coreset_points, assignment, radius) = greedy_minimum_maximum(data, self.tau);

        // TODO: Here we need to put a call to a function to extend the
        // coreset points selection

        let weights = if let Some(ancillary) = ancillary {
            ancillary.compute_weights(&assignment)
        } else {
            weight_by_count(&assignment)
        };
        self.coreset_fit.replace(CoresetFit {
            coreset_points: data.select(Axis(0), coreset_points.as_slice().unwrap()),
            radius,
            weights,
        });

        assignment
    }

    pub fn coreset_points(&self) -> Option<ArrayView2<f32>> {
        self.coreset_fit.as_ref().map(|cf| cf.coreset_points.view())
    }

    pub fn coreset_radii(&self) -> Option<ArrayView1<f32>> {
        self.coreset_fit.as_ref().map(|cf| cf.radius.view())
    }

    pub fn coreset_weights(&self) -> Option<ArrayViewD<usize>> {
        self.coreset_fit.as_ref().map(|cf| cf.weights.view())
    }
}

impl Compose for Coreset {
    fn compose(a: Self, b: Self) -> Self {
        assert!(a.coreset_fit.is_some());
        assert!(b.coreset_fit.is_some());
        let coreset_fit = Some(Compose::compose(
            a.coreset_fit.unwrap(),
            b.coreset_fit.unwrap(),
        ));
        Self {
            tau: a.tau + b.tau,
            coreset_fit,
        }
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

impl<'data, A> NChunks for AncillaryInfo<'data, A> {
    type Output<'slf> = AncillaryInfo<'slf, A> where Self: 'slf;

    fn nchunks(&self, num_chunks: usize) -> impl Iterator<Item = Self::Output<'_>> {
        let func = self.compute_weights_fn.clone();
        let data_chunks = self.ancillary.nchunks(num_chunks);
        data_chunks.map(move |c| AncillaryInfo {
            ancillary: c,
            compute_weights_fn: func.clone(),
        })
    }

    fn nchunks_size(&self, _num_chunks: usize) -> usize {
        unimplemented!()
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

pub struct ParallelCoreset {
    coresets: Vec<Coreset>,
    composed_fit: Option<CoresetFit>,
}

impl ParallelCoreset {
    pub fn new(tau: usize, threads: usize) -> Self {
        let mut coresets = Vec::with_capacity(threads);
        for _ in 0..threads {
            coresets.push(Coreset::new(tau));
        }
        Self {
            coresets,
            composed_fit: None,
        }
    }

    pub fn fit<S: Data<Elem = f32>, A: Send + Sync>(
        &mut self,
        data: &ArrayBase<S, Ix2>,
        ancillary: Option<AncillaryInfo<A>>,
    ) {
        let coresets = &mut self.coresets;
        let n_chunks = coresets.len();
        let chunks = data.nchunks(n_chunks);
        let ancillary_chunks = ancillary.nchunks(n_chunks);
        std::thread::scope(|scope| {
            let mut handles = Vec::new();
            for ((coreset, chunk), ancillary_chunk) in
                coresets.iter_mut().zip(chunks).zip(ancillary_chunks)
            {
                let h = scope.spawn(move || {
                    coreset.fit_predict(&chunk, ancillary_chunk);
                });
                handles.push(h);
            }

            for h in handles {
                h.join().unwrap();
            }
        });

        eprintln!("Putting the solution together");
        // put together the solution
        let mut composed = coresets[0].clone();
        for coreset in &coresets[1..] {
            composed = Compose::compose(composed, coreset.clone());
        }

        self.composed_fit.replace(composed.coreset_fit.unwrap());
    }

    pub fn coreset_points(&self) -> Option<ArrayView2<f32>> {
        self.composed_fit
            .as_ref()
            .map(|cf| cf.coreset_points.view())
    }

    pub fn coreset_radii(&self) -> Option<ArrayView1<f32>> {
        self.composed_fit.as_ref().map(|cf| cf.radius.view())
    }

    pub fn coreset_weights(&self) -> Option<ArrayViewD<usize>> {
        self.composed_fit.as_ref().map(|cf| cf.weights.view())
    }
}
