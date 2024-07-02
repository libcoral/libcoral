use ndarray::{concatenate, prelude::*, Data};
use std::sync::Arc;

trait FnWeight<A>: Fn(&Array1<usize>, &[A]) -> ArrayD<usize> + Send + Sync {}

pub struct AncillaryInfo<'data, A> {
    ancillary: &'data [A],
    compute_weights_fn: Arc<dyn FnWeight<A>>,
}

impl<'data, A> AncillaryInfo<'data, A> {
    fn compute_weights(&self, assignment: &Array1<usize>) -> ArrayD<usize> {
        assert_eq!(assignment.len(), self.ancillary.len());
        (self.compute_weights_fn)(assignment, self.ancillary)
    }
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
    pub fn fit_predict<'data, S: Data<Elem = f32>, A>(
        &mut self,
        data: &ArrayBase<S, Ix2>,
        ancillary: Option<AncillaryInfo<'data, A>>,
    ) -> Array1<usize> {
        use crate::gmm::*;

        let (coreset_points, assignment, radius) = greedy_minimum_maximum(data, self.tau);
        let weights = if let Some(ancillary) = ancillary {
            ancillary.compute_weights(&assignment)
        } else {
            ArrayD::ones(IxDyn(&[coreset_points.len()]))
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
}

impl NChunks for Array1<usize> {
    type Output<'slf> = ArrayView1<'slf, usize>;

    fn nchunks(&self, num_chunks: usize) -> impl Iterator<Item = Self::Output<'_>> {
        let size = (self.len() as f64 / num_chunks as f64).ceil() as usize;
        self.axis_chunks_iter(Axis(0), size)
    }
}

impl<S: ndarray::Data<Elem = f32>> NChunks for ArrayBase<S, Ix2> {
    type Output<'slf> = ArrayView2<'slf, f32> where S: 'slf;

    fn nchunks(&self, num_chunks: usize) -> impl Iterator<Item = Self::Output<'_>> {
        let size = (self.nrows() as f64 / num_chunks as f64).ceil() as usize;
        self.axis_chunks_iter(Axis(0), size)
    }
}

impl<S: ndarray::Data<Elem = usize>> NChunks for ArrayBase<S, IxDyn> {
    type Output<'slf> = ArrayViewD<'slf, usize> where S: 'slf;

    fn nchunks(&self, num_chunks: usize) -> impl Iterator<Item = Self::Output<'_>> {
        let size = (self.shape()[0] as f64 / num_chunks as f64).ceil() as usize;
        self.axis_chunks_iter(Axis(0), size)
    }
}

impl<T> NChunks for &[T] {
    type Output<'slf> = &'slf [T] where Self: 'slf;

    fn nchunks(&self, num_chunks: usize) -> impl Iterator<Item = Self::Output<'_>> {
        let size = (self.len() as f64 / num_chunks as f64).ceil() as usize;
        self.chunks(size)
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
}

impl<C: NChunks> NChunks for Option<C> {
    type Output<'slf> = Option<C::Output<'slf>> where Self: 'slf;

    fn nchunks(&self, num_chunks: usize) -> impl Iterator<Item = Self::Output<'_>> {
        match self {
            Some(data) => data.nchunks(num_chunks).map(|c| Some(c)),
            // None => vec![None; num_chunks].into_iter(),
            None => panic!(),
        }
    }
}

impl NChunks for () {
    type Output<'slf> = ();

    fn nchunks(&self, num_chunks: usize) -> impl Iterator<Item = Self::Output<'_>> {
        vec![(); num_chunks].into_iter()
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
