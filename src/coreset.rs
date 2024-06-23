use ndarray::{concatenate, prelude::*, Data};

pub trait AncillaryData: Sized {
    type Weights: Clone;

    fn compute_weights(&self, assignment: &Array1<usize>) -> Self::Weights;
}

impl AncillaryData for () {
    type Weights = Array1<usize>;

    fn compute_weights(&self, assignment: &Array1<usize>) -> Self::Weights {
        // assign weight 1 to all coreset points if there is no ancillary data
        Array1::ones(assignment.iter().max().unwrap() + 1)
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

#[derive(Clone)]
struct CoresetFit<W> {
    coreset_points: Array2<f32>,
    radius: Array1<f32>,
    weights: W,
}

impl<W: Compose> Compose for CoresetFit<W> {
    fn compose(a: Self, b: Self) -> Self {
        Self {
            coreset_points: Compose::compose(a.coreset_points, b.coreset_points),
            radius: Compose::compose(a.radius, b.radius),
            weights: Compose::compose(a.weights, b.weights),
        }
    }
}

#[derive(Clone)]
pub struct Coreset<A>
where
    A: AncillaryData + Clone,
    A::Weights: Clone,
{
    /// the size of the coreset, i.e. the number of proxy
    /// points to be selected
    tau: usize,
    /// the function to compute the weight of each proxy point
    coreset_fit: Option<CoresetFit<A::Weights>>,
}

impl<A> Coreset<A>
where
    A: AncillaryData + Clone,
    A::Weights: Clone,
{
    pub fn new(tau: usize) -> Self {
        Self {
            tau,
            coreset_fit: None,
        }
    }

    /// Compute the coreset points and their weights. Return the array with the assignment of input
    /// data points to the closest coreset point, i.e. the proxy function.
    pub fn fit_predict<S: Data<Elem = f32>>(
        &mut self,
        data: &ArrayBase<S, Ix2>,
        ancillary: A,
    ) -> Array1<usize> {
        use crate::gmm::*;

        let (coreset_points, assignment, radius) = greedy_minimum_maximum(data, self.tau);
        let weights = ancillary.compute_weights(&assignment);
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

    pub fn coreset_weights(&self) -> Option<A::Weights> {
        // FIXME: remove this clone
        self.coreset_fit.as_ref().map(|cf| cf.weights.clone())
    }
}

impl<A> Compose for Coreset<A>
where
    A: AncillaryData + Clone,
    A::Weights: Compose + Clone,
{
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
        let size = (self.len() as f64 / num_chunks as f64).ceil() as usize;
        self.axis_chunks_iter(Axis(0), size)
    }
}

impl NChunks for () {
    type Output<'slf> = ();

    fn nchunks(&self, num_chunks: usize) -> impl Iterator<Item = Self::Output<'_>> {
        vec![(); num_chunks].into_iter()
    }
}

pub struct ParallelCoreset<A>
where
    A: AncillaryData + Clone,
    A::Weights: Compose + Clone,
{
    coresets: Vec<Coreset<A>>,
    composed_fit: Option<CoresetFit<A::Weights>>,
}

impl<A> ParallelCoreset<A>
where
    A: AncillaryData + NChunks + Send + Clone,
    A::Weights: Compose + Send + Clone,
{
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

    pub fn fit<S: Data<Elem = f32>, P>(&mut self, data: &ArrayBase<S, Ix2>, ancillary: P)
    where
        for<'a> P: NChunks<Output<'a> = A> + 'static,
    {
        let coresets = &mut self.coresets;
        let n_chunks = coresets.len();
        let chunks = data.nchunks(n_chunks);
        let ancillary_chunks = ancillary.nchunks(n_chunks);
        std::thread::scope(|scope| {
            for ((coreset, chunk), ancillary_chunk) in
                coresets.iter_mut().zip(chunks).zip(ancillary_chunks)
            {
                scope.spawn(move || {
                    coreset.fit_predict(&chunk, ancillary_chunk);
                });
            }
        });

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

    pub fn coreset_weights(&self) -> Option<A::Weights> {
        // FIXME: remove this clone
        self.composed_fit.as_ref().map(|cf| cf.weights.clone())
    }
}
