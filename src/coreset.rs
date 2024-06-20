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

pub struct Coreset<A>
where
    A: AncillaryData,
{
    /// the size of the coreset, i.e. the number of proxy
    /// points to be selected
    tau: usize,
    /// the function to compute the weight of each proxy point
    coreset_fit: Option<CoresetFit<A::Weights>>,
}

impl<A> Coreset<A>
where
    A: AncillaryData,
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
    A: AncillaryData,
    A::Weights: Compose,
{
    fn compose(a: Self, b: Self) -> Self {
        let coreset_fit = match (a.coreset_fit, b.coreset_fit) {
            (Some(afit), Some(bfit)) => Some(Compose::compose(afit, bfit)),
            (None, Some(bfit)) => Some(bfit),
            (Some(afit), None) => Some(afit),
            (None, None) => None,
        };
        Self {
            tau: a.tau + b.tau,
            coreset_fit,
        }
    }
}
