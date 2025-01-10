pub mod euclideandata;
pub mod angulardata;

pub trait MetricData {
    fn distance(&self, i: usize, j: usize) -> f32;
    fn all_distances(&self, j: usize, out: &mut [f32]);
    fn num_points(&self) -> usize;
    fn dimensions(&self) -> usize;
}

pub trait Subset {
    type Out: MetricData;
    fn subset<I: IntoIterator<Item = usize>>(&self, indices: I) -> Self::Out;
}

pub use self::euclideandata::EuclideanData;
pub use self::angulardata::AngularData;
