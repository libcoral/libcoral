use libcoral::{
    coreset::{CoresetBuilder, FittedCoreset},
    diversity::{DiversityKind, DiversityMaximization},
    matroid::{PartitionMatroid, TransversalMatroid},
};
use numpy::*;
use pyo3::prelude::*;

#[pyclass]
#[pyo3(name = "Coreset")]
/// Build a coreset out of the given data points. Loosely follows
/// the scikit-learn interface.
///
/// ## References
///
/// - Matteo Ceccarello, Andrea Pietracaprina, Geppino Pucci:
///   Solving k-center Clustering (with Outliers) in MapReduce and Streaming, almost as Accurately as Sequentially.
///   Proc. VLDB Endow. 12(7): 766-778 (2019)
pub struct PyCoreset {
    builder: CoresetBuilder<(), ()>,
    fitted: Option<FittedCoreset<()>>,
}

#[pymethods]
impl PyCoreset {
    #[new]
    #[pyo3(signature = (size, num_threads=1))]
    /// Set up the coreset. If `num_threads > 1`, the coreset will be built
    /// in parallel using multiple threads.
    fn new(size: usize, num_threads: usize) -> Self {
        let builder = CoresetBuilder::with_tau(size).with_threads(num_threads);
        Self {
            builder,
            fitted: None,
        }
    }

    /// Fit the coreset to the given data, i.e. selects the coreset points,
    /// and their respective radii and weights, out of the data.
    fn fit(&mut self, data: PyReadonlyArray2<f32>) {
        let data = data.as_array();
        self.fitted.replace(self.builder.fit(&data, None));
    }

    /// Fit the coreset to the given data, i.e. selects the coreset points,
    /// and their respective radii and weights, out of the data.
    /// Furthermore, return a vector with the assignment of data points
    /// to coreset points.
    fn fit_transform<'py>(
        mut slf: PyRefMut<'py, Self>,
        data: PyReadonlyArray2<'py, f32>,
    ) -> Bound<'py, PyArray1<usize>> {
        slf.fit(data);
        let assignment = slf.fitted.as_ref().unwrap().assignment();
        assignment.to_pyarray_bound(slf.py())
    }

    #[getter]
    /// On a fitted coreset, return the actual coreset points
    fn points_(slf: PyRef<Self>) -> Option<Bound<PyArray2<f32>>> {
        slf.fitted
            .as_ref()
            .map(|coreset| coreset.points().to_pyarray_bound(slf.py()))
    }

    #[getter]
    /// On a fitted coreset, return the radius of each fitted point
    fn radii_(slf: PyRef<Self>) -> Option<Bound<PyArray1<f32>>> {
        slf.fitted
            .as_ref()
            .map(|coreset| coreset.radii().to_pyarray_bound(slf.py()))
    }

    #[getter]
    /// On a fitted coreset, return the weight of each fitted point
    fn weights_(slf: PyRef<Self>) -> Option<Bound<PyArrayDyn<usize>>> {
        slf.fitted
            .as_ref()
            .map(|coreset| coreset.weights().to_pyarray_bound(slf.py()))
    }
}

#[derive(FromPyObject, Clone)]
enum MatroidDescriptionContent {
    Partition(Vec<usize>),
    Transversal(usize),
}

#[derive(Clone)]
#[pyclass]
pub struct MatroidDescription {
    description: MatroidDescriptionContent,
}

#[pymethods]
impl MatroidDescription {
    #[new]
    fn new(description: MatroidDescriptionContent) -> Self {
        Self { description }
    }
}

#[pyclass]
#[pyo3(name = "DiversityMaximization")]
pub struct PyDiversityMaximization {
    k: usize,
    kind: DiversityKind,
    coreset_size: Option<usize>,
    threads: usize,
    epsilon: f32,
    matroid_description: Option<MatroidDescription>,
}

#[pymethods]
impl PyDiversityMaximization {
    #[new]
    #[pyo3(signature = (k, kind, coreset_size=None, num_threads=1, epsilon=0.01, matroid=None))]
    fn new(
        k: usize,
        kind: &str,
        coreset_size: Option<usize>,
        num_threads: usize,
        epsilon: f32,
        matroid: Option<MatroidDescription>,
    ) -> Self {
        let kind = match kind {
            "remote-edge" | "edge" => DiversityKind::RemoteEdge,
            "remote-clique" | "clique" => DiversityKind::RemoteClique,
            _ => panic!("Wrong kind"),
        };
        Self {
            k,
            kind,
            coreset_size,
            threads: num_threads,
            epsilon,
            matroid_description: matroid,
        }
    }

    fn solve<'py>(
        self_: PyRef<'py, Self>,
        data: PyReadonlyArray2<'py, f32>,
        ancillary: Option<Bound<'py, PyAny>>,
    ) -> Bound<'py, PyArray1<usize>> {
        let sol = if let Some(matroid) = self_.matroid_description.as_ref() {
            match &matroid.description {
                MatroidDescriptionContent::Partition(v) => {
                    let matroid = PartitionMatroid::new(v.clone());
                    let data = data.as_array();
                    let ancillary = ancillary
                        .as_ref()
                        .map(|anc| anc.extract::<Vec<usize>>().unwrap())
                        .unwrap();
                    let ancillary = ancillary.as_slice();
                    let diversity = DiversityMaximization::new(self_.k, self_.kind)
                        .with_epsilon(self_.epsilon)
                        .with_threads(self_.threads)
                        .with_matroid(matroid);
                    let diversity = if let Some(coreset_size) = self_.coreset_size {
                        diversity.with_coreset(coreset_size)
                    } else {
                        diversity
                    };
                    diversity.solve(&data, Some(ancillary))
                }
                MatroidDescriptionContent::Transversal(topics) => {
                    let matroid = TransversalMatroid::new(*topics);
                    let data = data.as_array();
                    let ancillary = ancillary
                        .as_ref()
                        .map(|anc| anc.extract::<Vec<Vec<usize>>>().unwrap())
                        .unwrap();
                    let ancillary = ancillary.as_slice();
                    let diversity = DiversityMaximization::new(self_.k, self_.kind)
                        .with_epsilon(self_.epsilon)
                        .with_threads(self_.threads)
                        .with_matroid(matroid);
                    let diversity = if let Some(coreset_size) = self_.coreset_size {
                        diversity.with_coreset(coreset_size)
                    } else {
                        diversity
                    };
                    diversity.solve(&data, Some(ancillary))
                }
            }
        } else {
            let data = data.as_array();
            let diversity =
                DiversityMaximization::new(self_.k, self_.kind).with_threads(self_.threads);
            let diversity = if let Some(coreset_size) = self_.coreset_size {
                diversity.with_coreset(coreset_size)
            } else {
                diversity
            };
            diversity.solve(&data, None)
        };
        sol.to_pyarray_bound(self_.py())
    }
}

#[pymodule]
#[pyo3(name = "libcoral")]
fn py_libcoral(m: &Bound<'_, PyModule>) -> PyResult<()> {
    pyo3_log::init();
    m.add_class::<PyCoreset>()?;
    m.add_class::<PyDiversityMaximization>()?;
    m.add_class::<MatroidDescription>()?;
    Ok(())
}
