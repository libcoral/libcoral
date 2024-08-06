use libcoral::coreset::{CoresetBuilder, FittedCoreset};
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

// #[pyclass]
// #[pyo3(name = "DiversityMaximization")]
// pub struct PyDiversityMaximization {
//     inner: DiversityMaximization,
// }
//
// #[pymethods]
// impl PyDiversityMaximization {
//     #[new]
//     #[pyo3(signature = (k, kind, coreset_size=None, num_threads=None))]
//     fn new(k: usize, kind: &str, coreset_size: Option<usize>, num_threads: Option<usize>) -> Self {
//         let kind = match kind {
//             "remote-edge" | "edge" => DiversityKind::RemoteEdge,
//             "remote-clique" | "clique" => DiversityKind::RemoteClique,
//             _ => panic!("Wrong kind"),
//         };
//         let mut inner = DiversityMaximization::new(k, kind);
//         if let Some(num_threads) = num_threads {
//             let coreset_size = coreset_size.unwrap_or(1000);
//             inner = inner.with_coreset(coreset_size).with_threads(num_threads)
//         } else if let Some(coreset_size) = coreset_size {
//             inner = inner.with_coreset(coreset_size);
//         }
//         Self { inner }
//     }
//
//     fn cost<'py>(mut self_: PyRefMut<'py, Self>, data: PyReadonlyArray2<'py, f32>) -> f32 {
//         self_.inner.cost(&data.as_array())
//     }
//
//     fn fit<'py>(mut self_: PyRefMut<'py, Self>, data: PyReadonlyArray2<'py, f32>) {
//         self_.inner.fit(&data.as_array());
//     }
//
//     fn solution_indices(self_: PyRef<Self>) -> PyResult<Bound<PyArray1<usize>>> {
//         let py = self_.py();
//         self_
//             .inner
//             .get_solution_indices()
//             .ok_or(PyValueError::new_err("model not trained"))
//             .map(|sol| sol.into_pyarray_bound(py))
//     }
// }

#[pymodule]
#[pyo3(name = "libcoral")]
fn py_libcoral(m: &Bound<'_, PyModule>) -> PyResult<()> {
    pyo3_log::init();
    m.add_class::<PyCoreset>()?;
    // m.add_class::<PyDiversityMaximization>()?;
    Ok(())
}
