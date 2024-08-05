use libcoral::diversity::{DiversityKind, DiversityMaximization};
use numpy::*;
use pyo3::{exceptions::PyValueError, prelude::*};

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
    // m.add_class::<PyDiversityMaximization>()?;
    Ok(())
}
