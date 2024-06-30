use libcoral::coreset::{Coreset, ParallelCoreset};
use numpy::*;
use pyo3::prelude::*;

#[pyclass]
#[pyo3(name = "Coreset")]
pub struct PyCoreset {
    inner: ParallelCoreset<()>,
}

#[pymethods]
impl PyCoreset {
    #[new]
    fn new(coreset_size: usize, num_threads: usize) -> Self {
        Self {
            inner: ParallelCoreset::new(coreset_size, num_threads),
        }
    }

    fn fit<'py>(mut self_: PyRefMut<'py, Self>, data: PyReadonlyArray2<'py, f32>) {
        self_.inner.fit(&data.as_array(), ());
    }

    /// Get information about the fitted coreset.
    ///
    /// Returns a triplet with the following information:
    ///
    ///  - The actual coreset points (2 dimensional float32 array)
    ///  - The weight of each coreset point (1 dimensional integer array)
    ///  - The radius of each coreeset cluster (1 dimensional float32 array)
    fn get_fit(
        self_: PyRef<'_, Self>,
    ) -> (
        Bound<'_, PyArray2<f32>>,
        Bound<'_, PyArray1<usize>>,
        Bound<'_, PyArray1<f32>>,
    ) {
        let centers = self_.inner.coreset_points().unwrap();
        let weights = self_.inner.coreset_weights().unwrap();
        let radii = self_.inner.coreset_radii().unwrap();
        let py = self_.py();
        (
            centers.to_pyarray_bound(py),
            weights.to_pyarray_bound(py),
            radii.to_pyarray_bound(py),
        )
    }
}

#[pymodule]
#[pyo3(name = "libcoral")]
fn py_libcoral(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyCoreset>()?;
    Ok(())
}
