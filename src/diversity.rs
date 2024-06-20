use crate::{coreset::Coreset, gmm::greedy_minimum_maximum};
use ndarray::{prelude::*, Data};

pub enum DiversityKind {
    /// Maximize the minimum distance
    RemoteEdge,
}

impl DiversityKind {
    fn solve<S: Data<Elem = f32>>(&self, data: &ArrayBase<S, Ix2>, k: usize) -> Array1<usize> {
        match self {
            Self::RemoteEdge => {
                let (sol, _, _) = greedy_minimum_maximum(data, k);
                sol
            }
        }
    }
}

pub struct DiversityMaximization {
    k: usize,
    kind: DiversityKind,
    solution: Option<Array1<usize>>,
    coreset_size: Option<usize>,
}

impl DiversityMaximization {
    pub fn new(k: usize, kind: DiversityKind) -> Self {
        Self {
            k,
            kind,
            solution: None,
            coreset_size: None,
        }
    }

    pub fn with_coreset(self, coreset_size: usize) -> Self {
        assert!(coreset_size > self.k);
        Self {
            k: self.k,
            kind: self.kind,
            solution: self.solution,
            coreset_size: Some(coreset_size),
        }
    }

    pub fn fit<S: Data<Elem = f32>>(&mut self, data: &ArrayBase<S, Ix2>) {
        if let Some(coreset_size) = self.coreset_size {
            let mut coreset = Coreset::new(coreset_size);
            // TODO: actually use ancillary data, if present
            coreset.fit_predict(data, ());
            let data = coreset.coreset_points().unwrap();
            self.solution.replace(self.kind.solve(&data, self.k));
        } else {
            self.solution.replace(self.kind.solve(data, self.k));
        }
    }
}
