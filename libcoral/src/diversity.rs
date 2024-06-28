use crate::{
    coreset::{Coreset, ParallelCoreset},
    gmm::greedy_minimum_maximum,
};
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
    threads: usize,
}

impl DiversityMaximization {
    pub fn new(k: usize, kind: DiversityKind) -> Self {
        Self {
            k,
            kind,
            solution: None,
            coreset_size: None,
            threads: 1,
        }
    }

    pub fn with_coreset(self, coreset_size: usize) -> Self {
        assert!(coreset_size > self.k);
        Self {
            coreset_size: Some(coreset_size),
            ..self
        }
    }

    pub fn with_threads(self, threads: usize) -> Self {
        assert!(threads >= 1);
        Self { threads, ..self }
    }

    pub fn fit<S: Data<Elem = f32>>(&mut self, data: &ArrayBase<S, Ix2>) {
        match (self.threads, self.coreset_size) {
            (1, None) => {
                self.solution.replace(self.kind.solve(data, self.k));
            }
            (1, Some(coreset_size)) => {
                let mut coreset = Coreset::new(coreset_size);
                // TODO: actually use ancillary data, if present
                coreset.fit_predict(data, ());
                let data = coreset.coreset_points().unwrap();
                self.solution.replace(self.kind.solve(&data, self.k));
            }
            (threads, Some(coreset_size)) => {
                let mut coreset = ParallelCoreset::new(coreset_size, threads);
                // TODO: actually use ancillary data, if present
                coreset.fit(data, ());
                let data = coreset.coreset_points().unwrap();
                self.solution.replace(self.kind.solve(&data, self.k));
            }
            _ => panic!("you should specify a coreset size"),
        }
    }
}
