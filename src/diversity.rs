use crate::gmm::greedy_minimum_maximum;
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
}

impl DiversityMaximization {
    pub fn new(k: usize, kind: DiversityKind) -> Self {
        Self {
            k,
            kind,
            solution: None,
        }
    }

    pub fn fit<S: Data<Elem = f32>>(&mut self, data: &ArrayBase<S, Ix2>) {
        self.solution.replace(self.kind.solve(data, self.k));
    }
}
