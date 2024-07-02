use crate::{
    coreset::{Coreset, ParallelCoreset},
    gmm::{compute_sq_norms, eucl, greedy_minimum_maximum},
};
use ndarray::{prelude::*, Data};

pub enum DiversityKind {
    /// Maximize the minimum distance
    RemoteEdge,
    RemoteClique,
}

impl DiversityKind {
    fn solve<S: Data<Elem = f32>>(&self, data: &ArrayBase<S, Ix2>, k: usize) -> Array1<usize> {
        assert!(data.nrows() > k);
        match self {
            Self::RemoteEdge => {
                let (sol, _, _) = greedy_minimum_maximum(data, k);
                sol
            }
            Self::RemoteClique => {
                // This gives a two approximation
                let sol = maximum_weight_matching(data, k);
                // The gmm greedy algorithm might give a solution with a better cost than the
                // matching algorithm, at times. We return the best of the two.
                let (sol2, _, _) = greedy_minimum_maximum(data, k);
                if self.cost(&data.select(Axis(0), sol.as_slice().unwrap()))
                    > self.cost(&data.select(Axis(0), sol2.as_slice().unwrap()))
                {
                    log::debug!("Returning the solution from the matching");
                    sol
                } else {
                    log::debug!("Returning the solution from gmm");
                    sol2
                }
            }
        }
    }

    fn cost<S: Data<Elem = f32>>(&self, data: &ArrayBase<S, Ix2>) -> f32 {
        match self {
            Self::RemoteEdge => {
                let sq_norms = compute_sq_norms(data);
                let mut min = f32::INFINITY;
                for i in 0..data.nrows() {
                    for j in 0..i {
                        let d = eucl(&data.row(i), &data.row(j), sq_norms[i], sq_norms[j]);
                        min = min.min(d);
                    }
                }
                min
            }
            Self::RemoteClique => {
                let sq_norms = compute_sq_norms(data);
                let mut sum = 0.0;
                for i in 0..data.nrows() {
                    for j in 0..i {
                        let d = eucl(&data.row(i), &data.row(j), sq_norms[i], sq_norms[j]);
                        sum += d;
                    }
                }
                sum
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

    pub fn cost<S: Data<Elem = f32>>(&mut self, data: &ArrayBase<S, Ix2>) -> f32 {
        self.kind.cost(data)
    }

    pub fn fit<S: Data<Elem = f32>>(&mut self, data: &ArrayBase<S, Ix2>) {
        match (self.threads, self.coreset_size) {
            (1, None) => {
                self.solution.replace(self.kind.solve(data, self.k));
            }
            (1, Some(coreset_size)) => {
                let mut coreset = Coreset::new(coreset_size);
                // TODO: actually use ancillary data, if present
                coreset.fit_predict::<_, ()>(data, None);
                let data = coreset.coreset_points().unwrap();
                self.solution.replace(self.kind.solve(&data, self.k));
            }
            (threads, Some(coreset_size)) => {
                let mut coreset = ParallelCoreset::new(coreset_size, threads);
                // TODO: actually use ancillary data, if present
                coreset.fit::<_, ()>(data, None);
                let data = coreset.coreset_points().unwrap();
                self.solution.replace(self.kind.solve(&data, self.k));
            }
            _ => panic!("you should specify a coreset size"),
        }
    }

    pub fn get_solution_indices(&self) -> Option<Array1<usize>> {
        self.solution.clone()
    }
}

#[derive(PartialEq, Debug)]
struct NonNaNF32(f32);
impl From<f32> for NonNaNF32 {
    fn from(value: f32) -> Self {
        assert!(!value.is_nan());
        Self(value)
    }
}
impl Eq for NonNaNF32 {}
impl Ord for NonNaNF32 {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0.total_cmp(&other.0)
    }
}
impl PartialOrd for NonNaNF32 {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

fn maximum_weight_matching<S: Data<Elem = f32>>(
    data: &ArrayBase<S, Ix2>,
    k: usize,
) -> Array1<usize> {
    use std::cmp::Reverse;
    use std::collections::BinaryHeap;

    // First, accumulate the distances we might need
    let sq_norms = compute_sq_norms(data);
    let max_heap_size = data.nrows() * k;
    let mut heap = BinaryHeap::new();
    for i in 0..data.nrows() {
        crate::check_signals();
        for j in 0..i {
            let d = eucl(&data.row(i), &data.row(j), sq_norms[i], sq_norms[j]);
            heap.push((Reverse(NonNaNF32::from(d)), i, j));
            if heap.len() > max_heap_size {
                heap.pop();
            }
        }
    }
    let dists = heap.into_sorted_vec();

    // Then, compute the matching
    let mut flags = vec![false; data.nrows()];
    let mut result = Vec::with_capacity(k);
    let mut dists_iter = dists.into_iter();
    while result.len() / 2 < k / 2 {
        // we express the condition in terms of pairs
        let (_d, i, j) = dists_iter.next().unwrap();
        if !flags[i] && !flags[j] {
            result.push(i);
            result.push(j);
            flags[i] = true;
            flags[j] = true;
        }
    }
    // Take care of the k-odd case
    if result.len() < k {
        // pick an arbitrary point
        let i = flags.into_iter().position(|f| !f).unwrap();
        result.push(i);
    }

    result.into()
}
