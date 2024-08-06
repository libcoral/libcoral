use std::collections::BTreeSet;

use crate::{
    coreset::{CoresetBuilder, ExtractCoresetPoints},
    gmm::{compute_sq_norms, eucl, greedy_minimum_maximum},
    matroid::{Matroid, PartitionMatroid, TransversalMatroid},
};
use ndarray::{prelude::*, Data};

#[derive(Clone, Copy, Debug)]
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

    fn solve_matroid<S: Data<Elem = f32>, A, M: Matroid<Item = A>>(
        &self,
        data: &ArrayBase<S, Ix2>,
        ancillary: &[A],
        k: usize,
        matroid: &M,
        epsilon: f32,
    ) -> Array1<usize> {
        assert!(data.nrows() > k);
        match self {
            Self::RemoteEdge => {
                unimplemented!("no known approximation algorithm exists, use explicit enumeration on a small enough coreset")
            }
            Self::RemoteClique => local_search(data, ancillary, k, epsilon, matroid, *self),
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

    fn cost_subset<S: Data<Elem = f32>>(
        &self,
        data: &ArrayBase<S, Ix2>,
        subset: &BTreeSet<usize>,
    ) -> f32 {
        match self {
            Self::RemoteEdge => {
                let sq_norms = compute_sq_norms(data);
                let mut min = f32::INFINITY;
                for &i in subset.iter() {
                    for &j in subset.iter() {
                        if i < j {
                            let d = eucl(&data.row(i), &data.row(j), sq_norms[i], sq_norms[j]);
                            min = min.min(d);
                        }
                    }
                }
                min
            }
            Self::RemoteClique => {
                let sq_norms = compute_sq_norms(data);
                let mut sum = 0.0;
                for &i in subset.iter() {
                    for &j in subset.iter() {
                        if i < j {
                            let d = eucl(&data.row(i), &data.row(j), sq_norms[i], sq_norms[j]);
                            sum += d;
                        }
                    }
                }
                sum
            }
        }
    }
}

pub struct DiversityMaximization<M: Matroid> {
    k: usize,
    kind: DiversityKind,
    coreset_size: Option<usize>,
    threads: usize,
    epsilon: f32,
    matroid: Option<M>,
}

impl DiversityMaximization<()> {
    pub fn new(k: usize, kind: DiversityKind) -> Self {
        Self {
            k,
            kind,
            coreset_size: None,
            threads: 1,
            epsilon: 0.01,
            matroid: None,
        }
    }
}

impl<M: Matroid + Sync> DiversityMaximization<M>
where
    M::Item: Send + Sync + Clone,
{
    pub fn with_threads(self, threads: usize) -> Self {
        Self { threads, ..self }
    }

    pub fn with_epsilon(self, epsilon: f32) -> Self {
        Self { epsilon, ..self }
    }

    pub fn with_matroid<M2: Matroid>(self, matroid: M2) -> DiversityMaximization<M2> {
        DiversityMaximization {
            matroid: Some(matroid),
            k: self.k,
            kind: self.kind,
            coreset_size: self.coreset_size,
            epsilon: self.epsilon,
            threads: self.threads,
        }
    }

    pub fn with_coreset(self, coreset_size: usize) -> Self {
        Self {
            coreset_size: Some(coreset_size),
            ..self
        }
    }
}

impl<M: Matroid + SelectDelegates<M::Item> + Sync> DiversityMaximization<M>
where
    M::Item: Send + Sync + Clone,
{
    /// Solves the diversity maximization problem on the given data. Ancillary data is required if
    /// the problem is constrained by a matroid.
    ///
    /// Returns an integer array of the indices of the points included in the solution.
    pub fn solve<S: Data<Elem = f32>>(
        &self,
        data: &ArrayBase<S, Ix2>,
        ancillary: Option<&[M::Item]>,
    ) -> Array1<usize> {
        if let Some(coreset_size) = self.coreset_size {
            if let Some(matroid) = self.matroid.as_ref() {
                let coreset = CoresetBuilder::with_tau(coreset_size)
                    .with_extractor(MatroidExtractCoresetPoints::new(self.k, matroid))
                    .with_threads(self.threads)
                    .fit(data, ancillary);
                let indices = self.kind.solve_matroid(
                    &coreset.points(),
                    coreset
                        .ancillary()
                        .expect("ancillary data is required with a matroid"),
                    self.k,
                    matroid,
                    self.epsilon,
                );
                coreset.invert_index(&indices)
            } else {
                let coreset = CoresetBuilder::with_tau(coreset_size)
                    .with_threads(self.threads)
                    .fit(data, None);
                let indices = self.kind.solve(&coreset.points(), self.k);
                // reverse the indices to the original data ones
                coreset.invert_index(&indices)
            }
        } else {
            if self.threads > 1 {
                log::warn!("no coreset is being constructed, use only a single thread");
            }
            if let Some(matroid) = self.matroid.as_ref() {
                self.kind.solve_matroid(
                    data,
                    ancillary.expect("ancillary data is required with a matroid"),
                    self.k,
                    matroid,
                    self.epsilon,
                )
            } else {
                self.kind.solve(data, self.k)
            }
        }
    }
}

#[derive(Clone)]
struct MatroidExtractCoresetPoints<'matroid, M: Matroid + SelectDelegates<M::Item> + Sync>
where
    M::Item: Clone + Send + Sync,
{
    k: usize,
    matroid: &'matroid M,
}
impl<'matroid, M: Matroid + SelectDelegates<M::Item> + Sync>
    MatroidExtractCoresetPoints<'matroid, M>
where
    M::Item: Clone + Send + Sync,
{
    fn new(k: usize, matroid: &'matroid M) -> Self {
        Self { k, matroid }
    }
}

impl<'matroid, M: Matroid + SelectDelegates<M::Item> + Sync> ExtractCoresetPoints
    for MatroidExtractCoresetPoints<'matroid, M>
where
    M::Item: Clone + Send + Sync,
{
    type Ancillary = M::Item;
    fn extract_coreset_points<S: Data<Elem = f32>>(
        &self,
        _data: &ArrayBase<S, Ix2>,
        ancillary: Option<&[Self::Ancillary]>,
        assigned: &[usize],
    ) -> Array1<usize> {
        let ancillary = ancillary.expect("ancillary data is required with matroids");
        self.matroid.select_delegates(self.k, ancillary, assigned)
    }
}

/// This trait gives the chance to some matroids (e.g. transversal matroids) to select a set of
/// delegates that is not an independent set. This is useful for instance for the transversal
/// matroid in some corner cases (see section 3.1.2 of the paper).
pub trait SelectDelegates<A> {
    fn select_delegates(&self, k: usize, ground_set: &[A], assigned: &[usize]) -> Array1<usize>;
}

impl SelectDelegates<()> for () {
    fn select_delegates(
        &self,
        _k: usize,
        _ground_set: &[()],
        _assigned: &[usize],
    ) -> Array1<usize> {
        unreachable!()
    }
}

impl SelectDelegates<usize> for PartitionMatroid {
    fn select_delegates(
        &self,
        k: usize,
        ground_set: &[usize],
        assigned: &[usize],
    ) -> Array1<usize> {
        let mut is = BTreeSet::new();

        for i in assigned {
            is.insert(*i);
            if !self.is_independent(ground_set, &is) {
                is.remove(i);
            }
            if is.len() == k {
                break;
            }
        }
        Array1::from_iter(is)
    }
}

impl SelectDelegates<Vec<usize>> for TransversalMatroid {
    fn select_delegates(
        &self,
        k: usize,
        ground_set: &[Vec<usize>],
        assigned: &[usize],
    ) -> Array1<usize> {
        if let Some(iset) = self.independent_set_of_size_in(ground_set, assigned.iter().copied(), k)
        {
            Array1::from_iter(iset)
        } else {
            // we have to pick at most k representatives from each topic if there is
            // no independent set of size k
            let mut counts = vec![0; self.num_topics()];
            let mut selection = Vec::new();
            for &idx in assigned {
                let should_add = ground_set[idx].iter().any(|topic| counts[*topic] < k);
                if should_add {
                    selection.push(idx);
                    for &topic in ground_set[idx].iter() {
                        counts[topic] += 1;
                    }
                    if counts.iter().all(|cnt| *cnt >= k) {
                        break;
                    }
                }
            }
            Array1::from_vec(selection)
        }
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

/// Runs the local search algorithm on the given data and ancillary information.
/// The ancillary information is used to enfoce the given matroid constraint.
fn local_search<A, S, M>(
    data: &ArrayBase<S, Ix2>,
    ancillary: &[A],
    k: usize,
    epsilon: f32,
    matroid: &M,
    diversity: DiversityKind,
) -> Array1<usize>
where
    S: Data<Elem = f32>,
    M: Matroid<Item = A>,
{
    if data.nrows() <= k {
        return Array1::from_iter(0..data.nrows());
    }

    // Pick an initial arbitrary maximal independent set
    let mut iset = matroid
        .independent_set_of_size(ancillary, k)
        .expect("matroid rank smaller than k");

    let mut found_improving_swap = true;

    while found_improving_swap {
        found_improving_swap = false;
        let threshold = (1.0 + epsilon / k as f32) * diversity.cost_subset(data, &iset);
        for i in 0..data.nrows() {
            if found_improving_swap {
                break;
            }
            if iset.contains(&i) {
                for j in 0..i {
                    if !iset.contains(&j) {
                        iset.remove(&i);
                        iset.insert(j);

                        if matroid.is_independent(ancillary, &iset)
                            && diversity.cost_subset(data, &iset) > threshold
                        {
                            found_improving_swap = true;
                            break;
                        } else {
                            iset.remove(&j);
                            iset.insert(i);
                        }
                    }
                }
            }
        }
    }

    Array1::from_iter(iset)
}
