//! Implementation of several related matroid utilities.

use std::collections::BTreeSet;

pub trait Matroid {
    /// The type of the items of the sets over which we have the matroid
    type Item;

    /// checks whether the given subset (passed as a set of indices) of the given set is
    /// independent.
    fn is_independent(&self, set: &[Self::Item], subset: &BTreeSet<usize>) -> bool;

    /// builds an independent set of the requested size, returning the set of indices into the
    /// input vector. If no independent set of size `k` is present (i.e. the matroid has a smaller
    /// rank), then returns None.
    fn independent_set_of_size(&self, set: &[Self::Item], k: usize) -> Option<BTreeSet<usize>> {
        self.independent_set_of_size_in(set, 0..set.len(), k)
    }

    fn independent_set_of_size_in<I: IntoIterator<Item = usize>>(
        &self,
        set: &[Self::Item],
        subset: I,
        k: usize,
    ) -> Option<BTreeSet<usize>> {
        if set.len() < k {
            return None;
        }

        let mut is = BTreeSet::new();

        for i in subset {
            is.insert(i);
            if !self.is_independent(set, &is) {
                is.remove(&i);
            }
            if is.len() == k {
                return Some(is);
            }
        }

        // There is no independent set of the given size
        None
    }
}

impl Matroid for () {
    type Item = ();
    fn is_independent(&self, _set: &[Self::Item], _subset: &BTreeSet<usize>) -> bool {
        unreachable!()
    }
}

pub struct PartitionMatroid {
    category_caps: Vec<usize>,
}

impl PartitionMatroid {
    pub fn new(category_caps: Vec<usize>) -> Self {
        Self { category_caps }
    }
}

impl Matroid for PartitionMatroid {
    type Item = usize;

    fn is_independent(&self, set: &[usize], subset: &BTreeSet<usize>) -> bool {
        let mut counts = vec![0; self.category_caps.len()];
        for &i in subset {
            counts[set[i]] += 1;
            if counts[set[i]] > self.category_caps[set[i]] {
                return false;
            }
        }
        true
    }
}

pub struct TransversalMatroid {
    /// the maximum topic value
    topics: usize,
}

impl Matroid for TransversalMatroid {
    type Item = Vec<usize>;

    fn is_independent(&self, ground_set: &[Self::Item], subset: &BTreeSet<usize>) -> bool {
        subset.len() <= self.topics
            && self.maximum_matching_size(ground_set, subset) == subset.len()
    }
}

impl TransversalMatroid {
    pub(crate) fn num_topics(&self) -> usize {
        self.topics + 1
    }

    fn maximum_matching_size(&self, ground_set: &[Vec<usize>], set: &BTreeSet<usize>) -> usize {
        let n_topics = self.topics + 1;
        let mut visited = vec![false; n_topics];
        let mut representatives: Vec<Option<usize>> = vec![None; n_topics];
        for &idx in set {
            // reset the flags
            visited.fill(false);
            // try to accomodate the new element
            self.find_matching_for(ground_set, idx, &mut representatives, &mut visited);
        }

        representatives.iter().filter(|opt| opt.is_some()).count()
    }

    fn find_matching_for(
        &self,
        ground_set: &[Vec<usize>],
        idx: usize,
        representatives: &mut [Option<usize>],
        visited: &mut [bool],
    ) -> bool {
        for &topic in ground_set[idx].iter() {
            assert!(topic <= self.topics);
            if !visited[topic] {
                visited[topic] = true;
                let can_set = if let Some(displacing_idx) = representatives[topic] {
                    // try to move the representative to another set
                    self.find_matching_for(ground_set, displacing_idx, representatives, visited)
                } else {
                    true
                };

                if can_set {
                    representatives[topic].replace(idx);
                    return true;
                }
            }
        }

        false
    }
}
