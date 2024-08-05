//! Implementation of several related matroid utilities.

use std::{cell::RefCell, collections::BTreeSet};

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
        if set.len() < k {
            return None;
        }

        let mut is = BTreeSet::new();

        for i in 0..set.len() {
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
