//! Implementation of several related matroid utilities.
//!
//! The original code for this module was implemented as part of
//! [`macaco`](https://github.com/Cecca/macaco)

use log::*;
use std::{cell::RefCell, rc::Rc};
use std::{collections::HashMap, marker::PhantomData};
use std::{collections::VecDeque, time::Instant};
use thread_local::ThreadLocal;

pub trait Matroid<T> {
    fn is_independent(&self, set: &[T]) -> bool;
    fn is_independent_ref(&self, set: &[&T]) -> bool;

    fn rank(&self) -> usize;

    fn maximal_independent_set<'a>(&self, set: &[&'a T]) -> Vec<&'a T>;

    fn is_maximal(&self, is: &[T], set: &[T]) -> bool {
        let mut is = Vec::from_iter(is.iter());
        for x in set {
            is.push(x);
            if self.is_independent_ref(&is) {
                return false;
            }
            is.pop();
        }
        true
    }
}

pub trait IndependentSet<'points, T> {
    fn update(&mut self, x: &'points T);
    fn points(&self) -> Vec<&'points T>;
}

struct PartitionIndependentSet<'matroid, 'points, T: PartitionMatroidElement> {
    matroid: &'matroid PartitionMatroid<T>,
    points: Vec<&'points T>,
}

impl<'matroid, 'points, T: PartitionMatroidElement> IndependentSet<'points, T>
    for PartitionIndependentSet<'matroid, 'points, T>
{
    fn update(&mut self, x: &'points T) {
        self.points.push(x);
        if !self.matroid.is_independent_ref(&self.points) {
            self.points.pop();
        }
    }

    fn points(&self) -> Vec<&'points T> {
        todo!()
    }
}

struct TransversalIndependentSet<'matroid, 'points, T: TransversalMatroidElement> {
    matroid: &'matroid TransversalMatroid<T>,
    points: Vec<&'points T>,
    visited: Vec<bool>,
    representatives: Vec<Option<usize>>,
}

impl<'matroid, 'points, T: TransversalMatroidElement> IndependentSet<'points, T>
    for TransversalIndependentSet<'matroid, 'points, T>
{
    fn update(&mut self, x: &'points T) {
        // First, extend the set, we shall see if such extension is valid
        self.points.push(x);

        // reset the flags
        self.visited.fill(false);

        // At this point the assumption is that `self.points` already contains an independent set,
        // with a valid assignment in `representatives`.
        //
        // Therefore, we try to find a matching for the point we just pushed
        find_matching_for(
            &self.matroid.topics,
            &self.points,
            self.points.len() - 1,
            &mut self.representatives,
            &mut self.visited,
        );

        // Now we count the size of the independent set by counting the number of topics
        // which have some representative
        let size = self
            .representatives
            .iter()
            .filter(|opt| opt.is_some())
            .count();

        debug_assert_eq!(size, self.matroid.maximum_matching_size(&self.points));

        // If the size of the independent set in `self.points` is smaller than
        // `self.points` itself, it means that the point we just inserted does not
        // fit in there, so we shall remove it
        if size != self.points.len() {
            self.points.pop();
        }
    }

    fn points(&self) -> Vec<&'points T> {
        todo!()
    }
}

fn find_matching_for<T: TransversalMatroidElement>(
    topics: &[u32],
    set: &[&T],
    idx: usize,
    representatives: &mut [Option<usize>],
    visited: &mut [bool],
) -> bool {
    for topic in set_intersection(topics.iter().copied(), set[idx].topics().iter().copied()) {
        if let Ok(topic) = topics.binary_search(&topic) {
            if !visited[topic] {
                visited[topic] = true;
                let can_set = if let Some(displacing_idx) = representatives[topic] {
                    // try to move the representative to another set
                    find_matching_for(topics, set, displacing_idx, representatives, visited)
                } else {
                    true
                };

                if can_set {
                    representatives[topic].replace(idx);
                    return true;
                }
            }
        }
    }

    false
}

pub fn augment<T: Clone + PartialEq>(
    matroid: Rc<dyn Matroid<T>>,
    independent_set: &[T],
    set: &[T],
) -> Vec<T> {
    let mut is = Vec::from_iter(independent_set.iter());
    for x in set {
        if !is.contains(&x) {
            is.push(x);
            if !matroid.is_independent_ref(&is) {
                is.pop();
            }
        }
    }

    is.into_iter().cloned().collect()
}

/// Element of a set on which we can impose a transversal matroid
pub trait TransversalMatroidElement {
    fn topics(&self) -> &[u32];
}

impl<V: AsRef<[u32]>> TransversalMatroidElement for V {
    fn topics(&self) -> &[u32] {
        self.as_ref()
    }
}

pub struct TransversalMatroid<T> {
    topics: Vec<u32>,
    _marker: PhantomData<T>,
    // Temporary storage to be reused across invocations, prior to cleaning
    scratch_visited: ThreadLocal<RefCell<Vec<bool>>>,
    scratch_representatives: ThreadLocal<RefCell<Vec<Option<usize>>>>,
    scratch_pairing_set: ThreadLocal<RefCell<Vec<Option<usize>>>>,
    scratch_pairing_topic: ThreadLocal<RefCell<Vec<Option<usize>>>>,
    scratch_distances: ThreadLocal<RefCell<Vec<usize>>>,
    scratch_queue: ThreadLocal<RefCell<VecDeque<usize>>>,
}

impl<T: TransversalMatroidElement> Matroid<T> for TransversalMatroid<T> {
    fn rank(&self) -> usize {
        self.topics.len()
    }

    fn is_independent(&self, set: &[T]) -> bool {
        let set: Vec<&T> = set.iter().collect();
        debug_assert!(self.maximum_matching_size(&set) == self.maximum_matching_size2(&set));
        set.len() < self.topics.len() && self.maximum_matching_size(&set) == set.len()
    }

    fn is_independent_ref(&self, set: &[&T]) -> bool {
        debug_assert!(self.maximum_matching_size(&set) == self.maximum_matching_size2(&set));
        set.len() < self.topics.len() && self.maximum_matching_size(&set) == set.len()
    }

    fn maximal_independent_set<'a>(&self, set: &[&'a T]) -> Vec<&'a T>
    where
        Self: Sized,
    {
        let mut is = TransversalIndependentSet {
            matroid: self,
            points: Vec::new(),
            visited: vec![false; self.topics.len()],
            representatives: vec![None; self.topics.len()],
        };

        for x in set {
            is.update(*x);
        }

        is.points
    }
}

impl<T: TransversalMatroidElement> TransversalMatroid<T> {
    pub fn new(topics: Vec<u32>) -> Self {
        Self {
            topics,
            _marker: PhantomData,
            scratch_visited: ThreadLocal::new(),
            scratch_representatives: ThreadLocal::new(),
            scratch_pairing_set: ThreadLocal::new(),
            scratch_pairing_topic: ThreadLocal::new(),
            scratch_distances: ThreadLocal::new(),
            scratch_queue: ThreadLocal::new(),
        }
    }

    fn maximum_matching_size2(&self, set: &[&T]) -> usize {
        // Clear the thread local temporary storage
        let mut pairing_set = self
            .scratch_pairing_set
            .get_or(|| RefCell::new(Vec::new()))
            .borrow_mut();
        let mut pairing_topic = self
            .scratch_pairing_topic
            .get_or(|| RefCell::new(Vec::new()))
            .borrow_mut();
        let mut distances = self
            .scratch_distances
            .get_or(|| RefCell::new(Vec::new()))
            .borrow_mut();
        pairing_set.clear();
        pairing_set.resize(set.len(), None);
        pairing_topic.clear();
        pairing_topic.resize(self.topics.len(), None);
        distances.clear();
        distances.resize(set.len() + 1, std::usize::MAX);

        let mut matching_size = 0;

        while self.bfs(set, &mut pairing_set, &mut pairing_topic, &mut distances) {
            for u in 0..pairing_set.len() {
                if pairing_set[u].is_none()
                    && self.dfs(u, set, &mut pairing_set, &mut pairing_topic, &mut distances)
                {
                    matching_size += 1;
                }
            }
        }

        matching_size
    }

    fn topic_index(&self, topic: u32) -> Option<usize> {
        (0..self.topics.len()).find(|&i| unsafe { *self.topics.get_unchecked(i) == topic })
    }

    fn bfs(
        &self,
        set: &[&T],
        pairing_set: &mut [Option<usize>],
        pairing_topic: &mut [Option<usize>],
        // the last element is the special dummy vertex
        distances: &mut [usize],
    ) -> bool {
        let dummy = distances.len() - 1;
        let infty = std::usize::MAX;

        let mut queue = self
            .scratch_queue
            .get_or(|| RefCell::new(VecDeque::new()))
            .borrow_mut();
        queue.clear();

        for u in 0..pairing_set.len() {
            if pairing_set[u].is_none() {
                queue.push_back(u);
                distances[u] = 0;
            } else {
                distances[u] = infty;
            }
        }
        distances[dummy] = infty;

        while let Some(u) = queue.pop_front() {
            if distances[u] < distances[dummy] && u != dummy {
                for topic in set[u].topics() {
                    if let Some(v) = self.topic_index(*topic) {
                        let pair_v = pairing_topic[v].unwrap_or(dummy);
                        if distances[pair_v] == infty {
                            distances[pair_v] = distances[u] + 1;
                            queue.push_back(pair_v);
                        }
                    }
                }
            }
        }

        distances[dummy] < std::usize::MAX
    }

    fn dfs(
        &self,
        root: usize,
        set: &[&T],
        pairing_set: &mut [Option<usize>],
        pairing_topic: &mut [Option<usize>],
        // the last element is the special dummy vertex
        distances: &mut [usize],
    ) -> bool {
        let dummy = distances.len() - 1;
        let infty = std::usize::MAX;
        let u = root;

        if u != dummy {
            for topic in set[u].topics() {
                if let Some(v) = self.topic_index(*topic) {
                    let pair_v = pairing_topic[v].unwrap_or(dummy);
                    if distances[pair_v] == distances[u] + 1
                        && self.dfs(pair_v, set, pairing_set, pairing_topic, distances)
                    {
                        pairing_set[u].replace(v);
                        pairing_topic[v].replace(u);
                        return true;
                    }
                }
            }
            distances[u] = infty;
            false
        } else {
            true
        }
    }

    fn maximum_matching_size(&self, set: &[&T]) -> usize {
        let n_topics = self.topics.len();
        let mut visited = self
            .scratch_visited
            .get_or(|| RefCell::new(vec![false; n_topics]))
            .borrow_mut();
        visited.fill(false);
        let mut representatives = self
            .scratch_representatives
            .get_or(|| RefCell::new(vec![None; self.topics.len()]))
            .borrow_mut();
        representatives.fill(None);

        for idx in 0..set.len() {
            // reset the flags
            visited.fill(false);
            // try to accomodate the new element
            self.find_matching_for(set, idx, &mut representatives, &mut visited);
        }

        representatives.iter().filter(|opt| opt.is_some()).count()
    }

    fn topic_idx(&self, topic: u32) -> Option<usize> {
        self.topics.binary_search(&topic).ok()
    }

    fn find_matching_for(
        &self,
        set: &[&T],
        idx: usize,
        representatives: &mut [Option<usize>],
        visited: &mut [bool],
    ) -> bool {
        for topic in set_intersection(
            self.topics.iter().copied(),
            set[idx].topics().iter().copied(),
        ) {
            if let Some(topic) = self.topic_idx(topic) {
                if !visited[topic] {
                    visited[topic] = true;
                    let can_set = if let Some(displacing_idx) = representatives[topic] {
                        // try to move the representative to another set
                        self.find_matching_for(set, displacing_idx, representatives, visited)
                    } else {
                        true
                    };

                    if can_set {
                        representatives[topic].replace(idx);
                        return true;
                    }
                }
            }
        }

        false
    }
}

fn set_intersection<I1: IntoIterator<Item = u32>, I2: IntoIterator<Item = u32>>(
    a: I1,
    b: I2,
) -> impl Iterator<Item = u32> {
    let mut i1 = a.into_iter();
    let mut i2 = b.into_iter();

    let mut cur_a = i1.next();
    let mut cur_b = i2.next();

    std::iter::from_fn(move || loop {
        match (cur_a, cur_b) {
            (Some(a), Some(b)) => {
                if a < b {
                    cur_a = i1.next();
                } else if a > b {
                    cur_b = i2.next();
                } else {
                    cur_a = i1.next();
                    cur_b = i2.next();
                    return Some(a);
                }
            }
            _ => return None,
        }
    })
}

/// Element of a set on which we can impose a partition matroid
pub trait PartitionMatroidElement {
    fn category(&self) -> u32;
}

impl PartitionMatroidElement for u32 {
    fn category(&self) -> u32 {
        *self
    }
}

pub struct PartitionMatroid<T: PartitionMatroidElement> {
    categories: HashMap<u32, u32>,
    scratch: ThreadLocal<RefCell<HashMap<u32, u32>>>,
    _marker: PhantomData<T>,
}

impl<T: PartitionMatroidElement> PartitionMatroid<T> {
    pub fn new(categories: HashMap<u32, u32>) -> Self {
        Self {
            categories,
            scratch: ThreadLocal::new(),
            _marker: std::marker::PhantomData,
        }
    }
}

impl<T: PartitionMatroidElement> Matroid<T> for PartitionMatroid<T> {
    fn rank(&self) -> usize {
        self.categories.values().sum::<u32>() as usize
    }

    fn is_independent(&self, set: &[T]) -> bool {
        // perf_counters::inc_matroid_oracle_count();
        let mut counts = self
            .scratch
            .get_or(|| RefCell::new(self.categories.clone()))
            .borrow_mut();
        // reset counters
        for (k, c) in self.categories.iter() {
            *counts.get_mut(k).unwrap() = *c;
        }
        for x in set {
            let cat = x.category();
            // Categories not explicitly mentioned in the matroid
            // default to a limit of 0. This makes for a less verbose specification
            // of constraints
            if counts.get(&cat).unwrap_or(&0) == &0 {
                return false;
            } else {
                counts.get_mut(&cat).map(|c| *c -= 1);
            }
        }
        true
    }

    fn is_independent_ref(&self, set: &[&T]) -> bool {
        // perf_counters::inc_matroid_oracle_count();
        let mut counts = self
            .scratch
            .get_or(|| RefCell::new(self.categories.clone()))
            .borrow_mut();
        // reset counters
        for (k, c) in self.categories.iter() {
            *counts.get_mut(k).unwrap() = *c;
        }
        for x in set {
            let cat = x.category();
            // Categories not explicitly mentioned in the matroid
            // default to a limit of 0. This makes for a less verbose specification
            // of constraints
            if counts.get(&cat).unwrap_or(&0) == &0 {
                return false;
            } else {
                counts.get_mut(&cat).map(|c| *c -= 1);
            }
        }
        true
    }

    fn maximal_independent_set<'a>(&self, set: &[&'a T]) -> Vec<&'a T> {
        let mut is = PartitionIndependentSet {
            matroid: self,
            points: Vec::new(),
        };
        let rank = self.rank();

        for x in set {
            is.update(*x);
            if is.points.len() == rank {
                break;
            }
        }

        is.points
    }
}

pub trait Weight {
    fn weight(&self) -> u32;
}

impl Weight for (usize, &Vec<usize>) {
    fn weight(&self) -> u32 {
        self.1.len() as u32
    }
}

pub fn weighted_matroid_intersection<'a, V: Weight, M1: Matroid<V>, M2: Matroid<V>>(
    set: &'a [V],
    m1: &M1,
    m2: &M2,
) -> impl Iterator<Item = &'a V> + 'a {
    let mut independent_set = vec![false; set.len()];
    let mut graph = ExchangeGraph::default();
    let mut last = 0;
    while augment_intersection(set, m1, m2, &mut independent_set, &mut graph) {
        // All of the statements in this while body are for debug purposes
        let current_size = independent_set.iter().filter(|included| **included).count();
        #[cfg(debug_assertions)]
        {
            let current_items: Vec<&V> = independent_set
                .iter()
                .enumerate()
                .filter(|(_, included)| **included)
                .map(|(i, _)| &set[i])
                .collect();
            debug_assert!(m1.is_independent_ref(&current_items));
            debug_assert!(m2.is_independent_ref(&current_items));
        }
        debug!(
            "Independent set of size {} and weight {}",
            current_size,
            independent_set
                .iter()
                .enumerate()
                .filter(|(_, included)| **included)
                .map(|(i, _)| set[i].weight())
                .sum::<u32>()
        );
        assert!(current_size > last);
        last = current_size;
    }
    independent_set
        .into_iter()
        .zip(set.iter())
        .filter(|p| p.0)
        .map(|p| p.1)
}

/// Augment the given independent set in place. If there is no common independent set larger than the given one,
/// return false, otherwise return true
fn augment_intersection<'a, V: Weight, M1: Matroid<V>, M2: Matroid<V>>(
    set: &[V],
    m1: &M1,
    m2: &M2,
    independent_set: &mut [bool],
    graph: &mut ExchangeGraph,
) -> bool {
    graph.update(set, m1, m2, independent_set);

    let timer = Instant::now();
    let mut independent_set_elements: Vec<&V> = independent_set
        .iter()
        .zip(set.iter())
        .filter(|p| *p.0)
        .map(|p| p.1)
        .collect();
    debug_assert!(m1.is_independent_ref(&independent_set_elements));
    debug_assert!(m2.is_independent_ref(&independent_set_elements));
    debug!(
        "Retrieved references to independent set elements in {:?}",
        timer.elapsed()
    );

    let timer = Instant::now();
    // define the source and destination sets.
    // When the input independent set is empty, it makes sense that the sets
    // x1 and x2 are equal, and corresponding to the full set.
    let x1: Vec<usize> = independent_set
        .iter()
        .enumerate()
        .filter(|(i, included)| {
            !**included && {
                independent_set_elements.push(&set[*i]);
                let b = m1.is_independent_ref(&independent_set_elements);
                independent_set_elements.pop();
                b
            }
        })
        .map(|p| p.0)
        .collect();
    let x2: Vec<usize> = independent_set
        .iter()
        .enumerate()
        .filter(|(i, included)| {
            !**included && {
                independent_set_elements.push(&set[*i]);
                let b = m2.is_independent_ref(&independent_set_elements);
                independent_set_elements.pop();
                b
            }
        })
        .map(|p| p.0)
        .collect();

    // --------------------------------------------------------------------------
    // now we look at paths from x1 to x2 in the graph. To do so, we run the
    // Bellman-Ford algorithm from each source vertex to get to the destinations.
    // For efficiency, if x2 is larger than x1, then we flip the edges in the
    // graph and use nodes in x2 as sources, so to do fewer invocations of the
    // shortest-paths algorithm.
    // The caveat is that weights are on the nodes instead of the edges.
    // Therefore a path formed by a single node is a valid path. However,
    // in our context, we accept such a path as valid *only* if the node is
    // both in x1 and in x2, that is it is both a source and a destination.
    //
    debug!(
        "looking for best paths from {} sources to {} destinations (defined in {:?})",
        x1.len(),
        x2.len(),
        timer.elapsed()
    );
    let timer = Instant::now();

    let singleton_paths: Vec<(i32, Vec<usize>)> =
        set_intersection(x1.iter().map(|i| *i as u32), x2.iter().map(|i| *i as u32))
            .map(|i| (graph.length[i as usize], vec![i as usize]))
            .collect();
    debug!("There are {} singleton paths", singleton_paths.len());

    // compute paths from the set of smaller cardinality
    // let (sources, destinations) = (x1, x2);
    let (sources, destinations) = if x1.len() < x2.len() {
        (x1, x2)
    } else {
        debug!("more sources than destinations, flipping the edges");
        graph.reverse(); // flip the edges first
        (x2, x1)
    };
    // find the best path, if any
    if let Some((_, path)) = sources
        .iter()
        .flat_map(|i| graph.bellman_ford(*i, &destinations))
        .chain(singleton_paths.into_iter())
        .min_by_key(|(d, path)| (*d, path.len()))
    {
        for i in path {
            // Computing the xor on the flags array is equivalent to computing the
            // symmetric difference of the path and the independent set
            independent_set[i] ^= true;
        }
        debug!("augmenting path found (if any) in {:?}", timer.elapsed());
        true
    } else {
        debug!("no augmenting path! {:?}", timer.elapsed());
        false
    }
}

#[derive(Clone, Default)]
struct ExchangeGraph {
    reversed: bool,
    length: Vec<i32>,
    edges: Vec<(usize, usize)>,
    distance: Vec<Option<i32>>,
    predecessor: Vec<Option<usize>>,
}

impl ExchangeGraph {
    fn update<'a, V: Weight, M1: Matroid<V>, M2: Matroid<V>>(
        &mut self,
        set: &[V],
        m1: &M1,
        m2: &M2,
        independent_set: &mut [bool],
    ) {
        self.length.clear();
        self.distance.clear();
        self.predecessor.clear();
        self.edges.clear();
        self.reversed = false;

        let timer = std::time::Instant::now();
        let n = set.len();

        // reuse already allocated space, if any
        self.length
            .extend(set.iter().zip(independent_set.iter()).map(|(v, included)| {
                if *included {
                    v.weight() as i32
                } else {
                    -(v.weight() as i32)
                }
            }));
        debug!("computed lengths in {:?}", timer.elapsed());

        // edge building. This is the costly operation.
        //
        // edge invariants, where I is the independent set:
        //  - (y, x) is in the graph iff I - y + x is independent in m1
        //  - (x, y) is in the graph iff I - y + x is independent in m2
        let timer = std::time::Instant::now();
        for (y, _) in independent_set.iter().enumerate().filter(|p| *p.1) {
            // The independent set without y
            let mut scratch: Vec<&V> = independent_set
                .iter()
                .enumerate()
                .filter(|p| *p.1 && p.0 != y)
                .map(|p| &set[p.0])
                .collect();
            for (x, _) in independent_set.iter().enumerate().filter(|p| !p.1) {
                scratch.push(&set[x]);
                // call the independent set oracle and possibly push the edge only if it
                // is not already in the list of edges
                if m1.is_independent_ref(&scratch) {
                    self.edges.push((y, x));
                }
                if m2.is_independent_ref(&scratch) {
                    self.edges.push((x, y));
                }
                scratch.pop();
            }
        }
        debug!("created edges in {:?}", timer.elapsed());
        debug!(
            "Created exchange graph with {} edges and {} nodes in {:?}",
            self.edges.len(),
            n,
            timer.elapsed()
        );

        self.distance.resize(n, None);
        self.predecessor.resize(n, None);
    }

    /// reverse the arcs
    fn flip_edges(&mut self) {
        for edge in self.edges.iter_mut() {
            *edge = (edge.1, edge.0);
        }
        self.reversed = !self.reversed;
    }

    fn reverse(&mut self) {
        if !self.reversed {
            self.flip_edges();
        }
        assert!(self.reversed);
    }

    /// Iterator on the paths reaching `i`
    fn iter_path<'a>(&'a self, i: usize) -> impl Iterator<Item = usize> + 'a {
        let mut current = Some(i);
        std::iter::from_fn(move || {
            if let Some(i) = current {
                let toret = current;
                current = self.predecessor[i];
                toret
            } else {
                None
            }
        })
    }

    /// return the shortest of all the shortest paths starting from `src`
    /// and going to `dsts` as a sequence of indices.
    /// Ties are broken by picking the one with fewest edges.
    ///
    /// If no path exist, then None is returned.
    /// Otherwise, return the weight of the path and the sequence of its nodes
    fn bellman_ford(&mut self, src: usize, dsts: &[usize]) -> Option<(i32, Vec<usize>)> {
        let n = self.length.len();

        // reset the support arrays
        self.distance.fill(None);
        self.predecessor.fill(None);

        self.distance[src].replace(self.length[src]);

        // compute shortest paths
        for _ in 0..n {
            let mut updated = false;
            for &(u, v) in &self.edges {
                // edge relaxation
                if let Some(du) = self.distance[u] {
                    if let Some(dv) = self.distance[v] {
                        if du + self.length[v] < dv {
                            updated = true;
                            self.distance[v].replace(du + self.length[v]);
                            self.predecessor[v].replace(u);
                        }
                    } else {
                        updated = true;
                        self.distance[v].replace(du + self.length[v]);
                        self.predecessor[v].replace(u);
                    }
                }
            }
            if !updated {
                // Early break if no nodes are updated: it means we explored all paths.
                break;
            }
        }

        // Check the lengths of the paths
        #[cfg(debug_assertions)]
        for dst in dsts.iter() {
            if let Some(d) = self.distance[*dst] {
                let path: Vec<usize> = self.iter_path(*dst).collect();
                let weights: Vec<i32> = path.iter().map(|v| self.length[*v]).collect();
                let w = weights.iter().sum::<i32>();
                assert!(w == d);
            }
        }

        // using flat map we filter out unreachable destinations
        if let Some(shortest_dist) = dsts.iter().flat_map(|i| self.distance[*i]).min() {
            // Look, among the destinations
            dsts.iter()
                // for the ones at minimum distance
                .filter(|i| {
                    self.distance[**i].is_some() && self.distance[**i].unwrap() == shortest_dist
                })
                // which consist of more than one node (singleton paths are handled by the caller)
                .filter(|i| self.iter_path(**i).count() > 1)
                // and reached with minimum number of steps
                .min_by_key(|i| self.iter_path(**i).count())
                // for that one, materialize the path
                .map(|i| {
                    let path: Vec<usize> = self.iter_path(*i).collect();
                    assert!(path.len() > 0);
                    (self.distance[*i].unwrap(), path)
                })
        } else {
            None
        }
    }
}
