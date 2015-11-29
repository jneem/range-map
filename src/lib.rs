// Copyright 2015 Joe Neeman.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![cfg_attr(test, feature(plugin))]
#![cfg_attr(test, plugin(quickcheck_macros))]

extern crate itertools;
extern crate num;

#[cfg(test)]
extern crate quickcheck;

use itertools::Itertools;
use num::traits::PrimInt;
use std::cmp::{max, min, Ordering};
use std::fmt::{Debug, Formatter};
use std::iter::FromIterator;
use std::{mem, usize};

const DISPLAY_LIMIT: usize = 10;

/// A range of elements, including the endpoints.
#[derive(Copy, Clone, Hash, PartialEq, PartialOrd, Eq, Ord)]
pub struct Range<T> {
    pub start: T,
    pub end: T,
}

impl<T: Debug> Debug for Range<T> {
    fn fmt(&self, f: &mut Formatter) -> Result<(), std::fmt::Error> {
        try!(self.start.fmt(f));
        try!(f.write_str(" -- "));
        try!(self.end.fmt(f));
        Ok(())
    }
}

impl<T: PrimInt> Range<T> {
    /// Creates a new range with the given start and endpoints (inclusive).
    ///
    /// # Panics
    ///  - if `start` is strictly larger than `end`
    pub fn new(start: T, end: T) -> Range<T> {
        if start > end {
            panic!("Ranges must be ordered");
        }
        Range { start: start, end: end }
    }

    /// Creates a new range containing everything.
    pub fn full() -> Range<T> {
        Range { start: T::min_value(), end: T::max_value() }
    }

    /// Creates a new range containing a single thing.
    pub fn single(x: T) -> Range<T> {
        Range::new(x, x)
    }

    /// Tests whether a given element belongs to this range.
    pub fn contains(&self, x: T) -> bool {
        self.start <= x && x <= self.end
    }

    /// Checks whether the intersections overlap.
    pub fn intersects(&self, other: &Self) -> bool {
        self.start <= other.end && self.end >= other.start
    }

    /// Computes the intersection between two ranges. Returns none if the intersection is empty.
    pub fn intersection(&self, other: &Self) -> Option<Self> {
        if self.intersects(other) {
            Some(Range::new(max(self.start, other.start), min(self.end, other.end)))
        } else {
            None
        }
    }

    /// Returns the smallest range that covers `self` and `other`.
    pub fn cover(&self, other: &Self) -> Self {
        Range::new(min(self.start, other.start), max(self.end, other.end))
    }
}

impl<T: PrimInt> PartialEq<T> for Range<T> {
    fn eq(&self, x: &T) -> bool {
        self.contains(*x)
    }
}

impl<T: PrimInt> PartialOrd<T> for Range<T> {
    fn partial_cmp(&self, x: &T) -> Option<Ordering> {
        if self.end < *x {
            Some(Ordering::Less)
        } else if self.start > *x {
            Some(Ordering::Greater)
        } else {
            Some(Ordering::Equal)
        }
    }
}

/// A set of characters. Optionally, each character in the set may be associated with some data.
#[derive(Clone, Eq, Hash, PartialEq)]
pub struct RangeMap<T, V> {
    elts: Vec<(Range<T>, V)>,
}

impl<T: Debug, V: Debug> Debug for RangeMap<T, V> {
    // When alternate formatting is specified, only prints out the first buch of mappings.
    fn fmt(&self, f: &mut Formatter) -> Result<(), std::fmt::Error> {
        try!(f.write_fmt(format_args!("RangeMap (")));

        if f.alternate() {
            try!(f.debug_map()
                 .entries(self.elts.iter().map(|x| (&x.0, &x.1)).take(DISPLAY_LIMIT))
                 .finish());
            if self.elts.len() > DISPLAY_LIMIT {
                try!(f.write_str("..."));
            }
        } else {
            try!(f.debug_map()
                 .entries(self.elts.iter().map(|x| (&x.0, &x.1)))
                 .finish());
        }
        try!(f.write_str(")"));
        Ok(())
    }
}

impl<T: Debug + PrimInt, V: Clone + Debug + Eq> FromIterator<(Range<T>, V)> for RangeMap<T, V> {
    /// Builds a `RangeMap` from an iterator over pairs. If any ranges overlap, they must map to
    /// the same value.
    ///
    /// # Panics
    ///  - if there are ranges that overlap and do not map to the same value.
    fn from_iter<I: IntoIterator<Item=(Range<T>, V)>>(iter: I) -> Self {
        let mut vec: Vec<_> = iter.into_iter().collect();
        vec.sort_by(|x, y| x.0.cmp(&y.0));

        RangeMap::from_sorted_vec(vec)
    }
}

impl<T: Debug + PrimInt, V: Clone + Debug + Eq> RangeMap<T, V> {
    /// Creates a new empty `RangeMap`.
    pub fn new() -> RangeMap<T, V> {
        RangeMap {
            elts: Vec::new(),
        }
    }

    // Creates a `RangeMap` from a `Vec`, which must contain ranges in ascending order. If any
    // ranges overlap, they must map to the same value.
    //
    // Panics if the ranges are not sorted, or if they overlap without mapping to the same value.
    fn from_sorted_vec(vec: Vec<(Range<T>, V)>) -> RangeMap<T, V> {
        let mut ret = RangeMap { elts: vec };
        ret.normalize();
        ret
    }

    // Creates a RangeMap from a Vec, which must be sorted and normalized.
    //
    // Panics unless `vec` is sorted and normalized.
    fn from_norm_vec(vec: Vec<(Range<T>, V)>) -> RangeMap<T, V> {
        for i in 1..vec.len() {
            if vec[i].0.start <= vec[i-1].0.end {
                panic!("vector {:?} has overlapping ranges {:?} and {:?}", vec, vec[i-1], vec[i]);
            }
            // If vec[i-1].0.end is T::max_value() then we've already panicked, so the unwrap is
            // safe.
            if vec[i].0.start == vec[i-1].0.end.checked_add(&T::one()).unwrap()
                    && vec[i].1 == vec[i-1].1 {
                panic!("vector {:?} has adjacent ranges with same value {:?} and {:?}",
                    vec, vec[i-1], vec[i]);
            }
        }

        RangeMap { elts: vec }
    }

    /// Returns the number of mapped ranges.
    ///
    /// Note that this is not usually the same as the number of mapped values.
    pub fn num_ranges(&self) -> usize {
        self.elts.len()
    }

    /// Tests whether this map is empty.
    pub fn is_empty(&self) -> bool {
        self.elts.is_empty()
    }

    /// Tests whether this `CharMap` maps every value.
    pub fn is_full(&self) -> bool {
        let mut last_end = T::min_value();
        for &(range, _) in &self.elts {
            if range.start > last_end {
                return false;
            }
            last_end = range.end;
        }
        last_end == T::max_value()
    }

    /// Iterates over all the mapped ranges and values.
    pub fn ranges_values<'a>(&'a self) -> std::slice::Iter<'a, (Range<T>, V)> {
        self.elts.iter()
    }

    /// Iterates over all mappings.
    pub fn keys_values<'a> (&'a self) -> PairIter<'a, T, V> {
        PairIter {
            map: self,
            next_range_idx: if self.is_empty() { None } else { Some(0) },
            next_key: if self.is_empty() { T::min_value() } else { self.elts[0].0.start },
        }
    }

    /// Finds the value that `x` maps to, if it exists.
    ///
    /// Runs in `O(log n)` time, where `n` is the number of mapped ranges.
    pub fn get(&self, x: T) -> Option<&V> {
        self.elts
            // The unwrap is ok because Range<T>::partial_cmp(&T) never returns None.
            .binary_search_by(|r| r.0.partial_cmp(&x).unwrap())
            .ok()
            .map(|idx| &self.elts[idx].1)
    }

    // Minimizes the number of ranges in this map.
    //
    // If there are any overlapping ranges that map to the same data, merges them. Assumes that the
    // ranges are sorted according to their start.
    //
    // Panics if there are overlapping ranges that map to different values.
    fn normalize(&mut self) {
        let mut vec = Vec::with_capacity(self.elts.len());
        mem::swap(&mut vec, &mut self.elts);

        for (range, val) in vec.into_iter() {
            if let Some(&mut (ref mut last_range, ref last_val)) = self.elts.last_mut() {
                if range.start <= last_range.end && &val != last_val {
                    panic!("overlapping ranges {:?} and {:?} map to values {:?} and {:?}",
                           last_range, range, last_val, val);
                }

                if range.start <= last_range.end.saturating_add(T::one()) && &val == last_val {
                    last_range.end = max(range.end, last_range.end);
                    continue;
                }
            }

            self.elts.push((range, val));
        }
    }

    /// Returns those mappings whose keys belong to the given set.
    pub fn intersection(&self, other: &RangeSet<T>) -> RangeMap<T, V> {
        let mut ret = Vec::new();
        let mut other_iter = other.map.elts.iter().peekable();

        for &(ref r, ref data) in &self.elts {
            while let Some(&&(ref s, _)) = other_iter.peek() {
                if let Some(int) = s.intersection(r) {
                    ret.push((int, data.clone()));
                }

                if s.end >= r.end {
                    break;
                } else {
                    other_iter.next();
                }
            }
        }

        RangeMap::from_sorted_vec(ret)
    }

    /// Counts the number of mapped keys.
    ///
    /// This saturates at `usize::MAX`.
    pub fn num_keys(&self) -> usize {
        self.ranges_values().fold(0, |acc, range| {
            acc.saturating_add((range.0.end - range.0.start).to_usize().unwrap_or(usize::MAX))
               .saturating_add(1)
        })
    }

    /// Returns the set of mapped chars, forgetting what they are mapped to.
    pub fn to_range_set(&self) -> RangeSet<T> {
        RangeSet::from_sorted_vec(self.elts.iter().map(|x| (x.0, ())).collect())
    }

    /// Modifies the values in place.
    pub fn map_values<F>(&mut self, mut f: F) where F: FnMut(&V) -> V {
        for &mut (_, ref mut data) in &mut self.elts {
            *data = f(data);
        }
    }

    /// Modifies this map to contain only those mappings with values `v` satisfying `f(v)`.
    pub fn retain_values<F>(&mut self, mut f: F) where F: FnMut(&V) -> bool {
        self.elts.retain(|x| f(&x.1));
    }
}

#[derive(Copy, Clone, Debug)]
pub struct PairIter<'a, T: 'a, V: 'a> {
    map: &'a RangeMap<T, V>,
    next_range_idx: Option<usize>,
    next_key: T,
}

impl<'a, T: PrimInt, V> Iterator for PairIter<'a, T, V> {
    type Item = (T, &'a V);
    fn next(&mut self) -> Option<Self::Item> {
        if let Some(idx) = self.next_range_idx {
            let ret = (self.next_key, &self.map.elts[idx].1);

            if self.next_key < self.map.elts[idx].0.end {
                self.next_key = self.next_key + T::one();
            } else if idx < self.map.elts.len() - 1 {
                self.next_range_idx = Some(idx + 1);
                self.next_key = self.map.elts[idx + 1].0.start;
            } else {
                self.next_range_idx = None;
            }

            Some(ret)
        } else {
            None
        }
    }
}

/// A set of integers, implemented as a sorted list of (inclusive) ranges.
#[derive(Clone, Eq, Hash, PartialEq)]
pub struct RangeSet<T> {
    map: RangeMap<T, ()>,
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct RangeIter<'a, T: PrimInt + 'a> {
    set: &'a RangeSet<T>,
    next_idx: usize,
}

impl<'a, T: Debug + PrimInt> Iterator for RangeIter<'a, T> {
    type Item = Range<T>;
    fn next(&mut self) -> Option<Range<T>> {
        if self.next_idx < self.set.num_ranges() {
            let ret = Some(self.set.map.elts[self.next_idx].0);
            self.next_idx += 1;
            ret
        } else {
            None
        }
    }
}

#[derive(Copy, Clone, Debug)]
pub struct EltIter<'a, T: 'a + PrimInt> {
    set: &'a RangeSet<T>,
    next_range_idx: Option<usize>,
    next_elt: T,
}

impl<'a, T: Debug + PrimInt> Iterator for EltIter<'a, T> {
    type Item = T;
    fn next(&mut self) -> Option<T> {
        if let Some(idx) = self.next_range_idx {
            let ret = Some(self.next_elt);
            if self.next_elt >= self.set.map.elts[idx].0.end {
                if idx + 1 < self.set.num_ranges() {
                    self.next_range_idx = Some(idx + 1);
                    self.next_elt = self.set.map.elts[idx + 1].0.start;
                } else {
                    self.next_range_idx = None;
                }
            } else {
                self.next_elt = self.next_elt + T::one();
            }
            ret
        } else {
            None
        }
    }
}

impl<T: Debug + PrimInt> Debug for RangeSet<T> {
    // When alternate formatting is specified, only prints out the first buch of mappings.
    fn fmt(&self, f: &mut Formatter) -> Result<(), std::fmt::Error> {
        try!(f.write_fmt(format_args!("RangeSet (")));

        if f.alternate() {
            try!(f.debug_set().entries(self.ranges().take(DISPLAY_LIMIT)).finish());
            if self.num_ranges() > DISPLAY_LIMIT {
                try!(f.write_str("..."));
            }
        } else {
            try!(f.debug_set().entries(self.ranges()).finish());
        }
        try!(f.write_str(")"));
        Ok(())
    }
}

impl<T: Debug + PrimInt> FromIterator<Range<T>> for RangeSet<T> {
    /// Builds a `RangeSet` from an iterator over `Range`s.
    fn from_iter<I: IntoIterator<Item=Range<T>>>(iter: I) -> Self {
        RangeSet {
            map: iter.into_iter().map(|x| (x, ())).collect()
        }
    }
}

impl<T: Debug + PrimInt> RangeSet<T> {
    /// Creates a new empty `RangeSet`.
    pub fn new() -> RangeSet<T> {
        RangeSet { map: RangeMap::new() }
    }

    /// Tests if this set is empty.
    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }

    /// Tests whether this set contains every valid value of `T`.
    pub fn is_full(&self) -> bool {
        // We are assuming normalization here.
        self.num_ranges() == 1 && self.map.elts[0].0 == Range::full()
    }

    /// Returns the number of ranges used to represent this set.
    pub fn num_ranges(&self) -> usize {
        self.map.num_ranges()
    }

    /// Returns the number of elements in the set.
    ///
    /// This saturates at `usize::MAX`.
    pub fn num_elements(&self) -> usize {
        self.map.num_keys()
    }

    /// Returns an iterator over all ranges in this set.
    pub fn ranges<'a>(&'a self) -> RangeIter<'a, T> {
        RangeIter {
            set: self,
            next_idx: 0,
        }
    }

    /// Returns an iterator over all elements in this set.
    pub fn elements<'a>(&'a self) -> EltIter<'a, T> {
        if self.map.elts.is_empty() {
            EltIter { set: self, next_range_idx: None, next_elt: T::min_value() }
        } else {
            EltIter {
                set: self,
                next_range_idx: Some(0),
                next_elt: self.map.elts[0].0.start,
            }
        }
    }

    /// Checks if this set contains a value.
    pub fn contains(&self, val: T) -> bool {
        self.map.get(val).is_some()
    }

    // Creates a RangeSet from a vector. The vector must be sorted, but it does not need to be
    // normalized.
    fn from_sorted_vec(vec: Vec<(Range<T>, ())>) -> RangeSet<T> {
        RangeSet { map: RangeMap::from_sorted_vec(vec) }
    }

    // Creates a RangeSet from a vector. The vector must be normalized, in the sense that it should
    // contain no adjacent ranges.
    fn from_norm_vec(vec: Vec<(Range<T>, ())>) -> RangeSet<T> {
        RangeSet { map: RangeMap::from_norm_vec(vec) }
    }

    /// Returns the union between `self` and `other`.
    pub fn union(&self, other: &RangeSet<T>) -> RangeSet<T> {
        if self.is_empty() {
            return other.clone();
        } else if other.is_empty() {
            return self.clone();
        }

        let mut ret = Vec::with_capacity(self.map.elts.len() + other.map.elts.len());
        let mut it1 = self.map.elts.iter();
        let mut it2 = other.map.elts.iter();
        let mut r1 = it1.next();
        let mut r2 = it2.next();
        let mut cur_range: Option<Range<T>> = None;

        while r1.is_some() || r2.is_some() {
            let r1_start = if let Some(&(r, _)) = r1 { r.start } else { T::max_value() };
            let r2_start = if let Some(&(r, _)) = r2 { r.start } else { T::max_value() };
            if let Some(cur) = cur_range {
                if min(r1_start, r2_start) > cur.end.saturating_add(T::one()) {
                    ret.push((cur_range.unwrap(), ()));
                    cur_range = None;
                }
            }

            let cover = |cur: &mut Option<Range<T>>, next: &Range<T>| {
                if let &mut Some(ref mut r) = cur {
                    *r = r.cover(next);
                } else {
                    *cur = Some(*next);
                }
            };

            if r1_start < r2_start || r2.is_none() {
                cover(&mut cur_range, &r1.unwrap().0);
                r1 = it1.next();
            } else {
                cover(&mut cur_range, &r2.unwrap().0);
                r2 = it2.next();
            }
        }

        if cur_range.is_some() {
            ret.push((cur_range.unwrap(), ()));
        }

        RangeSet::from_norm_vec(ret)
    }

    /// Creates a set that contains every value of `T`.
    pub fn full() -> RangeSet<T> {
        RangeSet::from_norm_vec(vec![(Range::full(), ())])
    }

    /// Creates a set containing a single element.
    pub fn single(x: T) -> RangeSet<T> {
        RangeSet::from_norm_vec(vec![(Range::single(x), ())])
    }

    /// Creates a set containing all elements except the given ones.
    ///
    /// # Panics
    ///  - if `chars` is not sorted or not unique.
    pub fn except<I: Iterator<Item=T>>(it: I) -> RangeSet<T> {
        let mut ret = Vec::new();
        let mut next_allowed = T::min_value();
        let mut last_forbidden = T::max_value();

        for i in it {
            if i > next_allowed {
                ret.push((Range::new(next_allowed, i - T::one()), ()));
            } else if i < next_allowed.saturating_sub(T::one()) {
                panic!("input to RangeSet::except must be sorted");
            }

            last_forbidden = i;
            next_allowed = i.saturating_add(T::one());
        }

        if last_forbidden < T::max_value() {
            ret.push((Range::new(last_forbidden + T::one(), T::max_value()), ()));
        }
        RangeSet::from_norm_vec(ret)
    }

    /// Finds the intersection between this set and `other`.
    pub fn intersection(&self, other: &RangeSet<T>) -> RangeSet<T> {
        RangeSet { map: self.map.intersection(other) }
    }

    /// Returns the set of all characters that are not in this set.
    pub fn negated(&self) -> RangeSet<T> {
        let mut ret = Vec::with_capacity(self.num_ranges() + 1);
        let mut last_end = T::max_value();

        for range in self.ranges() {
            if range.start > last_end {
                ret.push((Range::new(last_end, range.start - T::one()), ()));
            }
            last_end = range.end.saturating_add(T::one());
        }
        if last_end < T::max_value() {
            ret.push((Range::new(last_end, T::max_value()), ()));
        }

        RangeSet::from_norm_vec(ret)
    }
}

/// A multi-valued mapping from primitive integers to other data.
#[derive(Clone, Eq, Hash, PartialEq)]
pub struct RangeMultiMap<T, V> {
    elts: Vec<(Range<T>, V)>,
}

impl<T: Debug + PrimInt, V: Clone + Debug + PartialEq> Debug for RangeMultiMap<T, V> {
    fn fmt(&self, f: &mut Formatter) -> Result<(), std::fmt::Error> {
        try!(f.write_fmt(format_args!("RangeMultiMap (")));

        if f.alternate() {
            try!(f.debug_map()
                .entries(self.ranges_values().map(|x| (&x.0, &x.1)).take(DISPLAY_LIMIT))
                .finish());
            if self.num_ranges() > DISPLAY_LIMIT {
                try!(f.write_str("..."));
            }
        } else {
            try!(f.debug_set()
                .entries(self.ranges_values().map(|x| (&x.0, &x.1)))
                .finish());
        }
        try!(f.write_str(")"));
        Ok(())
    }
}

impl<T: Debug + PrimInt, V: Clone + Debug + PartialEq> RangeMultiMap<T, V> {
    /// Creates a new empty map.
    pub fn new() -> RangeMultiMap<T, V> {
        RangeMultiMap { elts: Vec::new() }
    }

    /// Returns the number of mapped ranges.
    pub fn num_ranges(&self) -> usize {
        self.elts.len()
    }

    /// Adds a new mapping from a range of characters to `value`.
    pub fn insert(&mut self, range: Range<T>, value: V) {
        self.elts.push((range, value));
    }

    /// Creates a map from a vector of pairs.
    pub fn from_vec(vec: Vec<(Range<T>, V)>) -> RangeMultiMap<T, V> {
        RangeMultiMap { elts: vec }
    }

    /// Returns a new `RangeMultiMap` containing only the mappings for keys that belong to the
    /// given set.
    pub fn intersection(&self, other: &RangeSet<T>) -> RangeMultiMap<T, V> {
        let mut ret = Vec::new();
        for &(ref my_range, ref data) in &self.elts {
            let start_idx = other.map.elts
                .binary_search_by(|r| r.0.end.cmp(&my_range.start))
                .unwrap_or_else(|x| x);
            for &(ref other_range, _) in &other.map.elts[start_idx..] {
                if my_range.start > other_range.end {
                    break;
                } else if let Some(r) = my_range.intersection(other_range) {
                    ret.push((r, data.clone()));
                }
            }
        }
        RangeMultiMap::from_vec(ret)
    }

    /// Returns a new `RangeMultiMap`, containing only those mappings with values `v` satisfying
    /// `f(v)`.
    pub fn filter_values<F>(&self, mut f: F) -> RangeMultiMap<T, V> where F: FnMut(&V) -> bool {
        RangeMultiMap::from_vec(self.elts.iter().filter(|x| f(&x.1)).cloned().collect())
    }

    /// Splits the set of ranges into equal or disjoint ranges.
    ///
    /// The output is a `RangeMultiMap` in which every pair of `Range`s are either identical or
    /// disjoint.
    fn split(&self) -> RangeMultiMap<T, V> {
        let mut ret = RangeMultiMap::new();
        let mut start_chars = Vec::new();

        for &(ref range, _) in self.elts.iter() {
            start_chars.push(range.start);
            if range.end < T::max_value() {
                start_chars.push(range.end + T::one());
            }
        }

        start_chars.sort();
        start_chars.dedup();

        for &(range, ref state) in self.elts.iter() {
            let mut idx = match start_chars.binary_search(&range.start) {
                Ok(i) => i+1,
                Err(i) => i,
            };
            let mut last = range.start;
            loop {
                if idx >= start_chars.len() || start_chars[idx] > range.end {
                    ret.elts.push((Range::new(last, range.end), state.clone()));
                    break;
                } else {
                    ret.elts.push((Range::new(last, start_chars[idx] - T::one()), state.clone()));
                    last = start_chars[idx];
                    idx += 1;
                }
            }
        }

        ret
    }

    /// Returns the underlying `Vec`.
    pub fn into_vec(self) -> Vec<(Range<T>, V)> {
        self.elts
    }

    /// Iterates over all the mapped ranges and values.
    pub fn ranges_values<'a>(&'a self) -> std::slice::Iter<'a, (Range<T>, V)> {
        self.elts.iter()
    }
}


impl<T: Debug + PrimInt, V: Clone + Debug + Ord> RangeMultiMap<T, V> {
    /// Makes the ranges sorted and non-overlapping. The data associated with each range will
    /// be a sorted `Vec<T>` instead of a single `T`.
    pub fn group(&self) -> RangeMap<T, Vec<V>> {
        let mut split = self.split().elts;
        split.sort();
        split.dedup();

        let mut vec: Vec<(Range<T>, Vec<V>)> = Vec::new();
        let grouped = split.into_iter().group_by_lazy(|&(range, _)| range);
        for (range, pairs) in grouped.into_iter() {
            let val_vec: Vec<_> = pairs.map(|x| x.1).collect();
            vec.push((range, val_vec));
        }
        RangeMap::from_sorted_vec(vec)
    }
}

#[cfg(test)]
mod tests {
    use num::traits::PrimInt;
    use std::cmp::{max, min};
    use std::fmt::Debug;
    use std::ops::Add;
    use super::*;
    use quickcheck::{Arbitrary, Gen, TestResult};

    impl<T: Arbitrary + Debug + PrimInt> Arbitrary for Range<T> {
        fn arbitrary<G: Gen>(g: &mut G) -> Self {
            let a = T::arbitrary(g);
            let b = T::arbitrary(g);
            Range::new(min(a, b), max(a, b))
        }
    }

    impl<T> Arbitrary for RangeMultiMap<T, i32>
    where T: Arbitrary + Debug + PrimInt {
        fn arbitrary<G: Gen>(g: &mut G) -> Self {
            RangeMultiMap::from_vec(Vec::arbitrary(g))
        }

        fn shrink(&self) -> Box<Iterator<Item=Self>> {
            Box::new(self.elts.shrink().map(|v| RangeMultiMap::from_vec(v)))
        }
    }

    impl<T> Arbitrary for RangeMap<T, i32>
    where T: Arbitrary + Debug + PrimInt {
        fn arbitrary<G: Gen>(g: &mut G) -> Self {
            let map: RangeMap<T, Vec<_>> = RangeMultiMap::arbitrary(g).group();
            // TODO: replace fold with sum once it's stable
            map.ranges_values().map(|x| (x.0, x.1.iter().fold(0, Add::add))).collect()
        }

        fn shrink(&self) -> Box<Iterator<Item=Self>> {
            Box::new(self.elts.shrink().map(|v| RangeMap::from_norm_vec(v)))
        }
    }

    impl<T: Arbitrary + Debug + PrimInt> Arbitrary for RangeSet<T> {
        fn arbitrary<G: Gen>(g: &mut G) -> Self {
            RangeMap::arbitrary(g).to_range_set()
        }

        fn shrink(&self) -> Box<Iterator<Item=Self>> {
            Box::new(self.map.elts.shrink().map(|v| RangeSet::from_norm_vec(v)))
        }
    }

    #[quickcheck]
    fn range_intersects_intersection(r1: Range<i32>, r2: Range<i32>) -> bool {
        r1.intersection(&r2).is_some() == r1.intersects(&r2)
    }

    #[quickcheck]
    fn range_intersection_contains(r1: Range<i32>, r2: Range<i32>, x: i32) -> TestResult {
        if let Some(r) = r1.intersection(&r2) {
            TestResult::from_bool(r.contains(x) == (r1.contains(x) && r2.contains(x)))
        } else {
            TestResult::discard()
        }
    }

    #[test]
    #[should_panic]
    fn range_backwards() {
        map(vec![(5, 1, 1), (6, 10, 2)]);
    }

    #[quickcheck]
    fn range_intersection_cover(r1: Range<i32>, r2: Range<i32>) -> bool {
        r1 == r1.cover(&r2).intersection(&r1).unwrap()
    }

    fn map(vec: Vec<(i32, i32, i32)>) -> RangeMap<i32, i32> {
        vec.into_iter()
            .map(|(a, b, c)| (Range::new(a, b), c))
            .collect()
    }

    #[test]
    fn rangemap_overlapping() {
        assert_eq!(map(vec![(1, 5, 1), (2, 10, 1)]), map(vec![(1, 10, 1)]));
        assert_eq!(map(vec![(1, 5, 1), (2, 10, 1), (9, 11, 1)]), map(vec![(1, 11, 1)]));
        map(vec![(1, 5, 1), (6, 10, 2)]);
    }

    #[test]
    #[should_panic]
    fn rangemap_overlapping_nonequal() {
        map(vec![(1, 5, 1), (5, 10, 2)]);
    }

    #[quickcheck]
    fn rangemap_intersection(map: RangeMap<i32, i32>, set: RangeSet<i32>) -> bool {
        let int = map.intersection(&set);
        set.elements().all(|x| map.get(x) == int.get(x))
            && int.keys_values().all(|x| set.contains(x.0))
    }

    #[quickcheck]
    fn rangemap_num_ranges(map: RangeMap<i32, i32>) -> bool {
        map.num_ranges() == map.ranges_values().count()
    }

    #[quickcheck]
    fn rangemap_num_keys(map: RangeMap<i32, i32>) -> bool {
        map.num_keys() == map.keys_values().count()
    }

    #[quickcheck]
    fn rangemap_map_values(map: RangeMap<i32, i32>, x: i32) -> bool {
        let mut new_map = map.clone();
        new_map.map_values(|y| x + y);
        new_map.keys_values().all(|(k, v)| map.get(k).unwrap() + x == *v)
            && map.keys_values().all(|(k, v)| new_map.get(k).unwrap() - x == *v)
    }

    #[quickcheck]
    fn rangemap_retain_values(map: RangeMap<i32, i32>, r: Range<i32>) -> bool {
        let mut new_map = map.clone();
        new_map.retain_values(|v| r.contains(*v));
        new_map.keys_values().all(|(_, v)| r.contains(*v))
            && map.keys_values().all(|(k, v)| !r.contains(*v) || new_map.get(k).unwrap() == v)
    }

    #[quickcheck]
    fn rangeset_contains(set: RangeSet<i32>) -> bool {
        set.elements().all(|e| set.contains(e))
    }

    #[quickcheck]
    fn rangeset_num_ranges(set: RangeSet<i32>) -> bool {
        set.num_ranges() == set.ranges().count()
    }

    #[quickcheck]
    fn rangeset_num_elements(set: RangeSet<i32>) -> bool {
        set.num_elements() == set.elements().count()
    }

    #[quickcheck]
    fn rangeset_union(s1: RangeSet<i32>, s2: RangeSet<i32>) -> bool {
        let un = s1.union(&s2);
        un.elements().all(|e| s1.contains(e) || s2.contains(e))
            && s1.elements().all(|e| un.contains(e))
            && s2.elements().all(|e| un.contains(e))
    }

    #[quickcheck]
    fn rangeset_intersection(s1: RangeSet<i32>, s2: RangeSet<i32>) -> bool {
        let int = s1.intersection(&s2);
        int.elements().all(|e| s1.contains(e) && s2.contains(e))
            && s1.elements().all(|e| int.contains(e) == s2.contains(e))
    }

    #[quickcheck]
    fn rangeset_negate(set: RangeSet<i8>) -> bool {
        let neg = set.negated();
        neg.elements().all(|e| !set.contains(e))
            && set.elements().all(|e| !neg.contains(e))
    }

    #[quickcheck]
    fn rangeset_except(mut except: Vec<i8>) -> bool {
        except.sort();
        let set = RangeSet::except(except.iter().cloned());
        set.elements().all(|e| except.binary_search(&e).is_err())
            && except.iter().all(|&e| !set.contains(e))
    }

    #[test]
    #[should_panic]
    fn rangeset_except_unsorted() {
        RangeSet::except([1i32, 3, 2].into_iter());
    }

    // TODO: test RangeMultiMap
}

