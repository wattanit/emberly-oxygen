//! The key-value cache (Spec §3): per layer, one growing `Vec<f32>` for keys
//! and one for values, each laid out `[position][kv_heads × head_dim]`.
//! Appending a position is one extend, and reading position `t` is one
//! contiguous slice, so attention walks the cache without any gathering.

// The forward pass (task 4.5) is the first caller of the cache, and removes this.
#![allow(dead_code)]

use crate::Error;

pub(crate) struct KvCache {
    keys: Vec<Vec<f32>>,
    values: Vec<Vec<f32>>,
    /// `kv_heads × head_dim`: the values stored per position.
    width: usize,
    /// The model's context length: no layer holds more positions than this.
    max_positions: usize,
}

impl KvCache {
    /// An empty cache. Nothing is reserved up front: at the full context a
    /// 30-layer SmolLM2 cache is hundreds of megabytes, and most runs are short.
    pub fn new(layers: usize, kv_heads: usize, head_dim: usize, max_positions: usize) -> KvCache {
        KvCache {
            keys: vec![Vec::new(); layers],
            values: vec![Vec::new(); layers],
            width: kv_heads * head_dim,
            max_positions,
        }
    }

    /// Positions stored so far. Every layer holds the same number between steps.
    pub fn len(&self) -> usize {
        self.keys.first().map_or(0, |keys| keys.len() / self.width.max(1))
    }

    /// Whether one more position fits. The forward pass asks before it touches
    /// any layer, so a step that does not fit leaves the cache as it was.
    pub fn ensure_room(&self) -> Result<(), Error> {
        if self.len() >= self.max_positions {
            return Err(self.full());
        }
        Ok(())
    }

    /// Appends one position's key and value to a layer.
    /// Panics if either has the wrong width or the layer does not exist: the
    /// engine builds both from the configuration, so that is an internal bug.
    pub fn append(&mut self, layer: usize, key: &[f32], value: &[f32]) -> Result<(), Error> {
        assert_eq!(key.len(), self.width, "a key holds kv_heads × head_dim values");
        assert_eq!(value.len(), self.width, "a value holds kv_heads × head_dim values");
        if self.keys[layer].len() / self.width.max(1) >= self.max_positions {
            return Err(self.full());
        }
        self.keys[layer].extend_from_slice(key);
        self.values[layer].extend_from_slice(value);
        Ok(())
    }

    /// The key of one position: `kv_heads × head_dim` values, head after head.
    pub fn key(&self, layer: usize, position: usize) -> &[f32] {
        Self::at(&self.keys[layer], self.width, position)
    }

    /// The value of one position, laid out like a key.
    pub fn value(&self, layer: usize, position: usize) -> &[f32] {
        Self::at(&self.values[layer], self.width, position)
    }

    fn at(data: &[f32], width: usize, position: usize) -> &[f32] {
        assert!(
            (position + 1) * width <= data.len(),
            "position {position} is not in the cache ({} positions)",
            data.len() / width.max(1)
        );
        &data[position * width..(position + 1) * width]
    }

    fn full(&self) -> Error {
        Error::Rejected(format!(
            "the context is full: the model holds at most {} positions (FR-18)",
            self.max_positions
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A key whose values say where they belong: layer, position, then index.
    fn entry(layer: usize, position: usize, width: usize, offset: f32) -> Vec<f32> {
        (0..width).map(|i| offset + (layer * 1000 + position * 10 + i) as f32).collect()
    }

    // Spec §3: positions come back in the order they went in, in the [position][kv_heads × head_dim] layout.
    #[test]
    fn positions_read_back_in_order() {
        let (kv_heads, head_dim) = (3, 4);
        let width = kv_heads * head_dim;
        let mut cache = KvCache::new(2, kv_heads, head_dim, 16);
        assert_eq!(cache.len(), 0);
        for position in 0..5 {
            cache.append(0, &entry(0, position, width, 0.0), &entry(0, position, width, 0.5)).unwrap();
            cache.append(1, &entry(1, position, width, 0.0), &entry(1, position, width, 0.5)).unwrap();
            assert_eq!(cache.len(), position + 1);
        }
        for position in 0..5 {
            assert_eq!(cache.key(0, position), entry(0, position, width, 0.0));
            assert_eq!(cache.value(1, position), entry(1, position, width, 0.5));
        }
    }

    // Spec §3: within a position the heads sit one after another, each `head_dim` long.
    #[test]
    fn heads_are_contiguous_within_a_position() {
        let mut cache = KvCache::new(1, 3, 2, 8);
        cache.append(0, &[0., 1., 10., 11., 20., 21.], &[0.; 6]).unwrap();
        cache.append(0, &[100., 101., 110., 111., 120., 121.], &[0.; 6]).unwrap();
        let key = cache.key(0, 1);
        assert_eq!(&key[2..4], &[110., 111.], "head 1 of position 1");
    }

    // Spec §3: layers are separate, and keys and values do not mix.
    #[test]
    fn layers_keys_and_values_are_separate() {
        let mut cache = KvCache::new(3, 1, 2, 8);
        cache.append(1, &[1., 2.], &[3., 4.]).unwrap();
        cache.append(2, &[5., 6.], &[7., 8.]).unwrap();
        assert_eq!((cache.key(1, 0), cache.value(1, 0)), (&[1., 2.][..], &[3., 4.][..]));
        assert_eq!((cache.key(2, 0), cache.value(2, 0)), (&[5., 6.][..], &[7., 8.][..]));
    }

    // Spec §14, FR-18: the context limit is a rejected request, and a refusal changes nothing.
    #[test]
    fn the_context_limit_is_rejected_not_a_panic() {
        let mut cache = KvCache::new(2, 1, 2, 3);
        for position in 0..3 {
            let x = [position as f32; 2];
            cache.append(0, &x, &x).unwrap();
            cache.append(1, &x, &x).unwrap();
            assert!(position == 2 || cache.ensure_room().is_ok());
        }
        let err = cache.ensure_room().unwrap_err();
        assert!(matches!(err, Error::Rejected(_)) && err.to_string().contains('3'), "{err}");
        assert!(matches!(cache.append(0, &[9., 9.], &[9., 9.]), Err(Error::Rejected(_))));
        assert_eq!(cache.len(), 3);
        for position in 0..3 {
            assert_eq!(cache.key(0, position), [position as f32; 2], "stored data is untouched");
        }
    }

    // FR-18: a cache with room says so.
    #[test]
    fn an_empty_cache_has_room_even_with_no_layers() {
        assert!(KvCache::new(2, 1, 2, 1).ensure_room().is_ok());
        let none = KvCache::new(0, 1, 2, 1);
        assert_eq!(none.len(), 0);
    }

    // Spec §14: the wrong width, a missing layer or an unstored position is an internal bug.
    #[test]
    #[should_panic(expected = "a key holds")]
    fn a_key_of_the_wrong_width_panics() {
        KvCache::new(1, 2, 2, 4).append(0, &[0.; 3], &[0.; 4]).unwrap();
    }

    #[test]
    #[should_panic(expected = "a value holds")]
    fn a_value_of_the_wrong_width_panics() {
        KvCache::new(1, 2, 2, 4).append(0, &[0.; 4], &[0.; 5]).unwrap();
    }

    #[test]
    #[should_panic(expected = "not in the cache")]
    fn reading_an_unstored_position_panics() {
        let mut cache = KvCache::new(1, 1, 2, 4);
        cache.append(0, &[1., 2.], &[3., 4.]).unwrap();
        cache.key(0, 1);
    }
}
