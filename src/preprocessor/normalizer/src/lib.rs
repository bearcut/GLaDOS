use pyo3::prelude::*;

/// 1. THE STRUCT (Data Layout in Memory)
/// #[pyclass] tells PyO3 to generate a Python class wrapper around this Rust struct.
#[pyclass]
pub struct NormalizerEngine {
    entropy_threshold: f32,
}

/// 2. INTERNAL HELPER FUNCTION (Pure Rust - Not exposed to Python)
/// Notice `text: &str`. Just like `std::string_view` in C++, `&str` is a borrowed
/// read-only slice (pointer + length). It reads Python's string memory directly with ZERO copying!
fn calculate_entropy(text: &str) -> f32 {
    let bytes = text.as_bytes();
    let len = bytes.len();
    if len == 0 {
        return 0.0;
    }

    // Stack-allocated array of 256 unsigned 32-bit ints initialized to 0.
    // Why? Heap allocation (malloc/Vec) is slow. A fixed stack array takes <1 nanosecond.
    let mut counts = [0u32; 256];
    for &b in bytes {
        counts[b as usize] += 1;
    }

    let len_f = len as f32;
    let mut entropy = 0.0f32;

    for &count in counts.iter() {
        if count > 0 {
            let p = (count as f32) / len_f;
            entropy -= p * p.log2();
        }
    }
    entropy
}

/// 3. THE METHODS (Exposed to Python)
/// #[pymethods] exposes these functions as methods on the Python class.
#[pymethods]
impl NormalizerEngine {
    /// #[new] is the Python `__init__` constructor.
    #[new]
    pub fn new(entropy_threshold: f32) -> Self {
        NormalizerEngine { entropy_threshold }
    }

    /// Takes a borrowed string slice (`&str`) from Python and returns a new owned `String`.
    pub fn inspect_word(&self, word: &str) -> String {
        let h = calculate_entropy(word);

        if h >= self.entropy_threshold {
            // `format!` allocates a new heap String (like sprintf / std::format)
            format!("[HIGH_ENTROPY: {:.2}] {}", h, word)
        } else {
            // Convert the borrowed slice into an owned String to hand back to Python
            word.to_string()
        }
    }

    /// Directly expose the math function so you can test it from Python
    pub fn get_entropy(&self, word: &str) -> f32 {
        calculate_entropy(word)
    }
}

/// 4. MODULE REGISTRATION
/// The function name MUST match `name = "glados_normalizer"` in Cargo.toml.
/// This is what runs when Python executes `import glados_normalizer`.
#[pymodule]
fn normalizer(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<NormalizerEngine>()?;
    Ok(())
}
