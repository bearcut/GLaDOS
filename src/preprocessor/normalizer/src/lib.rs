use base64::{Engine as _, engine::general_purpose::STANDARD};
use pyo3::prelude::*;
use std::collections::HashMap;

const DELIMITERS: [u32; 16] = [
    9, 10, 13, 32, 34, 38, 40, 44, 46, 58, 59, 63, 91, 93, 123, 125,
];

#[pyclass]
pub struct NormalizerEngine {
    entropy_threshold: f32,
    map: HashMap<u32, u32>,
}

/// 2. INTERNAL HELPER FUNCTION

fn calculate_entropy(text: &str, map: &mut HashMap<u32, u32>) -> f32 {
    map.clear();

    let mut total_chars = 0usize;
    for c in text.chars() {
        *map.entry(c as u32).or_insert(0) += 1;
        total_chars += 1;
    }

    if total_chars == 0 {
        return 0.0;
    }

    let len_f = total_chars as f32;
    let mut entropy = 0.0f32;

    for &count in map.values() {
        let p = count as f32 / len_f;
        entropy -= p * p.log2();
    }
    entropy
}

fn binary_search(arr: &[u32; 16], c: u32) -> bool {
    if c > 125 {
        return false;
    }
    let mut low: usize = 0;
    let mut high: usize = 15;
    let mut mid: usize;
    while low <= high {
        mid = low + (high - low) / 2;
        if arr[mid] == c {
            return true;
        } else if arr[mid] < c {
            low = mid + 1;
        } else {
            if mid == 0 {
                break;
            }
            high = mid - 1;
        }
    }
    return false;
}

fn decode_base64(encoded: &str) -> Result<String, Box<dyn std::error::Error>> {
    let bytes = STANDARD.decode(encoded)?;
    let decoded_string = String::from_utf8(bytes)?;
    Ok(decoded_string)
}

/// 3. THE METHODS (Exposed to Python)
#[pymethods]
impl NormalizerEngine {
    #[new]
    pub fn new(entropy_threshold: f32) -> Self {
        let map = HashMap::<u32, u32>::new();
        NormalizerEngine {
            entropy_threshold,
            map,
        }
    }

    pub fn encode(&mut self, prompt: &str) -> String {
        let mut encoded_string = String::with_capacity(prompt.len());
        let mut buffer = String::new();
        for c in prompt.chars() {
            if binary_search(&DELIMITERS, c as u32) {
                let entropy = calculate_entropy(&buffer, &mut self.map);
                if entropy >= self.entropy_threshold {
                    match decode_base64(&buffer) {
                        Ok(decoded) => {
                            encoded_string.push_str(&format!(
                                "[RAW] {} [DECODED] {} [ENDDECODE]",
                                buffer, decoded
                            ));
                        }
                        Err(_) => {
                            encoded_string.push_str(&buffer);
                        }
                    }
                } else {
                    encoded_string.push_str(&buffer);
                }
                encoded_string.push(c);
                buffer.clear();
            } else {
                buffer.push(c);
            }
        }

        if !buffer.is_empty() {
            let entropy = calculate_entropy(&buffer, &mut self.map);
            if entropy >= self.entropy_threshold {
                match decode_base64(&buffer) {
                    Ok(decoded) => {
                        encoded_string.push_str(&format!("[RAW] {} [DECODED] {}", buffer, decoded));
                    }
                    Err(_) => {
                        encoded_string.push_str(&buffer);
                    }
                }
            } else {
                encoded_string.push_str(&buffer);
            }
        }

        encoded_string
    }

    pub fn flush(&mut self) {
        self.map.clear();
    }
}

/// 4. MODULE REGISTRATION
#[pymodule]
fn normalizer(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<NormalizerEngine>()?;
    Ok(())
}
