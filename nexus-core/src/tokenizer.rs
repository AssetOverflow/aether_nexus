//! Tokenizer wrapper for HuggingFace tokenizer.json files.
//!
//! Provides encode (text → token IDs) and decode (token IDs → text) using
//! the `tokenizers` crate, which supports BPE, WordPiece, Unigram, etc.

use tokenizers::Tokenizer as HfTokenizer;

/// Wrapper around the HuggingFace tokenizer.
pub struct Tokenizer {
    inner: HfTokenizer,
}

impl Tokenizer {
    /// Load a tokenizer from a directory containing `tokenizer.json`.
    pub fn from_dir(model_dir: &str) -> Result<Self, String> {
        let tokenizer_path = format!("{}/tokenizer.json", model_dir);
        let inner = HfTokenizer::from_file(&tokenizer_path)
            .map_err(|e| format!("Failed to load tokenizer from '{}': {}", tokenizer_path, e))?;
        Ok(Self { inner })
    }

    /// Load a tokenizer from a specific file path.
    pub fn from_file(path: &str) -> Result<Self, String> {
        let inner = HfTokenizer::from_file(path)
            .map_err(|e| format!("Failed to load tokenizer from '{}': {}", path, e))?;
        Ok(Self { inner })
    }

    /// Encode text into token IDs.
    pub fn encode(&self, text: &str) -> Result<Vec<u32>, String> {
        let encoding = self
            .inner
            .encode(text, false)
            .map_err(|e| format!("Encoding failed: {}", e))?;
        Ok(encoding.get_ids().to_vec())
    }

    /// Decode token IDs back to text.
    pub fn decode(&self, ids: &[u32]) -> Result<String, String> {
        self.inner
            .decode(ids, true)
            .map_err(|e| format!("Decoding failed: {}", e))
    }

    /// Get the vocabulary size.
    pub fn vocab_size(&self) -> usize {
        self.inner.get_vocab_size(true)
    }

    /// Encode text and return both IDs and tokens.
    pub fn encode_with_tokens(&self, text: &str) -> Result<(Vec<u32>, Vec<String>), String> {
        let encoding = self
            .inner
            .encode(text, false)
            .map_err(|e| format!("Encoding failed: {}", e))?;
        let ids = encoding.get_ids().to_vec();
        let tokens = encoding.get_tokens().to_vec();
        Ok((ids, tokens))
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tokenizer_roundtrip() {
        let model_dir = "models/inference/granite-2b-instruct";
        if !std::path::Path::new(&format!("{}/tokenizer.json", model_dir)).exists() {
            eprintln!(
                "Model not found at '{}' — skipping tokenizer test",
                model_dir
            );
            return;
        }

        let tok = Tokenizer::from_dir(model_dir).expect("Failed to load tokenizer");
        println!("Vocab size: {}", tok.vocab_size());
        assert!(tok.vocab_size() > 0);

        let text = "Hello, my name is";
        let ids = tok.encode(text).expect("Encoding failed");
        println!("Encoded '{}' → {:?}", text, ids);
        assert!(!ids.is_empty());

        let decoded = tok.decode(&ids).expect("Decoding failed");
        println!("Decoded → '{}'", decoded);
        assert!(decoded.contains("Hello"));
    }
}
