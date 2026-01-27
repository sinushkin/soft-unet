use std::collections::HashMap;
use serde::{Deserialize};

pub const OUTER_LABEL: &str = "outer";
pub const INNER_LABEL: &str = "inner";
#[derive(Debug, Clone, Deserialize)]
pub struct Configuration {
    pub label_map: HashMap<String, u8>,
    //inner - 255, outside outer - 0, near outside border 35 (this one)
    pub gradient_alpha: u8
}


pub fn load_configuration() -> Configuration {
    let mut label_map: HashMap<String, u8> = HashMap::new();
    label_map.insert("bag".to_string(), 5);
    label_map.insert("bug".to_string(), 5);
    //outer - gradient

    Configuration {
        label_map,
        gradient_alpha: 35
    }
}