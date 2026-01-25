use opencv::prelude::*;
use serde::{Deserialize, Deserializer, Serialize};

#[derive(Debug, Copy, Clone, Serialize, Deserialize)]
pub struct Point {
    pub x: f32,
    pub y: f32,
}

pub struct Contour {
    pub label: String,
    pub points: Vec<Point>,
}

pub struct ContoursStruct {
    pub outer: Contour,
    pub inners: Vec<Contour>,
}

pub struct Image {
    pub original: Mat,
    pub width: usize,
    pub height: usize,
    pub contours: Vec<ContoursStruct>,
}
