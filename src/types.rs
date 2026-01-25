use opencv::{
    core::{Mat, Point2f}
};
use opencv::core::Point;
use serde::{Deserialize, Serialize};

#[derive(Debug, Copy, Clone, Serialize, Deserialize)]
pub struct SuPoint {
    pub x: f32,
    pub y: f32,
}

impl SuPoint {
    pub(crate) fn new(x: i32, y: i32) -> Self {
        SuPoint{ x: x as f32, y: y as f32}
    }
}

pub struct Contour {
    pub label: String,
    pub points: Vec<SuPoint>,
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

impl From<SuPoint> for Point {
    fn from(value: SuPoint) -> Self {
        Point::new(value.x.round() as i32, value.y.round() as i32)
    }
}
impl From<SuPoint> for Point2f {
    fn from(value: SuPoint) -> Self {
        Point2f::new(value.x, value.y)
    }
}

