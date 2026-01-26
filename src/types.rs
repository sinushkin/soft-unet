use opencv::{
    core::{Mat, Point2f}
};
use opencv::core::{Point, Vector};
use serde::{Deserialize, Serialize};

pub const OUTER_LABEL: &str = "outer";
pub const INNER_LABEL: &str = "inner";

pub struct Image {
    pub original: Mat,
    pub size: RectSize,
    pub contour_crops: Vec<ContourCrop>,
}

pub struct ContourCrop {
    pub offset: SuPoint,
    pub size: RectSize,
    pub contour_group: ContourGroup
}

#[derive(Debug, Copy, Clone)]
pub struct RectSize {
    pub width: usize,
    pub height: usize,
}

pub struct Contour {
    pub label: String,
    pub points: Vec<SuPoint2F>,
}

pub struct ContourGroup {
    pub idx: usize,
    pub outer: Contour,
    pub inners: Vec<Contour>,
}

#[derive(Debug, Copy, Clone)]
pub struct SuPoint {
    pub x: usize,
    pub y: usize,
}

#[derive(Debug, Copy, Clone, Serialize, Deserialize)]
pub struct SuPoint2F {
    pub x: f32,
    pub y: f32,
}

impl SuPoint2F {
    #[warn(unused)]
    pub(crate) fn new(x: i32, y: i32) -> Self {
        SuPoint2F { x: x as f32, y: y as f32}
    }
}

impl From<SuPoint2F> for Point {
    fn from(value: SuPoint2F) -> Self {
        Point::new(value.x.round() as i32, value.y.round() as i32)
    }
}
impl From<&SuPoint2F> for Point {
    fn from(value: &SuPoint2F) -> Self {
        Point::new(value.x.round() as i32, value.y.round() as i32)
    }
}

impl From<SuPoint2F> for Point2f {
    fn from(value: SuPoint2F) -> Self {
        Point2f::new(value.x, value.y)
    }
}
impl From<&SuPoint2F> for Point2f {
    fn from(value: &SuPoint2F) -> Self {
        Point2f::new(value.x, value.y)
    }
}

pub fn su_points_to_cv_points(pts: &[SuPoint2F]) -> Vector<Point> {
    let mut result = Vector::<Point>::new();
    pts.iter().for_each(|p| result.push(p.into()));
    result
}


impl RectSize {
    pub fn new(width: usize, height: usize) -> Self {
        Self{ width, height }
    }
    pub fn destructure(&self) -> (i32, i32) {
        (self.width as i32, self.height as i32)
    }
}

