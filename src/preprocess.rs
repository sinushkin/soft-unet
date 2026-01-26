use opencv::core::Mat;
use crate::types::{ContourGroup, Image, RectSize};
use anyhow::Result;
use crate::spline::smooth_polygon;

pub fn preprocess_spline_contours(contour_group: &mut ContourGroup) -> Result<()> {
    contour_group.outer.points = smooth_polygon(&contour_group.outer.points, 10);
    for inner in &mut contour_group.inners {
        inner.points = smooth_polygon(&inner.points, 10);   //TODO configureadable
    }
    Ok(())
}