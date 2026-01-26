use opencv::core::Mat;
use crate::types::{ContoursStruct, Image, RectSize};
use anyhow::Result;
use crate::spline::smooth_polygon;

pub fn preprocess_spline_contours(image: Image) -> Result<Image> {
    let mut image = image;
    image.contours.iter_mut().try_for_each(|contour| -> Result<()> {
        contour.outer.points = smooth_polygon(&contour.outer.points, 10);
        for inner in &mut contour.inners {
            inner.points = smooth_polygon(&inner.points, 10);   //TODO configureadable
        }
        Ok(())
    })?;
    Ok(image)
}