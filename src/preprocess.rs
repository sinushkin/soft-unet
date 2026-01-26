use crate::spline::smooth_polygon;
use crate::types::{ContourGroup, Image};
use anyhow::Result;



//TODO аугментация
pub fn preprocess_image(image: &mut Image) -> Result<()> {
    image
        .contour_crops
        .iter_mut()
        .try_for_each(|contour_crop| -> Result<()> {
            preprocess_spline_contours(&mut contour_crop.contour_group)?;
            Ok(())
        })?;
    Ok(())
}

pub fn preprocess_spline_contours(contour_group: &mut ContourGroup) -> Result<()> {
    contour_group.outer.points = smooth_polygon(&contour_group.outer.points, 10);
    for inner in &mut contour_group.inners {
        inner.points = smooth_polygon(&inner.points, 10); //TODO configureadable
    }
    Ok(())
}
