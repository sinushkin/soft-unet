/// https://www.youtube.com/watch?v=oxWfLTQoC5A
/// https://www.youtube.com/watch?v=_3S3eTvBEns
///

use anyhow::Result;
use opencv::core::{Point, Rect, Scalar, Vector, CV_8UC1};
use opencv::imgproc;
use opencv::prelude::{Mat, MatExprTraitConst};
use crate::types::{su_points_to_cv_points, Contour, ContoursStruct, Image, RectSize, INNER_LABEL};

fn build_outer_inner_masks(contour: &ContoursStruct, size: &RectSize) -> Result<(Mat, Mat)> {
    let (width, height) = size.destructure();
    let mut outer_mask = Mat::zeros(height, width, CV_8UC1)?.to_mat()?;
    let mut inner_mask = Mat::zeros(height, width, CV_8UC1)?.to_mat()?;

    let outer_pts = su_points_to_cv_points(&contour.outer.points);
    let mut outer_vec = Vector::<Vector<Point>>::new();
    outer_vec.push(outer_pts);
    imgproc::fill_poly(
        &mut outer_mask,
        &outer_vec,
        Scalar::all(255.0),
        imgproc::LINE_8,
        0,
        Point::new(0,0),
    )?;


    let inner_vec : Vector::<Vector<Point>> = contour.inners.iter()
        .filter(|inner| inner.label.starts_with(INNER_LABEL))
        .map(|inner| su_points_to_cv_points(&inner.points)).collect();
    imgproc::fill_poly(
        &mut inner_mask,
        &inner_vec,
        Scalar::all(255.0),
        imgproc::LINE_8,
        0,
        Point::new(0,0),
    )?;

    Ok((outer_mask, inner_mask))
}


#[cfg(test)]
mod tests {
    use std::fs::create_dir_all;
    use std::path::{Path, PathBuf};
    use opencv::{imgcodecs, imgproc};
    use opencv::prelude::{Mat, MatExprTraitConst};
    use anyhow::Result;
    use opencv::core::{Point, Scalar};
    use opencv::core::Vector;
    use crate::lablelme_loader::load_labelme;
    use crate::preprocess::preprocess_spline_contours;
    use super::*;

    #[test]
    #[ignore]
    fn draw_gradient() -> Result<()> {
        let json_path = PathBuf::from("test-resources/IMG_20260123_232343.json");
        let image = load_labelme(&json_path)?;
        let image = preprocess_spline_contours(image)?;
        let (outer_mask, inner_mask) = build_outer_inner_masks(image.contours.get(0).unwrap(), &image.size)?;
        let out_dir = Path::new("out");
        create_dir_all(out_dir)?;

        imgcodecs::imwrite(out_dir.join("outer.png").to_str().unwrap(), &outer_mask, &Vector::new())?;
        imgcodecs::imwrite(out_dir.join("inner.png").to_str().unwrap(), &inner_mask, &Vector::new())?;
        Ok(())
    }
}