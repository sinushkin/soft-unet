use crate::types::{ContourGroup, INNER_LABEL, RectSize, su_points_to_cv_points, ContourCrop};
use anyhow::Result;
use opencv::core::{CV_8U, CV_8UC1, NORM_MINMAX, Point, Scalar, Vector, bitwise_and, bitwise_not, no_array, normalize, bitwise_or};
use opencv::prelude::{Mat, MatExprTraitConst};
use opencv::{imgcodecs, imgproc};
use std::path::Path;
use log::Level;


pub fn build_gradient_mask(contour_crop: &ContourCrop) -> Result<Mat> {
    let (outer_mask, inner_mask) =
        build_outer_inner_masks(&contour_crop.contour_group, &contour_crop.size)?;
    let result = build_gradient(&outer_mask, &inner_mask)?;
    Ok(result)
}

fn build_outer_inner_masks(contour: &ContourGroup, size: &RectSize) -> Result<(Mat, Mat)> {
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
        Point::new(0, 0),
    )?;

    let inner_vec: Vector<Vector<Point>> = contour
        .inners
        .iter()
        .filter(|inner| inner.label.starts_with(INNER_LABEL))
        .map(|inner| su_points_to_cv_points(&inner.points))
        .collect();
    imgproc::fill_poly(
        &mut inner_mask,
        &inner_vec,
        Scalar::all(255.0),
        imgproc::LINE_8,
        0,
        Point::new(0, 0),
    )?;

    Ok((outer_mask, inner_mask))
}

fn build_gradient(outer_mask: &Mat, inner_mask: &Mat) -> Result<Mat> {
    let out_dir = Path::new("out");
    // 255 → 0
    // 0 → 255
    let mut inv_inner = Mat::default();

    bitwise_not(&inner_mask, &mut inv_inner, &no_array())?;
    // space between inner and outer is allowed
    let mut ring = Mat::default();
    bitwise_and(&inv_inner, &outer_mask, &mut ring, &no_array())?;
    if log::log_enabled!(Level::Trace) {
        imgcodecs::imwrite(
            out_dir.join("ring.png").to_str().unwrap(),
            &ring,
            &Vector::new(),
        )?;
    }

    // gradient itself
    let mut inv_inner_dist = Mat::default();
    imgproc::distance_transform(&inv_inner, &mut inv_inner_dist, imgproc::DIST_L2, 5, CV_8U)?;
    if log::log_enabled!(Level::Trace) {
        imgcodecs::imwrite(
            out_dir.join("dist.png").to_str().unwrap(),
            &inv_inner_dist,
            &Vector::new(),
        )?;
    }

    let mut dist_normalized = Mat::default();
    normalize(
        &inv_inner_dist,
        &mut dist_normalized,
        35.0,   //TODO adjust using config or slider
        255.0,
        NORM_MINMAX,
        CV_8U,
        &no_array(),
    )?;

    if log::log_enabled!(Level::Trace) {
        imgcodecs::imwrite(
            out_dir.join("dist_normalized.png").to_str().unwrap(),
            &dist_normalized,
            &Vector::new(),
        )?;
    }

    let mut inv_dist_normalized = Mat::default();
    bitwise_not(&dist_normalized, &mut inv_dist_normalized, &no_array())?;
    if log::log_enabled!(Level::Trace) {
        imgcodecs::imwrite(
            out_dir.join("inv_dist_normalized.png").to_str().unwrap(),
            &inv_dist_normalized,
            &Vector::new(),
        )?;
    }

    // inner + gradient ring
    let mut result = Mat::default();
    bitwise_or(&inv_dist_normalized, &inner_mask, &mut result, &outer_mask)?;
    if log::log_enabled!(Level::Trace) {
        imgcodecs::imwrite(
            out_dir.join("result.png").to_str().unwrap(),
            &result,
            &Vector::new(),
        )?;
    }

    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lablelme_loader::load_labelme;
    use anyhow::Result;
    use opencv::core::Vector;
    use opencv::{imgcodecs};
    use std::fs::create_dir_all;
    use std::path::{Path, PathBuf};
    use env_logger::Builder;
    use log::LevelFilter;


    #[test]
    #[ignore]
    fn draw_gradient() -> Result<()> {
        let out_dir = Path::new("out");
        create_dir_all(out_dir)?;
        Builder::new()
            .filter_level(LevelFilter::Trace)
            .init();

        let json_path = PathBuf::from("test-resources/IMG_20260123_232343.json");
        let image = load_labelme(&json_path)?;
        let some_group_crop = image.contour_crops.get(0).unwrap();
        let (outer_mask, inner_mask) =
            build_outer_inner_masks(&some_group_crop.contour_group, &some_group_crop.size)?;
        imgcodecs::imwrite(
            out_dir.join("outer_mask.png").to_str().unwrap(),
            &outer_mask,
            &Vector::new(),
        )?;
        imgcodecs::imwrite(
            out_dir.join("inner_mask.png").to_str().unwrap(),
            &inner_mask,
            &Vector::new(),
        )?;

        build_gradient(&outer_mask, &inner_mask)?;
        /*
        let mut gaussian = Mat::default();
        imgproc::gaussian_blur(
            &result,
            &mut gaussian,
            core::Size::new(15, 15),
            0.0,
            0.0,
            core::BORDER_DEFAULT,
        )?;
        */
        Ok(())
    }
}
