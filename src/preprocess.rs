use std::collections::HashMap;
use crate::spline::smooth_polygon;
use crate::types::{Augmentations, Contour, ContourCrop, ContourGroup, Image, RawImage, RectSize, SuPoint, SuPoint2F};
use anyhow::{anyhow, bail, Result};
use log::warn;
use rand::RngExt;
use crate::augmentation::AffineTransformBuilder;
use crate::configuration::OUTER_LABEL;

pub fn preprocess_image(image: RawImage) -> Result<Augmentations> {
    let augmentation_list = (0..10).map(|_| -> Result<Image> {
        let augmented = augment(&image)?;
        let image = clean(augmented)?;
        let image = convert(image, true)?;        
        Ok(image)
    }).collect::<Result<Vec<_>>>()?;

    Ok(Augmentations {
        original: convert(image, false)?,
        augmentation_list,
    })
}

fn augment(image: &RawImage) -> Result<RawImage> {

    let mut rng = rand::rng();
    let center = SuPoint2F { x: image.size.width as f32 / 2.0, y: image.size.height as f32 / 2.0 };
    //pick up some contour center and rotate around it
    let rotation_point = if image.contours.is_empty() {
        center.into()
    }else {
        // ---- pick random contour group ----
        let idx = rng.random_range(0..image.contours.len());
        image.contours[idx].calculate_center()
    };

    // ---- random params (example ranges) ----
    let angle = rng.random_range(-20.0..20.0);     // degrees
    let shift_x = rng.random_range(-30.0..30.0);
    let shift_y = rng.random_range(-30.0..30.0);
    let shear_x = rng.random_range(-0.1..0.1);
    let shear_y = rng.random_range(-0.1..0.1);
    let scale = rng.random_range(0.9..1.1);
    let flip = rng.random_bool(0.5);

    // ---- build affine ----
    let mut builder = AffineTransformBuilder::new()
        .rotate(angle, rotation_point);   // rotation around contour center
    if flip {
        builder = builder.flip_horizontal(center.x);
    }
    builder = builder.shift_x(shift_x)
        .shift_y(shift_y)
        .shear_x(shear_x)
        .shear_y(shear_y)
        .scale_uniform(scale, center);

    let affine = builder.build();
    let mat = affine.apply_mat(&image.mat)?;
    let contours : Vec<Contour>= image.contours.iter()
        .map(|c| affine.apply_contour(c))
        .collect();
    Ok(RawImage{
        mat,
        size: image.size,
        contours,
    })
}


pub fn convert(raw_image: RawImage, skip_undefined_outer: bool) -> Result<Image> {
    let contour_crops = build_contours(raw_image.size, raw_image.contours, skip_undefined_outer)?;
    Ok(Image{
        mat: raw_image.mat,
        size: raw_image.size,
        contour_crops,
    })
}

///Some contours become outside the image. Its are not visible anymore. We remove them.
pub fn clean(image: RawImage) -> Result<RawImage> {
    let RectSize{ width , height } = image.size;
    let width = width as f32;
    let height = height as f32;
    let contours = image.contours.into_iter()
        .filter(|c| {
            let pts_count = c.points.iter()
                .filter(|p| p.x>=0.0 && p.x<width && p.y>=0.0 && p.y<height)
                .count();
            pts_count >= 3
        }).collect();
    
    Ok(RawImage{
        mat: image.mat,
        size: image.size,
        contours,
    })
}


fn build_contours(
    original_size: RectSize,
    shapes: Vec<Contour>,
    skip_undefined_outer: bool,
) -> Result<Vec<ContourCrop>> {

    let mut map: HashMap<usize, (Option<Contour>, Vec<Contour>)> = HashMap::new();

    for contour in shapes {
        let (is_outer, _prefix, idx) = parse_label(&contour.label)?;
        let entry = map.entry(idx).or_insert((None, Vec::new()));

        if is_outer {
            if entry.0.is_some() {
                bail!("Duplicate outer contour for index {}", idx);
            }
            entry.0 = Some(contour);
        } else {
            entry.1.push(contour);
        }
    }

    let mut result = Vec::with_capacity(map.len());

    for (idx, (outer_opt, inners)) in map {
        if skip_undefined_outer && outer_opt.is_none() {
            warn!("Skipping contour group with undefined outer contour for index {}", idx);
            continue;
        }
        let outer = outer_opt
            .ok_or_else(|| anyhow!("Missing outer contour for index {}", idx))?;

        // --- compute bounding box (based on outer)
        let (min_x, min_y, max_x, max_y) = contour_bbox(&outer.points);

        // clamp to image (optional safety)
        let min_x = min_x.max(0);
        let min_y = min_y.max(0);

        let max_x = max_x.min(original_size.width as i32 - 1);
        let max_y = max_y.min(original_size.height as i32 - 1);

        let width  = (max_x - min_x + 1) as usize;
        let height = (max_y - min_y + 1) as usize;

        let offset = SuPoint {
            x: min_x as usize,
            y: min_y as usize,
        };

        // --- shift contours into crop coordinates
        let shifted_outer = shift_contour(outer, min_x, min_y);

        let shifted_inners = inners.into_iter()
            .map(|c| shift_contour(c, min_x, min_y))
            .collect();

        let mut group = ContourGroup {
            idx,
            outer_center: shifted_outer.calculate_center(),
            outer: shifted_outer,
            inners: shifted_inners,
        };
        spline_contours(&mut group)?;

        result.push(ContourCrop {
            offset,
            size: RectSize { width, height },
            contour_group: group
        });
    }

    Ok(result)
}



pub fn parse_label(label: &str) -> Result<(bool, String, usize)> {
    // split once at '_'. outer_1 -> (outer, 1)
    let mut it = label.splitn(2, '_');
    let label = it.next().ok_or_else(|| anyhow::anyhow!("Bad label"))?;
    let num  = it.next().ok_or_else(|| anyhow::anyhow!("Bad label"))?;

    Ok((OUTER_LABEL == label, label.to_string(), num.parse()?))
}

fn contour_bbox(points: &[SuPoint2F]) -> (i32, i32, i32, i32) {
    let mut min_x = i32::MAX;
    let mut min_y = i32::MAX;
    let mut max_x = i32::MIN;
    let mut max_y = i32::MIN;

    for p in points {
        let x = p.x as i32;
        let y = p.y as i32;

        min_x = min_x.min(x);
        min_y = min_y.min(y);
        max_x = max_x.max(x);
        max_y = max_y.max(y);
    }

    (min_x, min_y, max_x, max_y)
}

fn shift_contour(contour: Contour, dx: i32, dy: i32) -> Contour {
    Contour {
        label: contour.label,
        points: contour.points.iter()
            .map(|p| SuPoint2F {
                x: p.x - dx as f32,
                y: p.y - dy as f32,
            })
            .collect(),
    }
}
pub fn spline_contours(contour_group: &mut ContourGroup) -> Result<()> {
    contour_group.outer.points = smooth_polygon(&contour_group.outer.points, 10);
    for inner in &mut contour_group.inners {
        inner.points = smooth_polygon(&inner.points, 10); //TODO configureadable
    }
    Ok(())
}
