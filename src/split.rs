use crate::types::{ContourCrop, RectSize};
use anyhow::{Result, bail};

/// Стратегия деления контуров
/// К примеру есть 6 групп - 3 в одну маску, 3 в другую

/// пока что одна стратегия.
/// если центр внешнего контура попал на левую половину - влево, на право - вправо
pub fn split_contours(
    contours: Vec<ContourCrop>,
    original_size: RectSize,
) -> Result<Vec<Vec<ContourCrop>>> {
    let mid_x = original_size.width / 2;

    let mut left = Vec::new();
    let mut right = Vec::new();

    for crop in contours {
        let outer = &crop.contour_group.outer;

        if outer.points.is_empty() {
            bail!("Outer contour has no points");
        }

        // center in crop-local coords
        let local_center = crop.contour_group.outer_center;

        // convert to original image coords
        let global_center_x = local_center.x as usize + crop.offset.x;

        if global_center_x < mid_x {
            left.push(crop);
        } else {
            right.push(crop);
        }
    }

    Ok(vec![left, right])
}
