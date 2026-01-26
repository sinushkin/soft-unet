use crate::types::{ContourCrop, RectSize, SuPoint, SuPoint2F};
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
        let local_center = contour_center(&outer.points);

        // convert to original image coords
        let global_center_x = local_center.x + crop.offset.x;

        if global_center_x < mid_x {
            left.push(crop);
        } else {
            right.push(crop);
        }
    }

    Ok(vec![left, right])
}

fn contour_center(points: &[SuPoint2F]) -> SuPoint {
    let mut sx = 0;
    let mut sy = 0;

    for p in points {
        sx += p.x as usize;
        sy += p.y as usize;
    }

    let n = points.len();
    SuPoint {
        x: sx / n,
        y: sy / n,
    }
}
