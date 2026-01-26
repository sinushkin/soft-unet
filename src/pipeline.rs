use crate::gradient::build_gradient_mask;
use crate::preprocess::preprocess_image;
use crate::split::split_contours;
use crate::types::{ContourCrop, Image, RectSize};
use anyhow::{anyhow, bail, Result};
use opencv::prelude::{Mat, MatExprTraitConst, MatTraitConst};
use std::collections::HashMap;
use crossbeam_channel::Sender;
use opencv::core::{Rect, CV_8UC1};

/// Для буферизирования результатов, только один поток будет писать на диск
pub struct SaveFileTask {
    pub prefix: String,
    pub tensor_idx: Option<usize>,
    pub mat: Mat,
}

pub fn run(image: Image, prefix: String, tx: Sender<SaveFileTask>) -> Result<()> {
    let mut image = image;
    preprocess_image(&mut image)?;
    let Image {
        original,
        size,
        contour_crops,
    } = image;

    let mut gradients_map: HashMap<usize, Mat> = HashMap::new();
    contour_crops
        .iter()
        .try_for_each(|contour_crop| -> Result<()> {
            let mat = build_gradient_mask(contour_crop)?;
            gradients_map.insert(contour_crop.contour_group.idx, mat);
            Ok(())
        })?;

    let tensors = split_contours(contour_crops, size)?;
    tensors
        .iter()
        .enumerate()
        .try_for_each(|(idx, crops)| -> Result<()> {
            let tensor = build_tensor(size, crops, &gradients_map)?;
            tx.send(SaveFileTask {
                prefix: prefix.clone(),
                tensor_idx: Some(idx),
                mat: tensor,
            })?;            
            Ok(())
        })?;
    tx.send(SaveFileTask {
        prefix,
        tensor_idx: None,
        mat: original,
    })?;
    Ok(())
}

fn build_tensor(size: RectSize, crops: &Vec<ContourCrop>, gradients_map: &HashMap<usize, Mat>) -> Result<Mat> {
    // --- create black image
    let mut tensor = Mat::zeros(
        size.height as i32,
        size.width as i32,
        CV_8UC1
    )?.to_mat()?;

    // --- paste each crop
    for crop in crops {
        let idx = crop.contour_group.idx;
        let grad = gradients_map
            .get(&idx)
            .ok_or_else(|| anyhow!("Missing gradient for idx {}", idx))?;

        let x = crop.offset.x as i32;
        let y = crop.offset.y as i32;
        let w = crop.size.width as i32;
        let h = crop.size.height as i32;

        // safety (optional but recommended)
        if x < 0 || y < 0 || x + w > size.width as i32 || y + h > size.height as i32 {
            bail!("Crop {} out of bounds", idx);
        }

        // --- ROI on destination
        let roi = Rect::new(x, y, w, h);
        let mut dst_roi = Mat::roi_mut(&mut tensor, roi)?;

        // --- copy gradient into ROI
        grad.copy_to_masked(&mut dst_roi, grad)?;
    }
    Ok(tensor)
}