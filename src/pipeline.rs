use crate::gradient::build_gradient_mask;
use crate::preprocess::preprocess_image;
use crate::split::split_contours;
use crate::types::{su_points_to_cv_points, Contour, ContourCrop, Image, RectSize};
use anyhow::{anyhow, bail, Result};
use opencv::prelude::{Mat, MatExprTraitConst, MatTraitConst};
use std::collections::HashMap;
use crossbeam_channel::Sender;
use opencv::core::{Point, Rect, Scalar, Vector, CV_8UC1};
use opencv::imgproc;
use crate::configuration::Configuration;
use itertools::Itertools;

pub struct PipelineTask {
    pub image: Image,
    pub prefix: String,
}

/// Для буферизирования результатов, только один поток будет писать на диск
pub struct PipelineTaskResult {
    pub prefix: String,
    pub tensor_idx: Option<usize>,
    pub mat: Mat,
}

pub struct Pipeline{
    configuration: Configuration
}

impl Pipeline {
    pub fn new(configuration: Configuration) -> Self {
        Self { configuration }
    }

    pub fn run(&self, task: PipelineTask, tx: Sender<PipelineTaskResult>) -> Result<()> {
        let PipelineTask { mut image, prefix } = task;
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
                let mut mat = build_gradient_mask(contour_crop, self.configuration.gradient_alpha)?;
                self.fill_specials(&mut mat, &contour_crop.contour_group.inners)?;
                gradients_map.insert(contour_crop.contour_group.idx, mat);
                Ok(())
            })?;

        let tensors = split_contours(contour_crops, size)?;
        tensors
            .iter()
            .enumerate()
            .try_for_each(|(idx, crops)| -> Result<()> {
                let tensor = Self::build_tensor(size, crops, &gradients_map)?;
                tx.send(PipelineTaskResult {
                    prefix: prefix.clone(),
                    tensor_idx: Some(idx),
                    mat: tensor,
                })?;
                Ok(())
            })?;
        tx.send(PipelineTaskResult {
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

    /// Fill special pixels (like films, bags) with preconfigurable values
    fn fill_specials(&self, mat: &mut Mat, inners: &Vec<Contour>) -> Result<()> {
        inners.iter()
            .filter(|inner| self.configuration
                .label_map.contains_key(&inner.label_prefix))
            .into_group_map_by(|inner| inner.label_prefix.as_str())
            .into_iter()
            .try_for_each(|(label_prefix, inners)| -> Result<()> {
                let mut polys = Vector::<Vector<Point>>::new();
                for inner in inners {
                    polys.push(su_points_to_cv_points(&inner.points));
                }

                let intensity = *self.configuration.label_map
                    .get(label_prefix)
                    .ok_or_else(|| anyhow!("Missing label value for {}", label_prefix))?;
                imgproc::fill_poly(
                    mat,
                    &polys,
                    Scalar::all(intensity as f64),
                    imgproc::LINE_8,
                    0,
                    Point::new(0, 0),
                )?;
                Ok(())
            })?;
        Ok(())
    }


}