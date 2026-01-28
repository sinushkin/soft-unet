use crate::gradient::build_gradient_mask;
use crate::preprocess::{parse_label, preprocess_image};
use crate::split::split_contours;
use crate::types::{su_points_to_cv_points, Contour, ContourCrop, Image, RawImage, RectSize};
use anyhow::{anyhow, bail, Result};
use opencv::prelude::{Mat, MatExprTraitConst, MatTraitConst};
use std::collections::HashMap;
use std::time::Instant;
use crossbeam_channel::Sender;
use opencv::core::{Point, Rect, Scalar, Vector, CV_8UC1};
use opencv::imgproc;
use crate::configuration::Configuration;
use itertools::Itertools;
use log::info;

pub struct PipelineTask {
    pub image: RawImage,
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
        let PipelineTask { image, prefix } = task;
        info!("Processing {}", prefix);
        let start = Instant::now();
        let augmentations = preprocess_image(image)?;
        let augmentation_time = start.elapsed();
        self.process_image(augmentations.original, prefix.clone(), tx.clone())?;
        augmentations.augmentation_list
            .into_iter()
            .enumerate()
            .try_for_each(|(aug_index, augmentation)| -> Result<()> {
                self.process_image(augmentation, format!("{}-aug-{}", &prefix, aug_index).to_string(), tx.clone())?;
                Ok(())
        })?;
        let total_time = start.elapsed();
        info!("{}: augmentation time: {:?}, total time: {:?}", prefix, augmentation_time, total_time);
        Ok(())
    }

    /// Построить маски для каждой группы контуров.
    /// построить тензор бекграунда
    /// построить тензоры сгруппированных масок
    /// Все отправить в tx
    fn process_image(&self, image: Image, prefix: String, tx: Sender<PipelineTaskResult>) -> Result<()> {
        let Image {
            mat,
            size,
            contour_crops
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

        let background_tensor = Self::build_background_tensor(size, &contour_crops, &gradients_map)?;
        tx.send(PipelineTaskResult {
            prefix: prefix.clone(),
            tensor_idx: Some(0),
            mat: background_tensor,
        })?;

        let tensors = split_contours(contour_crops, size)?;
        tensors
            .iter()
            .enumerate()
            .try_for_each(|(idx, crops)| -> Result<()> {
                let tensor = Self::build_tensor(size, crops, &gradients_map)?;
                tx.send(PipelineTaskResult {
                    prefix: prefix.clone(),
                    tensor_idx: Some(idx+1),
                    mat: tensor,
                })?;
                Ok(())
            })?;
        tx.send(PipelineTaskResult {
            prefix,
            tensor_idx: None,
            mat,
        })?;
        Ok(())
    }

    fn build_background_tensor(size: RectSize, crops: &Vec<ContourCrop>, gradients_map: &HashMap<usize, Mat>) -> Result<Mat> {
        let foreground = Self::build_tensor(size, crops, gradients_map)?;

        let mut background = Mat::default();
        // if foreground == 0 → 255 (background)
        // else → 0
        imgproc::threshold(
            &foreground,
            &mut background,
            0.0,
            255.0,
            imgproc::THRESH_BINARY_INV,
        )?;

        Ok(background)
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
        inners
            .iter()
            .map(|inner| -> Result<(String, &Contour)> {
                let (_, label_prefix, _) = parse_label(inner.label.as_str())?;
                Ok((label_prefix, inner))
            })
            .collect::<Result<Vec<_>>>()?      // <-- handle Result
            .into_iter()
            .filter(|(label_prefix, _)| {
                self.configuration.label_map.contains_key(label_prefix)
            })
            .into_group_map_by(|(label_prefix, _)| label_prefix.clone())
            .into_iter()
            .try_for_each(|(label_prefix, inners)| -> Result<()> {
                let mut polys = Vector::<Vector<Point>>::new();
                for inner in inners {
                    polys.push(su_points_to_cv_points(&inner.1.points));
                }

                let intensity = *self.configuration.label_map
                    .get(&label_prefix)
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