use opencv::core::{add, convert_scale_abs, no_array, randn, Scalar, Vec3b, CV_64FC1};
use opencv::imgproc;
use opencv::prelude::{Mat, MatExprTraitConst, MatTrait, MatTraitConst};
use rand::RngExt;
use anyhow::Result;
use rand::prelude::ThreadRng;

pub struct AugmentColour{
    rng : ThreadRng
}

//TODO tests
impl AugmentColour {

    pub fn new() -> Self {
        Self { rng : rand::rng()}
    }

    pub fn salt_and_pepper(&mut self, mat: &mut Mat, hardness: f32) -> Result<()> {
        if hardness <= 0.0 {
            return Ok(());
        }


        let total = (mat.rows() * mat.cols()) as usize;
        let count = (total as f32 * hardness * 0.05) as usize;

        let channels = mat.channels();

        for _ in 0..count {
            let x = self.rng.random_range(0..mat.cols());
            let y = self.rng.random_range(0..mat.rows());

            let val = if self.rng.random_bool(0.5) { 0u8 } else { 255u8 };
            match channels {
                1 => {
                    // grayscale
                    *mat.at_2d_mut::<u8>(y, x)? = val;
                }
                3 => {
                    // color (BGR)
                    let px = mat.at_2d_mut::<Vec3b>(y, x)?;
                    px[0] = val;
                    px[1] = val;
                    px[2] = val;
                }
                _ => anyhow::bail!("Unsupported channel count: {}", channels),
            }
        }

        Ok(())
    }

    pub fn gaussian(&mut self, mat: &mut Mat, hardness: f32) -> Result<()> {
        if hardness <= 0.0 {
            return Ok(());
        }

        let sigma = hardness * 30.0; // noise strength

        // noise has same shape & type as image
        let mut noise = Mat::zeros(mat.rows(), mat.cols(), mat.typ())?.to_mat()?;

        // IMPORTANT: mean/stddev must be 1-channel float mats

        let mean = Mat::new_rows_cols_with_default(
            1,
            1,
            CV_64FC1,
            Scalar::all(0.0),
        )?;

        let stddev = Mat::new_rows_cols_with_default(
            1,
            1,
            CV_64FC1,
            Scalar::all(sigma as f64),
        )?;

        // generate noise
        randn(&mut noise, &mean, &stddev)?;

        // add noise to image
        let mut tmp = Mat::default();
        add(mat, &noise, &mut tmp, &no_array(), -1)?;
        *mat = tmp;
        Ok(())
    }

    pub fn dropout(&mut self, mat: &mut Mat, hardness: f32) -> Result<()> {
        if hardness <= 0.0 {
            return Ok(());
        }
        let prob = hardness * 0.3; // up to 30%
        let channels = mat.channels();
        for y in 0..mat.rows() {
            for x in 0..mat.cols() {
                if self.rng.random_bool(prob as f64) {
                    if channels == 1 {
                        // grayscale
                        *mat.at_2d_mut::<u8>(y, x)? = 0;
                    } else if channels == 3 {
                        // color
                        let px = mat.at_2d_mut::<Vec3b>(y, x)?;
                        px[0] = 0;
                        px[1] = 0;
                        px[2] = 0;
                    }
                }
            }
        }

        Ok(())
    }

    pub fn contrast(&mut self, mat: &mut Mat, hardness: f32) -> Result<()> {
        if hardness <= 0.0 {
            return Ok(());
        }

        let alpha = 1.0 + self.rng.random_range(-0.8..0.8) * hardness;
        let mut tmp = Mat::default();
        convert_scale_abs(mat, &mut tmp, alpha as f64, 0.0)?;

        *mat = tmp;

        Ok(())
    }

    pub fn brightness(&mut self, mat: &mut Mat, hardness: f32) -> Result<()> {
        if hardness <= 0.0 { return Ok(()); }

        let beta = self.rng.random_range(-80.0..80.0) * hardness;
        let mut tmp = Mat::default();
        convert_scale_abs(mat, &mut tmp, 1.0, beta as f64)?;

        *mat = tmp;
        Ok(())
    }

    pub fn saturation(&mut self, mat: &mut Mat, hardness: f32) -> Result<()> {
        if hardness <= 0.0 || mat.channels() == 1 {
            return Ok(());
        }

        let mut hsv = Mat::default();
        imgproc::cvt_color(mat, &mut hsv, imgproc::COLOR_BGR2HSV, 0)?;

        let scale = 1.0 + self.rng.random_range(-1.0..1.0) * hardness;

        for y in 0..hsv.rows() {
            for x in 0..hsv.cols() {
                let px = hsv.at_2d_mut::<Vec3b>(y, x)?;
                px[1] = ((px[1] as f32 * scale).clamp(0.0, 255.0)) as u8;
            }
        }

        imgproc::cvt_color(&hsv, mat, imgproc::COLOR_HSV2BGR, 0)?;

        Ok(())
    }

    pub fn hue(&mut self, mat: &mut Mat, hardness: f32) -> Result<()> {
        if hardness <= 0.0 || mat.channels() == 1 {
            return Ok(());
        }

        let mut hsv = Mat::default();
        imgproc::cvt_color(mat, &mut hsv, imgproc::COLOR_BGR2HSV, 0)?;

        let shift = self.rng.random_range(-30..30) as i32;
        let shift = (shift as f32 * hardness) as i32;

        for y in 0..hsv.rows() {
            for x in 0..hsv.cols() {
                let px = hsv.at_2d_mut::<Vec3b>(y, x)?;
                let h = (px[0] as i32 + shift).rem_euclid(180);
                px[0] = h as u8;
            }
        }

        imgproc::cvt_color(&hsv, mat, imgproc::COLOR_HSV2BGR, 0)?;

        Ok(())
    }

    pub fn color_jitter(&mut self, mat: &mut Mat, hardness: f32) -> Result<()> {
        if hardness <= 0.0 || mat.channels() == 1 {
            return Ok(());
        }

        let r = 1.0 + self.rng.random_range(-hardness..hardness);
        let g = 1.0 + self.rng.random_range(-hardness..hardness);
        let b = 1.0 + self.rng.random_range(-hardness..hardness);

        for y in 0..mat.rows() {
            for x in 0..mat.cols() {
                let px = mat.at_2d_mut::<Vec3b>(y, x)?;
                px[0] = ((px[0] as f32 * b).clamp(0.0, 255.0)) as u8;
                px[1] = ((px[1] as f32 * g).clamp(0.0, 255.0)) as u8;
                px[2] = ((px[2] as f32 * r).clamp(0.0, 255.0)) as u8;
            }
        }

        Ok(())
    }

}