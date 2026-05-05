use opencv::prelude::{Mat, MatTraitConst};
use crate::types::{Contour, SuPoint2F};
use anyhow::Result;
use opencv::imgproc;

/// [x']   [ a  b  tx ]   [x]
/// [y'] = [ c  d  ty ] × [y]
/// [1 ]   [ 0  0  1  ]   [1]
#[derive(Debug, Copy, Clone)]
pub struct AffineTransform {
    pub a: f32,
    pub b: f32,
    pub c: f32,
    pub d: f32,
    pub tx: f32,
    pub ty: f32,
}

pub struct AffineTransformBuilder {
    t: AffineTransform,
}

impl Default for AffineTransform {
    fn default() -> Self {
        Self {
            a: 1.0, b: 0.0,
            c: 0.0, d: 1.0,
            tx: 0.0, ty: 0.0,
        }
    }
}

impl AffineTransform {

    #[inline]
    pub fn mul(self, o: AffineTransform) -> AffineTransform {
        AffineTransform {
            a: self.a * o.a + self.b * o.c,
            b: self.a * o.b + self.b * o.d,
            c: self.c * o.a + self.d * o.c,
            d: self.c * o.b + self.d * o.d,

            tx: self.a * o.tx + self.b * o.ty + self.tx,
            ty: self.c * o.tx + self.d * o.ty + self.ty,
        }
    }

    pub fn to_mat(&self) -> Result<Mat> {
        Mat::from_slice_2d(&[
            [self.a, self.b, self.tx],
            [self.c, self.d, self.ty],
        ]).map_err(|e| e.into())
    }

    pub fn mutate_point(&self, p: &mut SuPoint2F) {
        let x = p.x;
        let y = p.y;

        p.x = self.a * x + self.b * y + self.tx;
        p.y = self.c * x + self.d * y + self.ty;
    }

    pub fn mutate_points(&self, points: &mut [SuPoint2F]) {
        for p in points {
            self.mutate_point(p);
        }
    }

    pub fn apply_point(&self, p: &SuPoint2F) -> SuPoint2F {
        SuPoint2F {
            x: self.a * p.x + self.b * p.y + self.tx,
            y: self.c * p.x + self.d * p.y + self.ty,
        }
    }

    pub fn apply_contour(&self, contour: &Contour) -> Contour {
        let transformed_pts = self
            .apply_pts(contour.points.as_slice())
            .into_iter()
            .collect::<Vec<_>>();
        Contour {
            label: contour.label.clone(),
            points: transformed_pts,
        }
    }

    pub fn apply_pts(&self, pts: &[SuPoint2F]) -> Vec<SuPoint2F> {
        pts.iter()
            .map(|p| self.apply_point(p))
            .collect()
    }

    pub fn apply_mat(
        &self,
        mat: &Mat,
    ) -> Result<Mat> {

        let m = self.to_mat()?;
        let mut dst = Mat::default();
        let dst_size = mat.size()?;

        imgproc::warp_affine(
            mat,
            &mut dst,
            &m,
            dst_size,
            imgproc::INTER_LINEAR,
            opencv::core::BORDER_CONSTANT,
            opencv::core::Scalar::all(0.0),
        )?;

        Ok(dst)
    }
}

impl AffineTransformBuilder {

    pub fn new() -> Self {
        Self {
            t: AffineTransform::default(),
        }
    }

    pub fn build(self) -> AffineTransform {
        self.t
    }

    // 🔁 Rotate around arbitrary point (cx, cy)
    pub fn rotate(mut self, deg: f32, center: SuPoint2F) -> Self {
        let SuPoint2F{ x, y} = center;
        let rad = deg.to_radians();
        let cos = rad.cos();
        let sin = rad.sin();

        let r = AffineTransform {
            a: cos,
            b: -sin,
            c: sin,
            d: cos,

            // pivot compensation
            tx: x - cos * x + sin * y,
            ty: y - sin * x - cos * y,
        };

        self.t = r.mul(self.t);
        self
    }

    /// Scale around a pivot (cx, cy)
    pub fn scale(mut self, sx: f32, sy: f32, center: SuPoint2F) -> Self {
        let SuPoint2F{ x, y} = center;
        let s = AffineTransform {
            a: sx,
            b: 0.0,
            c: 0.0,
            d: sy,
            tx: x - sx * x,
            ty: y - sy * y,
        };

        self.t = s.mul(self.t);
        self
    }

    /// Uniform scale around pivot
    pub fn scale_uniform(mut self, s: f32, center: SuPoint2F) -> Self {
        self.scale(s, s, center)
    }

    // 📦 Shift X
    pub fn shift_x(mut self, dx: f32) -> Self {
        let tr = AffineTransform {
            a: 1.0, b: 0.0,
            c: 0.0, d: 1.0,
            tx: dx, ty: 0.0,
        };

        self.t = tr.mul(self.t);
        self
    }

    // 📦 Shift Y
    pub fn shift_y(mut self, dy: f32) -> Self {
        let tr = AffineTransform {
            a: 1.0, b: 0.0,
            c: 0.0, d: 1.0,
            tx: 0.0, ty: dy,
        };

        self.t = tr.mul(self.t);
        self
    }

    /// 📐 Shear
    /// shx = 0.1 → x-coordinate shifts 10% of y
    pub fn shear_x(mut self, shx: f32) -> Self {
        let sh = AffineTransform {
            a: 1.0,
            b: shx,
            c: 0.0,
            d: 1.0,
            tx: 0.0,
            ty: 0.0,
        };

        self.t = sh.mul(self.t);
        self
    }

    /// 📐 Shear
    /// shy = 0.2 → y-coordinate shifts 20% of x
    pub fn shear_y(mut self, shy: f32) -> Self {
        let sh = AffineTransform {
            a: 1.0,
            b: 0.0,
            c: shy,
            d: 1.0,
            tx: 0.0,
            ty: 0.0,
        };

        self.t = sh.mul(self.t);
        self
    }

    // 🔄 Flip horizontal around cx
    // Intuition:
    // Imagine cx is the vertical line you want to flip around.
    // X-coordinates reflect across this line:
    pub fn flip_horizontal(mut self, cx: f32) -> Self {
        let f = AffineTransform {
            a: -1.0,
            b: 0.0,
            c: 0.0,
            d: 1.0,
            tx: 2.0 * cx,
            ty: 0.0,
        };

        self.t = f.mul(self.t);
        self
    }

    // 🔃 Flip vertical around cy
    pub fn flip_vertical(mut self, cy: f32) -> Self {
        let f = AffineTransform {
            a: 1.0,
            b: 0.0,
            c: 0.0,
            d: -1.0,
            tx: 0.0,
            ty: 2.0 * cy,
        };

        self.t = f.mul(self.t);
        self
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use crate::lablelme_loader::load_labelme;
    use anyhow::Result;
    use std::fs::create_dir_all;
    use std::path::{Path, PathBuf};
    use opencv::core::{Point, Scalar, Vector};
    use opencv::imgcodecs;
    use crate::types::su_points_to_cv_points;

    #[test]
    #[ignore]
    fn draw_augmentation() -> Result<()> {
        let out_dir = Path::new("out");
        create_dir_all(out_dir)?;

        let json_path = PathBuf::from("test-resources/IMG_20260123_232343.json");
        let raw_image = load_labelme(&json_path)?;


        let affine = AffineTransformBuilder::new()
            .shift_x(20.0)
            .shift_y(-10.0)
            .shear_x(0.15)
            .flip_horizontal(raw_image.size.width as f32 / 2.0)
            .build();

        let mut aug_mat = affine.apply_mat(&raw_image.mat)?;
        let contour = affine.apply_contour(&raw_image.contours.first().unwrap());

        let mut outer_vec = Vector::<Vector<Point>>::new();
        let outer_pts = su_points_to_cv_points(&contour.points);
        outer_vec.push(outer_pts);

        imgproc::fill_poly(
            &mut aug_mat,
            &outer_vec,
            Scalar::new(123.0, 234.0, 34.0, 0.0), // Blue
            imgproc::LINE_8,
            0,
            Point::new(0, 0),
        )?;
        imgcodecs::imwrite(
            out_dir.join("aug_mat.png").to_str().unwrap(),
            &aug_mat,
            &Vector::new(),
        )?;

        Ok(())
    }
}
