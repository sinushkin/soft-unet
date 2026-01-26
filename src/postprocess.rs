
use anyhow::Result;
use opencv::core::Size;
use opencv::imgproc;
use opencv::prelude::Mat;

pub fn resize(src: &Mat) -> Result<Mat> {
    let mut dst = Mat::default();
    imgproc::resize(
        &src,
        &mut dst,
        Size::new(480, 256),//TODO сконфигурировать препроцесс и постпроцесс
        0.0,
        0.0,
        imgproc::INTER_AREA,
    )?;
    Ok(dst)
}
