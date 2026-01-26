use std::path::{Path, PathBuf};
use anyhow::Result;
use log::error;
use opencv::core::Vector;
use opencv::imgcodecs;
use rayon::spawn;

mod types;
mod lablelme_loader;
mod spline;
mod gradient;
mod preprocess;
mod pipeline;
mod split;

fn main() -> Result<()>{
    let path_buf = PathBuf::from("test-resources/IMG_20260123_232343.json");
    let image = lablelme_loader::load_labelme(&path_buf)?;
    let (tx, rx) = std::sync::mpsc::channel();
    spawn(|| {
        if let Err(e) = pipeline::run(image, "test".to_string(), tx) {
            error!("Error {:?}", e);
        }
    });

    let out_dir = Path::new("out");
    while let Ok(task) = rx.recv() {
        let postfix = task.tensor_idx.map(|idx| format!("_{}", idx)).unwrap_or_default();
        imgcodecs::imwrite(
            out_dir.join(format!("{}{}.png", task.prefix, postfix)).to_str().unwrap(),
            &task.mat,
            &Vector::new(),
        )?;
    }

    Ok(())
}
