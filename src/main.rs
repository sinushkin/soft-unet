use crate::lablelme_loader::load_labelme;
use crate::pipeline::{PipelineTask, PipelineTaskResult};
use anyhow::{Result, anyhow, bail};
use crossbeam_channel::{Receiver, bounded};
use env_logger::Builder;
use log::{LevelFilter, error, info};
use opencv::core::Vector;
use opencv::imgcodecs;
use rayon::ThreadPoolBuilder;
use rayon::prelude::*;
use std::path::{Path, PathBuf};
use std::{env, fs, thread};
use std::thread::JoinHandle;

mod gradient;
mod lablelme_loader;
mod pipeline;
mod preprocess;
mod spline;
mod split;
mod types;
mod postprocess;
mod configuration;
#[allow(unused)]
mod augmentation_affine;
mod augmentation_colour;

fn main() -> Result<()> {
    Builder::new().filter_level(LevelFilter::Info).init();
    let configuration = configuration::load_configuration();

    // --- parse CLI
    let dir = env::args()
        .nth(1)
        .ok_or_else(|| anyhow!("Usage: ./soft-unet <path-to-dir-with-jsons>"))?;

    let dir = PathBuf::from(dir);
    let jsons = list_jsons(dir)?;

    // --- rayon pool
    let cpu = 1.max(num_cpus::get()-1);
    info!("Using {} cpu", cpu);
    let pool = ThreadPoolBuilder::new()
        .num_threads(cpu)
        .build()?;

    // --- bounded channel for results
    let queue_limit = cpu * 3 / 2;
    let (tx, rx) = bounded(queue_limit);

    // --- processing
    let background_start_result =  pool.install(|| -> Result<JoinHandle<Result<()>>> {
        let background_join_handle = thread::spawn(move || {
        let pipeline = pipeline::Pipeline::new(configuration);
        jsons.par_iter().try_for_each(|path| -> Result<()> {
            let file_name = path.file_stem().unwrap().to_string_lossy().to_string();
            let image = load_labelme(&path)?;
            let task = PipelineTask{image, prefix: file_name};
            if let Err(e) = pipeline.run(task, tx.clone()) {
                error!("Pipeline Failed for {}: {:?}", path.display(), e);
            }
            Ok(())
        })});
        Ok(background_join_handle)
    });

    match background_start_result {
        Ok(join_handle) => {
            info!("Background thread started");
            let store_join_handle = thread::spawn( move || {
                if let Err(e) = save_to_disk(rx){
                    error!("Failed to save to disk: {:?}", e);
                }
            });
            join_handle.join().map_err(|_| anyhow!("Background thread failed"))??;
            store_join_handle.join().map_err(|_| anyhow!("Storing files thread failed"))?;
        },
        Err(e) => bail!("Failed to start background thread: {:?}", e),
    }

    Ok(())
}

fn list_jsons(dir: PathBuf) -> Result<Vec<PathBuf>> {
    if !dir.is_dir() {
        bail!("{} is not a directory", dir.display());
    }

    // --- collect json files
    let mut files: Vec<PathBuf> = fs::read_dir(&dir)?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().map(|x| x == "json").unwrap_or(false))
        .collect();
    files.sort(); // deterministic order

    info!("Found {} json files", files.len());
    Ok(files)
}

fn save_to_disk(rx: Receiver<PipelineTaskResult>) -> Result<()> {
    let out_dir = Path::new("out");
    fs::create_dir_all(out_dir)?;
    while let Ok(task) = rx.recv() {
        let postfix = task
            .tensor_idx
            .map(|idx| format!("_{}", idx))
            .unwrap_or_default();
        let resized = postprocess::resize(&task.mat)?;
        imgcodecs::imwrite(
            out_dir
                .join(format!("{}{}.png", task.prefix, postfix))
                .to_str()
                .unwrap(),
            &resized,
            &Vector::new(),
        )?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lablelme_loader::load_labelme;
    use anyhow::Result;
    use std::path::PathBuf;
    use crate::pipeline::PipelineTask;

    #[test]
    #[ignore]
    fn single_file_test() -> Result<()> {
        let path_buf = PathBuf::from("test-resources/IMG_20260123_232343.json");
        let image = load_labelme(&path_buf)?;
        let (tx, rx) = bounded(1);
        rayon::spawn(|| {
            let pipeline = pipeline::Pipeline::new(configuration::load_configuration());
            let task = PipelineTask{image, prefix: "test".to_string()};
            if let Err(e) = pipeline.run(task, tx) {
                error!("Error {:?}", e);
            }
        });
        save_to_disk(rx)?;
        Ok(())
    }
}
