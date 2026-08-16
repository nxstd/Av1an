use std::{
    fmt::{Debug, Display},
    fs::File,
    io::Write,
    path::Path,
    process::ExitStatus,
    sync::{
        atomic::{AtomicU8, Ordering},
        mpsc::Sender,
        Arc,
    },
    thread::available_parallelism,
};

use anyhow::bail;
use cfg_if::cfg_if;
use smallvec::SmallVec;
use thiserror::Error;
use tracing::{debug, error, warn};

use crate::{
    context::Av1anContext,
    ffmpeg::get_num_frames,
    finish_progress_bar,
    get_done,
    progress_bar::{
        dec_bar,
        inc_progress_bar_for_verbosity,
        update_mp_chunk,
        update_progress_bar_estimates,
        update_worker_progress_msg,
    },
    util::printable_base10_digits,
    target_quality::validate_full_rate_probe_frame_count,
    Chunk,
    DoneChunk,
    Instant,
};

#[derive(Debug)]
pub struct Broker<'a> {
    pub chunk_queue: Vec<Chunk>,
    pub project:     &'a Av1anContext,
}

#[derive(Clone)]
pub enum StringOrBytes {
    String(String),
    Bytes(Vec<u8>),
}

impl Debug for StringOrBytes {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::String(s) => {
                if f.alternate() {
                    f.write_str(&textwrap::indent(s, "        "))?; // 8 spaces
                } else {
                    f.write_str(s)?;
                }
            },
            Self::Bytes(b) => write!(f, "raw bytes: {b:?}")?,
        }

        Ok(())
    }
}

impl From<Vec<u8>> for StringOrBytes {
    fn from(bytes: Vec<u8>) -> Self {
        #[expect(
            clippy::option_if_let_else,
            reason = "https://github.com/rust-lang/rust-clippy/issues/15142"
        )]
        if let Ok(res) = simdutf8::basic::from_utf8(&bytes) {
            Self::String(res.to_string())
        } else {
            Self::Bytes(bytes)
        }
    }
}

impl From<String> for StringOrBytes {
    fn from(s: String) -> Self {
        Self::String(s)
    }
}

#[derive(Error, Debug)]
pub struct EncoderCrash {
    pub exit_status:        ExitStatus,
    pub stdout:             StringOrBytes,
    pub stderr:             StringOrBytes,
    pub source_pipe_stderr: StringOrBytes,
    pub ffmpeg_pipe_stderr: Option<StringOrBytes>,
}

impl Display for EncoderCrash {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "encoder crashed: {}\nstdout:\n{:#?}\nstderr:\n{:#?}\nsource pipe stderr:\n{:#?}",
            self.exit_status, self.stdout, self.stderr, self.source_pipe_stderr,
        )?;

        if let Some(ffmpeg_pipe_stderr) = &self.ffmpeg_pipe_stderr {
            write!(f, "\nffmpeg pipe stderr:\n{ffmpeg_pipe_stderr:#?}")?;
        }

        Ok(())
    }
}

fn is_hard_shutdown_requested(terminations_requested: &AtomicU8) -> bool {
    terminations_requested.load(Ordering::SeqCst) >= 2
}

fn can_finalize_target_quality(terminations_requested: &AtomicU8) -> bool {
    !is_hard_shutdown_requested(terminations_requested)
}

fn should_retry_target_quality(
    terminations_requested: &AtomicU8,
    current_try: usize,
    max_tries: usize,
) -> bool {
    can_finalize_target_quality(terminations_requested) && current_try < max_tries
}

fn target_quality_needs_probing(chunk: &Chunk) -> bool {
    chunk.target_quality.target.is_some() && chunk.tq_cq.is_none()
}

fn can_start_next_chunk(terminations_requested: &AtomicU8) -> bool {
    terminations_requested.load(Ordering::SeqCst) == 0
}

fn validate_reused_probe_frame_count(chunk: &Chunk, actual_frames: usize) -> anyhow::Result<()> {
    if chunk.target_quality.probing_rate == 1 && !chunk.ignore_frame_mismatch {
        validate_full_rate_probe_frame_count(chunk.frames(), actual_frames)?;
    }
    Ok(())
}

impl Broker<'_> {
    /// Main encoding loop. set_thread_affinity may be ignored if the value is
    /// invalid.
    #[tracing::instrument(skip(self))]
    #[allow(clippy::needless_pass_by_value)]
    pub fn encoding_loop(
        self,
        tx: Sender<()>,
        set_thread_affinity: Option<usize>,
        total_chunks: u32,
    ) -> anyhow::Result<()> {
        if !self.chunk_queue.is_empty() {
            let (sender, receiver) = crossbeam_channel::bounded(self.chunk_queue.len());

            for chunk in &self.chunk_queue {
                sender.send(chunk.clone())?;
            }
            drop(sender);

            crossbeam_utils::thread::scope(|s| {
                let terminations_requested = Arc::new(AtomicU8::new(0));
                let terminations_requested_clone = Arc::clone(&terminations_requested);
                ctrlc::set_handler(move || {
                    let count = terminations_requested_clone.fetch_add(1, Ordering::SeqCst) + 1;
                    if count == 1 {
                        error!("Shutting down. Waiting for current workers to finish...");
                    } else {
                        error!("Shutting down all workers...");
                    }
                })
                .expect("should set ctrlc handler");

                let consumers: Vec<_> = (0..self.project.args.workers)
                    .map(|idx| (receiver.clone(), &self, idx, Arc::clone(&terminations_requested)))
                    .map(|(rx, queue, worker_id, terminations_requested)| {
                        let tx = tx.clone();
                        s.spawn(move |_| {
                            cfg_if! {
                                if #[cfg(any(target_os = "linux", target_os = "windows"))] {
                                    if let Some(threads) = set_thread_affinity {
                                        if threads == 0 {
                                            warn!("Ignoring set_thread_affinity: Requested 0 threads");
                                        } else {
                                            match available_parallelism() {
                                                Ok(parallelism) => {
                                                    let available_threads = parallelism.get();
                                                    let mut cpu_set = SmallVec::<[usize; 16]>::new();
                                                    let start_thread = (threads * worker_id) % available_threads;
                                                    cpu_set.extend((start_thread..start_thread + threads).map(|t| t % available_threads));
                                                    if let Err(e) = affinity::set_thread_affinity(&cpu_set) {
                                                        warn!("Failed to set thread affinity for worker {worker_id}: {e}");
                                                    }
                                                },
                                                Err(e) => {
                                                    warn!("Failed to get thread count: {e}. Thread affinity will not be set");
                                                }
                                            }
                                        }
                                    }
                                }
                            }

                            while let Ok(mut chunk) = rx.recv() {
                                if can_start_next_chunk(&terminations_requested)
                                    && let Err(e) = queue.encode_chunk(
                                        &mut chunk,
                                        worker_id,
                                        &terminations_requested,
                                        total_chunks,
                                    )
                                {
                                    error!("[chunk {index}] {e}", index = chunk.index);
                                    tx.send(()).expect("should send successfully");
                                    return Err(());
                                }
                            }
                            Ok(())
                        })
                    })
                    .collect();
                for consumer in consumers {
                    consumer.join().expect("consumer should join successfully").ok();
                }

                if terminations_requested.load(Ordering::SeqCst) > 0 {
                    tx.send(()).expect("should send successfully");
                }
            })
            .expect("thread should spawn successfully");

            finish_progress_bar();
        }

        Ok(())
    }

    #[tracing::instrument(skip(self, chunk, terminations_requested), fields(chunk_index = format!("{:>05}", chunk.index)))]
    fn encode_chunk(
        &self,
        chunk: &mut Chunk,
        worker_id: usize,
        terminations_requested: &Arc<AtomicU8>,
        total_chunks: u32,
    ) -> anyhow::Result<()> {
        let st_time = Instant::now();

        // we display the index, so we need to subtract 1 to get the max index
        let padding = printable_base10_digits(self.chunk_queue.len() - 1) as usize;
        update_mp_chunk(worker_id, chunk.index, padding);

        if target_quality_needs_probing(chunk) {
            let (min, max) = chunk
                .target_quality
                .target
                .expect("Target Quality is configured when probing is required");
            update_worker_progress_msg(
                self.project.args.verbosity,
                worker_id,
                format!(
                    "Targeting {metric} Quality: {min}-{max}",
                    metric = chunk.target_quality.metric,
                    min = min,
                    max = max
                ),
            );
            for r#try in 1..=self.project.args.max_tries {
                let res = chunk.target_quality.per_shot_target_quality(
                    chunk,
                    Some(worker_id),
                    self.project.args.verbosity,
                    Some(terminations_requested),
                    self.project.args.vapoursynth_plugins,
                );
                match res {
                    Ok(cq) => {
                        chunk.tq_cq = Some(cq);
                        break;
                    },
                    Err(e) => {
                        if !should_retry_target_quality(
                            terminations_requested,
                            r#try,
                            self.project.args.max_tries,
                        ) {
                            if is_hard_shutdown_requested(terminations_requested) {
                                bail!(
                                    "Hard shutdown requested during Target Quality. Skipping \
                                     chunk {}",
                                    chunk.index
                                );
                            }
                            bail!(
                                "Target Quality failed after {} tries on chunk {}:\n{}",
                                r#try,
                                chunk.index,
                                e
                            );
                        }
                    },
                }
            }
        }

        if chunk.target_quality.target.is_some() {
            if !can_finalize_target_quality(terminations_requested) {
                bail!(
                    "Hard shutdown requested during Target Quality. Skipping chunk {}",
                    chunk.index
                );
            }

            if chunk.target_quality.params_copied
                && chunk.target_quality.probing_rate == 1
                && self.project.args.ffmpeg_filter_args.is_empty()
                && chunk.proxy.is_none()
                && let Some(optimal_q) = chunk.tq_cq
            {
                let extension = match self.project.args.encoder {
                    crate::encoder::Encoder::x264 => "264",
                    crate::encoder::Encoder::x265 => "hevc",
                    _ => "ivf",
                };
                let probe_file =
                    std::path::Path::new(&self.project.args.temp).join("split").join({
                        let q_str = crate::encoder::format_q(optimal_q);
                        format!("v_{:05}_{}.{}", chunk.index, q_str, extension)
                });

                if probe_file.exists() {
                    let reuse_validation = get_num_frames(&probe_file)
                        .and_then(|actual_frames| {
                            validate_reused_probe_frame_count(chunk, actual_frames)
                        });
                    if let Err(error) = reuse_validation {
                        warn!(
                            "Selected Target Quality probe for chunk {} is invalid; removing it \
                             and falling back to final encoding: {error}",
                            chunk.index
                        );
                        let _ = std::fs::remove_file(&probe_file);
                    } else {
                        let encode_dir =
                            std::path::Path::new(&self.project.args.temp).join("encode");
                        std::fs::create_dir_all(&encode_dir)?;
                        let output_file = encode_dir
                            .join(format!("{index:05}.{extension}", index = chunk.index));
                        std::fs::copy(&probe_file, &output_file)?;

                        inc_progress_bar_for_verbosity(
                            self.project.args.verbosity,
                            chunk.frames() as u64,
                        );

                        let progress_file = Path::new(&self.project.args.temp).join("done.json");
                        get_done().done.insert(chunk.name(), DoneChunk {
                            frames:     chunk.frames(),
                            size_bytes: output_file.metadata()?.len(),
                        });

                        let mut progress_file = File::create(progress_file)?;
                        progress_file.write_all(serde_json::to_string(get_done())?.as_bytes())?;

                        update_progress_bar_estimates(
                            chunk.frame_rate,
                            self.project.frames,
                            self.project.args.verbosity,
                            (get_done().done.len() as u32, total_chunks),
                        );

                        return Ok(());
                    }
                }
            }
        }

        if is_hard_shutdown_requested(terminations_requested) {
            bail!(
                "Hard shutdown requested after Target Quality. Skipping chunk {}",
                chunk.index
            );
        }

        // space padding at the beginning to align with "finished chunk"
        debug!(
            " started chunk {index:05}: {frames} frames",
            index = chunk.index,
            frames = chunk.frames()
        );

        let passes = chunk.passes;
        for current_pass in 1..=passes {
            for r#try in 1..=self.project.args.max_tries {
                let res = self.project.create_pipes(chunk, current_pass, worker_id, padding);
                if let Err((e, frames)) = res {
                    dec_bar(frames);

                    // If user presses CTRL+C more than once, do not let the worker finish
                    if terminations_requested.load(Ordering::SeqCst) > 1 {
                        bail!(
                            "Termination requested after Worker restart. Skipping chunk {}",
                            chunk.index
                        );
                    }

                    if r#try == self.project.args.max_tries {
                        bail!(
                            "[chunk {index}] encoder failed {tries} times, shutting down worker: \
                             {e}",
                            index = chunk.index,
                            tries = self.project.args.max_tries
                        );
                    }
                    // avoids double-print of the error message as both a WARN and ERROR,
                    // since `Broker::encoding_loop` will print the error message as well
                    warn!(
                        "Encoder failed (on chunk {index}):\n{e}",
                        index = chunk.index
                    );
                } else {
                    break;
                }
            }
        }

        let enc_time = st_time.elapsed();
        let fps = chunk.frames() as f64 / enc_time.as_secs_f64();

        let progress_file = Path::new(&self.project.args.temp).join("done.json");
        get_done().done.insert(chunk.name(), DoneChunk {
            frames:     chunk.frames(),
            size_bytes: Path::new(&chunk.output())
                .metadata()
                .expect("Unable to get size of finished chunk")
                .len(),
        });

        let mut progress_file = File::create(progress_file)?;
        progress_file.write_all(serde_json::to_string(get_done())?.as_bytes())?;

        update_progress_bar_estimates(
            chunk.frame_rate,
            self.project.frames,
            self.project.args.verbosity,
            (get_done().done.len() as u32, total_chunks),
        );

        debug!(
            "finished chunk {index:05}: {frames} frames, {fps:.2} fps, took {enc_time:.2?}",
            index = chunk.index,
            frames = chunk.frames()
        );

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::AtomicU8;

    use super::*;
    use crate::{vapoursynth, ChunkMethod, Encoder, Input, TargetQuality};

    fn target_quality_chunk(tq_cq: Option<f32>) -> Chunk {
        let mut target_quality = TargetQuality::default("/tmp", Encoder::svt_av1);
        target_quality.target = Some((95.0, 96.0));

        Chunk {
            temp: "/tmp".to_owned(),
            index: 12,
            input: Input::Video {
                path:         "test.mkv".into(),
                temp:         "/tmp".to_owned(),
                chunk_method: ChunkMethod::Select,
                is_proxy:     false,
                cache_mode:   vapoursynth::CacheSource::SOURCE,
            },
            proxy: None,
            source_cmd: vec![],
            proxy_cmd: None,
            output_ext: "ivf".to_owned(),
            start_frame: 0,
            end_frame: 5,
            frame_rate: 30.0,
            passes: 1,
            video_params: vec![],
            encoder: Encoder::svt_av1,
            noise_size: (None, None),
            target_quality,
            tq_cq,
            ignore_frame_mismatch: false,
        }
    }

    #[test]
    fn target_quality_retries_normal_errors_before_the_retry_limit() {
        let terminations_requested = AtomicU8::new(0);

        assert!(should_retry_target_quality(&terminations_requested, 1, 3));
        assert!(!should_retry_target_quality(&terminations_requested, 3, 3));
    }

    #[test]
    fn first_ctrl_c_allows_target_quality_retry() {
        let terminations_requested = AtomicU8::new(1);

        assert!(should_retry_target_quality(&terminations_requested, 1, 3));
    }

    #[test]
    fn second_ctrl_c_disables_target_quality_retry() {
        let terminations_requested = AtomicU8::new(2);

        assert!(!should_retry_target_quality(&terminations_requested, 1, 3));
    }

    #[test]
    fn first_ctrl_c_allows_target_quality_finalization() {
        let terminations_requested = AtomicU8::new(1);

        assert!(can_finalize_target_quality(&terminations_requested));
    }

    #[test]
    fn second_ctrl_c_prevents_target_quality_finalization() {
        let terminations_requested = AtomicU8::new(2);

        assert!(!can_finalize_target_quality(&terminations_requested));
    }

    #[test]
    fn first_ctrl_c_prevents_starting_another_queued_chunk() {
        let terminations_requested = AtomicU8::new(1);

        assert!(!can_start_next_chunk(&terminations_requested));
    }

    #[test]
    fn normal_operation_can_start_queued_chunks() {
        let terminations_requested = AtomicU8::new(0);

        assert!(can_start_next_chunk(&terminations_requested));
        assert!(can_finalize_target_quality(&terminations_requested));
    }

    #[test]
    fn valid_reused_full_rate_probe_is_accepted() {
        let chunk = target_quality_chunk(Some(42.0));

        assert!(validate_reused_probe_frame_count(&chunk, chunk.frames()).is_ok());
    }

    #[test]
    fn truncated_reused_probe_is_rejected_before_fast_reuse() {
        let chunk = target_quality_chunk(Some(42.0));

        assert!(validate_reused_probe_frame_count(&chunk, chunk.frames() - 1).is_err());
    }

    #[test]
    fn ignored_frame_mismatch_allows_reused_probe() {
        let mut chunk = target_quality_chunk(Some(42.0));
        chunk.ignore_frame_mismatch = true;

        assert!(validate_reused_probe_frame_count(&chunk, chunk.frames() - 1).is_ok());
    }

    #[test]
    fn fresh_target_quality_chunks_require_worker_side_probing() {
        assert!(target_quality_needs_probing(&target_quality_chunk(None)));
    }

    #[test]
    fn persisted_target_quality_cq_skips_worker_side_probing() {
        let mut old_chunk_json =
            serde_json::to_value(target_quality_chunk(None)).expect("chunk should serialize");
        old_chunk_json["per_shot_target_quality_cq"] = serde_json::json!(42.0);
        let chunk: Chunk =
            serde_json::from_value(old_chunk_json).expect("old chunk should deserialize");

        assert_eq!(chunk.tq_cq, Some(42.0));
        assert!(!target_quality_needs_probing(&chunk));
    }
}
