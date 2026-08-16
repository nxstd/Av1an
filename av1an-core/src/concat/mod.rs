#[cfg(test)]
mod tests;

use std::{
    fmt::{Display, Write as FmtWrite},
    fs::{self, DirEntry, File},
    io::{Read, Seek, SeekFrom, Write},
    path::{Path, PathBuf},
    process::{Command, Stdio},
};

use anyhow::{anyhow, Context};
use av_format::rational::Rational64;
use path_abs::{PathAbs, PathInfo};
use serde::{Deserialize, Serialize};
use tracing::{debug, error, warn};

use crate::{encoder::Encoder, util::read_in_dir};

#[derive(
    PartialEq,
    Eq,
    Copy,
    Clone,
    Serialize,
    Deserialize,
    Debug,
    strum::EnumString,
    strum::IntoStaticStr,
)]
pub enum ConcatMethod {
    #[strum(serialize = "mkvmerge")]
    MKVMerge,
    #[strum(serialize = "ffmpeg")]
    FFmpeg,
    #[strum(serialize = "ivf")]
    Ivf,
}

impl Display for ConcatMethod {
    #[inline]
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(<&'static str>::from(self))
    }
}

#[tracing::instrument(level = "debug")]
pub fn sort_files_by_filename(files: &mut [PathBuf]) {
    files.sort_unstable_by_key(|x| {
        // If the temp directory follows the expected format of 00000.ivf, 00001.ivf,
        // etc., then these unwraps will not fail
        x.file_stem()
            .expect("should have file stem")
            .to_string_lossy()
            .parse::<u32>()
            .expect("files should follow numeric pattern")
    });
}

#[derive(Debug)]
struct IvfHeader {
    bytes:  Vec<u8>,
    fourcc: [u8; 4],
    width:  u16,
    height: u16,
    rate:   u32,
    scale:  u32,
}

#[tracing::instrument(level = "debug")]
pub fn ivf(input: &Path, out: &Path) -> anyhow::Result<()> {
    let mut files: Vec<PathBuf> = read_in_dir(input)?.collect();

    sort_files_by_filename(&mut files);
    anyhow::ensure!(
        !files.is_empty(),
        "No IVF chunks found in {}",
        input.display()
    );

    let result = (|| {
        let mut first_input = File::open(&files[0])
            .with_context(|| format!("Failed to open IVF chunk {}", files[0].display()))?;
        let first_header = read_ivf_header(&mut first_input, &files[0])?;
        let mut output = File::create(out)
            .with_context(|| format!("Failed to create IVF output {}", out.display()))?;
        output.write_all(&first_header.bytes)?;

        let mut frame_count = 0u64;
        let mut pos_offset = 0u64;
        for (file_index, file) in files.iter().enumerate() {
            let mut input = File::open(file)
                .with_context(|| format!("Failed to open IVF chunk {}", file.display()))?;
            let header = read_ivf_header(&mut input, file)?;
            if file_index != 0 {
                validate_compatible_ivf_header(&first_header, &header, file)?;
            }

            let mut last_local_timestamp = 0u64;
            let mut frame_index = 0u64;
            while let Some(frame_header) = read_ivf_frame_header(&mut input, file)? {
                let payload_size =
                    u32::from_le_bytes(frame_header[..4].try_into().expect("slice length"));
                let local_timestamp =
                    u64::from_le_bytes(frame_header[4..].try_into().expect("slice length"));
                let output_timestamp =
                    local_timestamp.checked_add(pos_offset).ok_or_else(|| {
                        anyhow!(
                            "IVF timestamp overflow in {} at frame {}",
                            file.display(),
                            frame_index
                        )
                    })?;

                output.write_all(&payload_size.to_le_bytes())?;
                output.write_all(&output_timestamp.to_le_bytes())?;
                copy_ivf_payload(&mut input, &mut output, payload_size, file, frame_index)?;

                last_local_timestamp = local_timestamp;
                frame_index += 1;
                frame_count = frame_count
                    .checked_add(1)
                    .ok_or_else(|| anyhow!("IVF frame count overflow"))?;
            }
            pos_offset = pos_offset
                .checked_add(last_local_timestamp)
                .and_then(|value| value.checked_add(1))
                .ok_or_else(|| anyhow!("IVF timestamp offset overflow after {}", file.display()))?;
        }

        let frame_count =
            u32::try_from(frame_count).map_err(|_| anyhow!("IVF frame count exceeds u32"))?;
        output.seek(SeekFrom::Start(24))?;
        output.write_all(&frame_count.to_le_bytes())?;
        output.flush()?;
        Ok(())
    })();

    if result.is_err() {
        let _ = fs::remove_file(out);
    }
    result
}

fn read_ivf_header(reader: &mut File, path: &Path) -> anyhow::Result<IvfHeader> {
    let mut fixed_header = [0u8; 32];
    reader
        .read_exact(&mut fixed_header)
        .with_context(|| format!("IVF header shorter than 32 bytes in {}", path.display()))?;
    anyhow::ensure!(
        &fixed_header[..4] == b"DKIF",
        "Invalid IVF magic in {}",
        path.display()
    );

    let header_length = usize::from(u16::from_le_bytes(
        fixed_header[6..8].try_into().expect("slice length"),
    ));
    anyhow::ensure!(
        header_length >= fixed_header.len(),
        "Invalid IVF header length {} in {}",
        header_length,
        path.display()
    );

    let mut bytes = vec![0; header_length];
    bytes[..fixed_header.len()].copy_from_slice(&fixed_header);
    reader
        .read_exact(&mut bytes[fixed_header.len()..])
        .with_context(|| format!("Truncated extended IVF header in {}", path.display()))?;

    Ok(IvfHeader {
        fourcc: fixed_header[8..12].try_into().expect("slice length"),
        width: u16::from_le_bytes(fixed_header[12..14].try_into().expect("slice length")),
        height: u16::from_le_bytes(fixed_header[14..16].try_into().expect("slice length")),
        rate: u32::from_le_bytes(fixed_header[16..20].try_into().expect("slice length")),
        scale: u32::from_le_bytes(fixed_header[20..24].try_into().expect("slice length")),
        bytes,
    })
}

fn validate_compatible_ivf_header(
    first: &IvfHeader,
    current: &IvfHeader,
    path: &Path,
) -> anyhow::Result<()> {
    anyhow::ensure!(
        first.fourcc == current.fourcc,
        "Incompatible IVF FourCC in {}",
        path.display()
    );
    anyhow::ensure!(
        first.width == current.width && first.height == current.height,
        "Incompatible IVF dimensions in {}",
        path.display()
    );
    anyhow::ensure!(
        first.rate == current.rate && first.scale == current.scale,
        "Incompatible IVF rate/scale in {}",
        path.display()
    );
    Ok(())
}

fn read_ivf_frame_header(reader: &mut File, path: &Path) -> anyhow::Result<Option<[u8; 12]>> {
    let mut header = [0u8; 12];
    let mut read = 0;
    while read < header.len() {
        let bytes_read = reader
            .read(&mut header[read..])
            .with_context(|| format!("Failed to read IVF frame header in {}", path.display()))?;
        if bytes_read == 0 {
            if read == 0 {
                return Ok(None);
            }
            anyhow::bail!(
                "Truncated IVF frame header in {}: got {}/{} bytes",
                path.display(),
                read,
                header.len()
            );
        }
        read += bytes_read;
    }
    Ok(Some(header))
}

fn copy_ivf_payload(
    input: &mut File,
    output: &mut File,
    payload_size: u32,
    path: &Path,
    frame_index: u64,
) -> anyhow::Result<()> {
    let mut remaining = u64::from(payload_size);
    let mut buffer = [0u8; 64 * 1024];
    while remaining > 0 {
        let read_len =
            usize::try_from(remaining.min(buffer.len() as u64)).expect("buffer length fits usize");
        let bytes_read = input
            .read(&mut buffer[..read_len])
            .with_context(|| format!("Failed to read IVF payload in {}", path.display()))?;
        if bytes_read == 0 {
            anyhow::bail!(
                "Truncated IVF payload in {} at frame {}: expected {} bytes",
                path.display(),
                frame_index,
                payload_size
            );
        }
        output.write_all(&buffer[..bytes_read])?;
        remaining -= bytes_read as u64;
    }
    Ok(())
}

#[tracing::instrument(level = "debug")]
fn read_encoded_chunks(encode_dir: &Path) -> anyhow::Result<Vec<DirEntry>> {
    Ok(fs::read_dir(encode_dir)
        .with_context(|| {
            format!(
                "Failed to read encoded chunks from {}",
                encode_dir.display()
            )
        })?
        .collect::<Result<Vec<_>, _>>()?)
}

#[tracing::instrument(level = "debug")]
pub fn mkvmerge(
    temp_dir: &Path,
    output: &Path,
    encoder: Encoder,
    num_chunks: usize,
    output_fps: Option<Rational64>,
) -> anyhow::Result<()> {
    #[cfg(windows)]
    const MAXIMUM_CHUNKS_PER_MERGE: usize = usize::MAX;
    #[cfg(not(windows))]
    const MAXIMUM_CHUNKS_PER_MERGE: usize = 960;

    // mkvmerge does not accept UNC paths on Windows
    #[cfg(windows)]
    fn fix_path<P: AsRef<Path>>(p: P) -> String {
        const UNC_PREFIX: &str = r#"\\?\"#;

        let p = p.as_ref().display().to_string();
        p.strip_prefix(UNC_PREFIX).map_or_else(
            || p.clone(),
            |path| {
                path.strip_prefix("UNC")
                    .map_or_else(|| path.to_string(), |p2| format!("\\{p2}"))
            },
        )
    }

    #[cfg(not(windows))]
    fn fix_path<P: AsRef<Path>>(p: P) -> String {
        p.as_ref().display().to_string()
    }

    let audio_file = PathBuf::from(&temp_dir).join("audio.mkv");
    let audio_file = PathAbs::new(&audio_file)?;
    let audio_file = audio_file.as_path().exists().then(|| fix_path(audio_file));

    let encode_dir = PathBuf::from(temp_dir).join("encode");

    let output = PathAbs::new(output)?;

    assert!(num_chunks != 0);

    let num_chunk_groups = (num_chunks as f64 / MAXIMUM_CHUNKS_PER_MERGE as f64).ceil() as usize;
    let chunk_groups: Vec<Vec<String>> = (0..num_chunk_groups)
        .map(|group_index| {
            let start = group_index * MAXIMUM_CHUNKS_PER_MERGE;
            let end = (start + MAXIMUM_CHUNKS_PER_MERGE).min(num_chunks);
            (start..end)
                .map(|i| {
                    format!(
                        "{i:05}.{ext}",
                        ext = match encoder {
                            Encoder::x264 => "264",
                            Encoder::x265 => "hevc",
                            _ => "ivf",
                        }
                    )
                })
                .collect()
        })
        .collect();

    // If there is only one chunk group, we can skip the intermediate merge/file
    // creation
    if chunk_groups.len() == 1 {
        let options_path = PathBuf::from(&temp_dir).join("options.json");
        let options_json_contents = mkvmerge_options_json(
            &chunk_groups[0],
            &fix_path(output.to_string_lossy().as_ref()),
            audio_file.as_deref(),
            output_fps,
        );

        let mut options_json = File::create(options_path)?;
        options_json.write_all(options_json_contents?.as_bytes())?;

        let mut cmd = Command::new("mkvmerge");
        cmd.current_dir(&encode_dir);
        cmd.arg("@../options.json");

        let out = cmd
            .output()
            .with_context(|| "Failed to execute mkvmerge command for concatenation")?;

        if !out.status.success() {
            error!(
                "mkvmerge concatenation failed with output: {:#?}\ncommand: {:?}",
                out, cmd
            );
            return Err(anyhow!("mkvmerge concatenation failed"));
        }

        return Ok(());
    }

    chunk_groups.iter().enumerate().try_for_each(|(group_index, chunk_group)| {
        let group_options_path =
            PathBuf::from(&temp_dir).join(format!("group_options_{group_index:05}.json"));
        let group_options_output_path = PathAbs::new(
            PathBuf::from(&temp_dir).join(format!("group_output_{group_index:05}.mkv")),
        )?;

        let group_options_json_contents = mkvmerge_options_json(
            chunk_group,
            &fix_path(group_options_output_path.to_string_lossy().as_ref()),
            None,
            output_fps,
        );

        let mut group_options_json = File::create(group_options_path)?;
        group_options_json.write_all(group_options_json_contents?.as_bytes())?;

        let mut group_cmd = Command::new("mkvmerge");
        group_cmd.current_dir(&encode_dir);
        group_cmd.arg(format!("@../group_options_{group_index:05}.json"));

        let group_out = group_cmd
            .output()
            .with_context(|| "Failed to execute mkvmerge command for concatenation")?;

        if !group_out.status.success() {
            return Err(anyhow::Error::msg(format!(
                "Failed to execute mkvmerge command for concatenation: {}",
                String::from_utf8_lossy(&group_out.stderr)
            )));
        }

        Ok(())
    })?;

    let chunk_group_options_names: Vec<String> = (0..num_chunk_groups)
        .map(|group_index| format!("group_output_{group_index:05}.mkv"))
        .collect();

    let options_path = PathBuf::from(&temp_dir).join("options.json");
    let options_json_contents = mkvmerge_options_json(
        &chunk_group_options_names,
        &fix_path(output.to_string_lossy().as_ref()),
        audio_file.as_deref(),
        output_fps,
    );

    let mut options_json = File::create(options_path)?;
    options_json.write_all(options_json_contents?.as_bytes())?;

    let mut cmd = Command::new("mkvmerge");
    cmd.current_dir(temp_dir);
    cmd.arg("@./options.json");

    let out = cmd
        .output()
        .with_context(|| "Failed to execute mkvmerge command for concatenation")?;

    if !out.status.success() {
        // TODO: make an EncoderCrash-like struct, but without all the other fields so
        // it can be used in a more broad scope than just for the pipe/encoder
        error!(
            "mkvmerge concatenation failed with output: {:#?}\ncommand: {:?}",
            out, cmd
        );
        return Err(anyhow!("mkvmerge concatenation failed"));
    }

    Ok(())
}

/// Create mkvmerge options.json
#[tracing::instrument(level = "debug")]
pub fn mkvmerge_options_json(
    chunks: &[String],
    output: &str,
    audio: Option<&str>,
    output_fps: Option<Rational64>,
) -> anyhow::Result<String> {
    let mut file_string = String::with_capacity(
        64 + output.len()
            + audio.map_or(0, |a| a.len() + 2)
            + chunks.iter().map(|s| s.len() + 4).sum::<usize>(),
    );
    write!(file_string, "[\"-o\", {output:?}")?;
    if let Some(audio) = audio {
        write!(file_string, ", {audio:?}")?;
    }
    if let Some(output_fps) = output_fps {
        write!(
            file_string,
            ", \"--default-duration\", \"0:{}/{}fps\", \"[\"",
            output_fps.numer(),
            output_fps.denom()
        )?;
    } else {
        file_string.push_str(", \"[\"");
    }
    for chunk in chunks {
        write!(file_string, ", \"{chunk}\"")?;
    }
    file_string.push_str(",\"]\"]");

    Ok(file_string)
}

/// Concatenates using ffmpeg (does not work with x265, and may have incorrect
/// FPS with vpx)
#[tracing::instrument(level = "debug")]
pub fn ffmpeg(temp: &Path, output: &Path) -> anyhow::Result<()> {
    fn write_concat_file(temp_folder: &Path) -> anyhow::Result<()> {
        let concat_file = temp_folder.join("concat");
        let encode_folder = temp_folder.join("encode");

        let mut files = read_encoded_chunks(&encode_folder)?;

        files.sort_by_key(DirEntry::path);

        let mut contents = String::with_capacity(24 * files.len());

        for i in files {
            writeln!(
                contents,
                "file {}",
                format!("{path}", path = i.path().display())
                    .replace('\\', r"\\")
                    .replace(' ', r"\ ")
                    .replace('\'', r"\'")
            )?;
        }

        let mut file = File::create(concat_file)?;
        file.write_all(contents.as_bytes())?;

        Ok(())
    }

    let temp = PathAbs::new(temp)?;
    let temp = temp.as_path();

    let concat = temp.join("concat");
    let concat_file = concat.to_string_lossy();

    write_concat_file(temp)?;

    let audio_file = {
        let file = temp.join("audio.mkv");
        (file.exists() && file.metadata().expect("file should have metadata").len() > 1000)
            .then_some(file)
    };

    let mut cmd = Command::new("ffmpeg");

    cmd.stdout(Stdio::piped());
    cmd.stderr(Stdio::piped());

    if let Some(file) = audio_file {
        cmd.args([
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            &concat_file,
            "-i",
        ])
        .arg(file)
        .args(["-map", "0", "-map", "1", "-c", "copy"])
        .arg(output);
    } else {
        cmd.args([
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            &concat_file,
        ])
        .args(["-map", "0", "-c", "copy"])
        .arg(output);
    }

    debug!("FFmpeg concat command: {:?}", cmd);

    let out = cmd
        .output()
        .with_context(|| "Failed to execute FFmpeg command for concatenation")?;

    if !out.status.success() {
        error!(
            "FFmpeg concatenation failed with output: {:#?}\ncommand: {:?}",
            out, cmd
        );
        return Err(anyhow!("FFmpeg concatenation failed"));
    }

    Ok(())
}
