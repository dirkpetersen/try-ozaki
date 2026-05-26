"""Generate the bash script that runs inside the GPU worker container.

The script reads source from the Run:ai S3 datasource mount path and writes
results back to the same mount — no AWS credentials required inside the container.
"""

# Template uses @PLACEHOLDER@ tokens to avoid conflicts with shell ${} syntax.
_TEMPLATE = """\
#!/bin/bash
set -euo pipefail

JOB_ID="@JOB_ID@"
MOUNT="@DATASOURCE_MOUNT@"
MOUNT_JOB_DIR="$MOUNT/@S3_PREFIX@/$JOB_ID"
WORKDIR="/tmp/ozaki_job"
SRCDIR="$WORKDIR/src"
OZIMMU_DIR="/opt/ozIMMU"
CMAKE_FLAGS="@CMAKE_FLAGS@"
# Write logs locally; sync to mount at key checkpoints and on exit
LOGFILE="/tmp/ozaki_job.log"

sync_results() {
    # Copy local outputs back to the S3 mount
    cp "$LOGFILE" "$MOUNT_JOB_DIR/job.log" 2>/dev/null || true
    [ -f "$WORKDIR/out_orig.txt"  ] && cp "$WORKDIR/out_orig.txt"  "$MOUNT_JOB_DIR/out_orig.txt"  2>/dev/null || true
    [ -f "$WORKDIR/out_ozaki.txt" ] && cp "$WORKDIR/out_ozaki.txt" "$MOUNT_JOB_DIR/out_ozaki.txt" 2>/dev/null || true
    [ -f "/tmp/timing.json"       ] && cp "/tmp/timing.json"       "$MOUNT_JOB_DIR/timing.json"   2>/dev/null || true
}
trap sync_results EXIT

log() {
    local msg="[try-ozaki $(date +%H:%M:%S)] $*"
    echo "$msg" | tee -a "$LOGFILE"
}

log "Job $JOB_ID starting on $(hostname)"
log "Mount: $MOUNT"
log "Mount job dir: $MOUNT_JOB_DIR"
ls -la "$MOUNT_JOB_DIR" 2>&1 | tee -a "$LOGFILE" || true

# ── Install build dependencies ─────────────────────────────────────────────
log "Installing build dependencies (apt)..."
# Use || true so apt repo errors (e.g. CUDA keyring) don't abort the job
apt-get update 2>&1 | tail -3 >> "$LOGFILE" || true
DEBIAN_FRONTEND=noninteractive apt-get install -y \\
    cmake gfortran g++ libopenblas-dev libopenblas-base git \\
    >> "$LOGFILE" 2>&1
log "apt done: cmake=$(cmake --version 2>/dev/null | head -1), gfortran=$(gfortran --version 2>/dev/null | head -1)"

# ── Clone and build ozIMMU ─────────────────────────────────────────────────
# ozIMMU requires CUDA dev headers; build is best-effort
if [ ! -f "/usr/local/lib/libozIMMU.so" ] && [ ! -f "/usr/local/lib/libozIMMU.a" ]; then
    log "Unpacking pre-bundled ozIMMU + cutf from mount..."
    # ozIMMU sources are bundled by the app host and uploaded to S3 alongside src.tar.gz
    # to avoid needing outbound internet access from the GPU worker container.
    if [ -f "$MOUNT_JOB_DIR/ozimmu.tar.gz" ]; then
        tar -xzf "$MOUNT_JOB_DIR/ozimmu.tar.gz" -C "$(dirname $OZIMMU_DIR)"
        log "ozIMMU unpacked from bundle."
    else
        log "WARNING: ozimmu.tar.gz not found in mount, trying git clone (may fail without internet)..."
        git clone --depth=1 @OZIMMU_REPO@ "$OZIMMU_DIR" >> "$LOGFILE" 2>&1 || { log "WARNING: ozIMMU clone failed."; }
        git clone --depth=1 https://github.com/enp1s0/cutf "$OZIMMU_DIR/src/cutf" >> "$LOGFILE" 2>&1 \
            || { log "WARNING: cutf clone failed."; }
    fi
    if [ -d "$OZIMMU_DIR" ]; then
        log "Building ozIMMU..."
        cmake -S "$OZIMMU_DIR" -B "$OZIMMU_DIR/build" \\
            -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/usr/local \\
            -DAUTO_SUBMODULE_UPDATE=OFF \\
            -DBUILD_TEST=OFF \\
            >> "$LOGFILE" 2>&1 \\
            && cmake --build "$OZIMMU_DIR/build" -j$(nproc) >> "$LOGFILE" 2>&1 \\
            && cmake --install "$OZIMMU_DIR/build" >> "$LOGFILE" 2>&1 \\
            && echo '/usr/local/cuda/lib64' > /etc/ld.so.conf.d/cuda-ozaki.conf \\
            && echo '/usr/local/lib' >> /etc/ld.so.conf.d/cuda-ozaki.conf \\
            && ldconfig >> "$LOGFILE" 2>&1 \\
            && log "ozIMMU installed and ldconfig updated." \\
            || log "WARNING: ozIMMU build failed (Ozaki binary will not be available)."
    fi
else
    log "ozIMMU already installed, skipping build."
fi

# ── Unpack sources from mount ──────────────────────────────────────────────
log "Unpacking sources..."
mkdir -p "$WORKDIR"
cp "$MOUNT_JOB_DIR/src.tar.gz" "$WORKDIR/src.tar.gz"
tar -xzf "$WORKDIR/src.tar.gz" -C "$WORKDIR"
log "Source tree:"
find "$WORKDIR/src" -type f | tee -a "$LOGFILE"

# ── Build ORIGINAL (FP64 baseline) ────────────────────────────────────────
log "Building original (FP64) binary..."
mkdir -p "$WORKDIR/build_orig"
cmake -S "$SRCDIR/original" -B "$WORKDIR/build_orig" \\
    -DCMAKE_BUILD_TYPE=Release $CMAKE_FLAGS >> "$LOGFILE" 2>&1 \\
    && cmake --build "$WORKDIR/build_orig" -j$(nproc) >> "$LOGFILE" 2>&1 \\
    && log "Original build OK." \\
    || { log "ERROR: original build failed."; cat "$LOGFILE"; exit 1; }

ORIG_BIN=$(find "$WORKDIR/build_orig" -maxdepth 3 -type f -executable | head -1)
log "Original binary: $ORIG_BIN"

# ── Build OZAKI (rewritten) ───────────────────────────────────────────────
log "Building Ozaki-rewritten binary..."
mkdir -p "$WORKDIR/build_ozaki"
cmake -S "$SRCDIR/ozaki" -B "$WORKDIR/build_ozaki" \\
    -DCMAKE_BUILD_TYPE=Release $CMAKE_FLAGS \\
    -DUSE_OZAKI=ON -DOZIMMU_DIR=/usr/local \\
    -DCMAKE_EXE_LINKER_FLAGS="-L/usr/local/cuda/lib64 -L/usr/local/lib -Wl,-rpath,/usr/local/cuda/lib64:/usr/local/lib" >> "$LOGFILE" 2>&1 \\
    && cmake --build "$WORKDIR/build_ozaki" -j$(nproc) >> "$LOGFILE" 2>&1 \\
    && log "Ozaki build OK." \\
    || log "WARNING: Ozaki build failed (ozIMMU may not support this code yet)."

# Find the built executable, explicitly excluding CMake internal helpers
OZAKI_BIN=$(find "$WORKDIR/build_ozaki" -maxdepth 2 -type f -executable ! -path "*/CMakeFiles/*" 2>/dev/null | head -1)
log "Ozaki binary: ${OZAKI_BIN:-NONE}"

# ── Run ORIGINAL ──────────────────────────────────────────────────────────
log "Running original binary..."
T_ORIG_START=$(date +%s%N)
"$ORIG_BIN" > "$WORKDIR/out_orig.txt" 2>&1
T_ORIG_END=$(date +%s%N)
ELAPSED_ORIG=$(( (T_ORIG_END - T_ORIG_START) / 1000000 ))
log "Original finished in ${ELAPSED_ORIG}ms"
log "Original output:"
cat "$WORKDIR/out_orig.txt" | tee -a "$LOGFILE"
sync_results

# ── Run OZAKI ─────────────────────────────────────────────────────────────
if [ -n "${OZAKI_BIN:-}" ] && [ -f "$OZAKI_BIN" ]; then
    log "Running Ozaki binary..."
    T_OZ_START=$(date +%s%N)
    "$OZAKI_BIN" > "$WORKDIR/out_ozaki.txt" 2>&1
    T_OZ_END=$(date +%s%N)
    ELAPSED_OZAKI=$(( (T_OZ_END - T_OZ_START) / 1000000 ))
    log "Ozaki finished in ${ELAPSED_OZAKI}ms"
    log "Ozaki output:"
    cat "$WORKDIR/out_ozaki.txt" | tee -a "$LOGFILE"
else
    ELAPSED_OZAKI=0
    echo "Ozaki binary not available." > "$WORKDIR/out_ozaki.txt"
    log "Skipping Ozaki run (build failed or ozIMMU not available)."
fi

# ── Write timing JSON (pure bash, no python3 dependency) ──────────────────
printf '{"job_id":"%s","elapsed_orig_ms":%d,"elapsed_ozaki_ms":%d}\\n' \\
    "@JOB_ID@" "$ELAPSED_ORIG" "$ELAPSED_OZAKI" > "/tmp/timing.json"

log "Job complete. Syncing results to $MOUNT_JOB_DIR/"
sync_results
ls -la "$MOUNT_JOB_DIR/" 2>&1 | tee -a "$LOGFILE"
"""


def generate(
    job_id: str,
    s3_prefix: str,
    datasource_mount: str = "/mnt/runai-peterdir",
    cmake_flags: str = "",
    ozimmu_repo: str = "https://github.com/enp1s0/ozIMMU",
) -> str:
    return (
        _TEMPLATE
        .replace("@JOB_ID@", job_id)
        .replace("@S3_PREFIX@", s3_prefix)
        .replace("@DATASOURCE_MOUNT@", datasource_mount)
        .replace("@CMAKE_FLAGS@", cmake_flags)
        .replace("@OZIMMU_REPO@", ozimmu_repo)
    )
