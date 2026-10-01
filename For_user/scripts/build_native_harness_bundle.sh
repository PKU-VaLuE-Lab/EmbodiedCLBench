#!/usr/bin/env bash
set -euo pipefail

# Build a relocatable user-space harness bundle from the Docker images that
# define the current TongBench runtime.  Run this as a user with Docker access.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUTPUT_TAR="${OUTPUT_TAR-$REPO_ROOT/For_user/native_harness_bundle.tar.gz}"
OUTPUT_DIR="${OUTPUT_DIR:-}"
TMP_ROOT="${TMPDIR:-/tmp}"
STAGE="$(mktemp -d "$TMP_ROOT/tongbench-native-harnesses.XXXXXX")"

cleanup() {
  rm -rf "$STAGE"
}
trap cleanup EXIT

die() {
  echo "build_native_harness_bundle: $*" >&2
  exit 1
}

command -v docker >/dev/null 2>&1 || die "docker is required"
command -v tar >/dev/null 2>&1 || die "tar is required"
command -v sha256sum >/dev/null 2>&1 || die "sha256sum is required"

HERMES_IMAGE="${HERMES_IMAGE:-wildclawbench-hermes-agent:v0.5}"
CODEX_IMAGE="${CODEX_IMAGE:-wildclawbench-codex-ubuntu:v0.0}"
CLAUDE_IMAGE="${CLAUDE_IMAGE:-wildclawbench-claudecode-ubuntu:v0.2}"
OPENCLAW_IMAGE="${OPENCLAW_IMAGE:-wildclawbench-ubuntu:v1.3}"

copy_from_image() {
  local image="$1"
  local source="$2"
  local destination="$3"
  local container="tongbench-native-export-$$-$RANDOM"

  echo "[bundle] $image:$source -> $destination"
  docker create --name "$container" "$image" >/dev/null
  local status=0
  mkdir -p "$STAGE/$(dirname "$destination")"
  docker cp "$container:$source" "$STAGE/$destination" || status=$?
  docker rm "$container" >/dev/null || true
  if [[ "$status" -ne 0 ]]; then
    die "failed to copy $source from $image"
  fi
}

copy_dir_from_image() {
  local image="$1"
  local source="$2"
  local destination="$3"
  local container="tongbench-native-export-$$-$RANDOM"

  echo "[bundle] $image:$source/. -> $destination/"
  docker create --name "$container" "$image" >/dev/null
  local status=0
  mkdir -p "$STAGE/$destination"
  docker cp "$container:$source/." "$STAGE/$destination/" || status=$?
  docker rm "$container" >/dev/null || true
  if [[ "$status" -ne 0 ]]; then
    die "failed to copy $source from $image"
  fi
}

mkdir -p "$STAGE"/{bin,hermes,hermes_python,codex,claudecode,openclaw}

copy_dir_from_image "$HERMES_IMAGE" "/opt/hermes" "hermes"
copy_dir_from_image "$HERMES_IMAGE" "/root/.local/share/uv/python/cpython-3.12.13-linux-x86_64-gnu" "hermes_python"
copy_dir_from_image "$CODEX_IMAGE" "/usr/lib/node_modules/@openai/codex" "codex/node_modules/@openai/codex"
copy_from_image "$CODEX_IMAGE" "/usr/bin/node" "codex/bin/node"
copy_dir_from_image "$CLAUDE_IMAGE" "/claude_code" "claudecode/app"
copy_from_image "$CLAUDE_IMAGE" "/root/.bun/bin/bun" "claudecode/bun_home/bin/bun"
copy_dir_from_image "$OPENCLAW_IMAGE" "/usr/lib/node_modules/openclaw" "openclaw/node_modules/openclaw"
copy_from_image "$OPENCLAW_IMAGE" "/usr/bin/node" "openclaw/bin/node"

# The Hermes virtualenv points to the image's absolute uv-managed interpreter.
# Make that link relative so the bundle works after being moved to another
# user directory.
HERMES_VENV_BIN="$STAGE/hermes/.venv/bin"
if [[ -d "$HERMES_VENV_BIN" ]]; then
  ln -sfn "../../../hermes_python/bin/python3.12" "$HERMES_VENV_BIN/python"
  ln -sfn "python" "$HERMES_VENV_BIN/python3"
fi

cat > "$STAGE/bin/codex" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
BUNDLE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec "$BUNDLE_ROOT/codex/bin/node" \
  "$BUNDLE_ROOT/codex/node_modules/@openai/codex/bin/codex.js" "$@"
EOF

cat > "$STAGE/bin/openclaw" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
BUNDLE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec "$BUNDLE_ROOT/openclaw/bin/node" \
  "$BUNDLE_ROOT/openclaw/node_modules/openclaw/openclaw.mjs" "$@"
EOF

cat > "$STAGE/bin/bun" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
BUNDLE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec "$BUNDLE_ROOT/claudecode/bun_home/bin/bun" "$@"
EOF

cat > "$STAGE/bin/hermes" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
BUNDLE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$BUNDLE_ROOT/hermes${PYTHONPATH:+:$PYTHONPATH}"
exec "$BUNDLE_ROOT/hermes/.venv/bin/python3" -m hermes_cli.main "$@"
EOF

chmod -R a+rX "$STAGE"
chmod a+rx "$STAGE/bin"/*
# Claude Code ships its vendored ripgrep as a regular file in some image
# builds.  It is invoked as a binary at runtime, so make that permission
# explicit rather than relying on the source image's mode bits.
find "$STAGE/claudecode" -type f -path '*/vendor/*/rg' -exec chmod a+rx {} + 2>/dev/null || true

hermes_id="$(docker image inspect "$HERMES_IMAGE" --format '{{.Id}}')"
codex_id="$(docker image inspect "$CODEX_IMAGE" --format '{{.Id}}')"
claude_id="$(docker image inspect "$CLAUDE_IMAGE" --format '{{.Id}}')"
openclaw_id="$(docker image inspect "$OPENCLAW_IMAGE" --format '{{.Id}}')"
cat > "$STAGE/native_harness_manifest.json" <<EOF
{
  "format_version": 1,
  "platform": "$(uname -s)",
  "architecture": "$(uname -m)",
  "images": {
    "hermes": {"name": "$HERMES_IMAGE", "id": "$hermes_id"},
    "codex": {"name": "$CODEX_IMAGE", "id": "$codex_id"},
    "claudecode": {"name": "$CLAUDE_IMAGE", "id": "$claude_id"},
    "openclaw": {"name": "$OPENCLAW_IMAGE", "id": "$openclaw_id"}
  },
  "entrypoints": ["bin/hermes", "bin/codex", "bin/bun", "bin/openclaw"]
}
EOF

# Runtime images can contain local setup state (for example Claude Code's
# image-level .env).  Never carry that state into the portable bundle; API
# configuration is supplied by the caller at runtime.
while IFS= read -r -d '' credential_file; do
  echo "[bundle] excluding credential/setup file: ${credential_file#$STAGE/}"
  rm -f "$credential_file"
done < <(find "$STAGE" -type f \( -name .env -o -name auth.json -o -name credentials.json \) -print0)

# Do not hash the checksum file itself.  Including it makes every archive fail
# its own verification because writing the checksum list changes its bytes.
(cd "$STAGE" && find . -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS)

if [[ -n "$OUTPUT_DIR" ]]; then
  mkdir -p "$OUTPUT_DIR"
  if [[ -n "$(find "$OUTPUT_DIR" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
    die "output directory is not empty: $OUTPUT_DIR"
  fi
  cp -a "$STAGE/." "$OUTPUT_DIR/"
  echo "[bundle] directory: $OUTPUT_DIR"
fi

if [[ -n "$OUTPUT_TAR" ]]; then
  mkdir -p "$(dirname "$OUTPUT_TAR")"
  tar -C "$STAGE" -czf "$OUTPUT_TAR" .
  echo "[bundle] archive: $OUTPUT_TAR"
fi

du -sh "$STAGE"
