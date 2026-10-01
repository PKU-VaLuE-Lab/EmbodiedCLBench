#!/usr/bin/env bash
set -euo pipefail

# Keep the historical entry point in sync with the maintained implementation.
exec "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/scripts/run_stage0_smoke.sh" "$@"
