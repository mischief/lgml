#!/bin/sh
# Run the tests under luacov and write build/luacov.report.out.
# Usage: scripts/coverage.sh [meson test args...]
set -e
top=$(cd "$(dirname "$0")/.." && pwd)
b=$top/build
lr=$(luarocks path --lr-path)
rm -rf "$b/luacov" "$b/luacov.stats.out"
mkdir -p "$b/luacov"
meson test -C "$b" --setup coverage "$@" || echo "coverage: some tests failed" >&2
LUA_PATH="$lr;;" lua5.4 "$top/scripts/luacov_merge.lua" "$b/luacov.stats.out" "$b"/luacov/stats.*
cd "$b"
LUA_PATH="$top/lua/?.lua;$top/lua/?/init.lua;$lr;;" LUACOV_REPORT="$b/luacov.report.out" \
    luacov -c "$top/.luacov"
sed -n '/^Summary/,$p' luacov.report.out | sed "s|$top/||" | tail -n 1
