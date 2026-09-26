#!/bin/sh
# Build the AFL++ harness in build-fuzz and fuzz targets in parallel.
# usage: fuzz/run.sh [seconds [target ...]]
#        fuzz/run.sh replay target file ...
# Crashes land in build-fuzz/out/<target>/default/crashes.
set -eu

src=$(cd "$(dirname "$0")/.." && pwd)
b=$src/build-fuzz
san=-fsanitize=address,undefined

# libpng, libjpeg-turbo and zlib build from the wraps, so AFL and the
# sanitizers see inside them.
if [ ! -d "$b" ]; then
	CC=afl-clang-fast meson setup "$b" "$src" -Dbuildtype=debugoptimized \
	    -Db_sanitize=address,undefined -Db_lundef=false \
	    --force-fallback-for=libpng,libjpeg,zlib
fi
ninja -C "$b" lfuzz

# The C rocks, built with the same instrumentation and sanitizers.
# The build hooks rock runs inside luarocks, so it stays plain.
[ -d "$b/rocks/lib/luarocks/rocks-5.4/luarocks-build-hooks" ] ||
    luarocks --lua-version 5.4 --tree="$b/rocks" install luarocks-build-hooks
for r in lua-cjson picohttpparser; do
	[ -d "$b/rocks/lib/luarocks/rocks-5.4/$r" ] ||
	    luarocks --lua-version 5.4 --tree="$b/rocks" install "$r" \
	    CC=afl-clang-fast LD=afl-clang-fast CFLAGS="-O2 -fPIC $san" \
	    LIBFLAG="-shared $san"
done

export LUA_PATH="$src/lua/?.lua;$src/lua/?/init.lua;$b/rocks/share/lua/5.4/?.lua;;"
export LUA_CPATH="$b/rocks/lib/lua/5.4/?.so;;"
# No Vulkan driver: the harness uses only the CPU backend.
export VK_DRIVER_FILES=/nonexistent VK_ICD_FILENAMES=/nonexistent
export AFL_SKIP_CPUFREQ=1 AFL_NO_UI=1 AFL_I_DONT_CARE_ABOUT_MISSING_CRASHES=1
# Room above the C edges, with libpng and libjpeg-turbo, for Lua line coverage.
export AFL_MAP_SIZE=262144

# Outside afl-fuzz, cjson loads after the fork server check.
if [ "${1:-}" = replay ]; then
	t=$2
	shift 2
	AFL_IGNORE_PROBLEMS=1 exec "$b/lfuzz" "$src/fuzz/$t.lua" "$@"
fi

secs=${1:-600}
[ $# -gt 0 ] && shift
targets=${*:-json chat http safetensors vocab wav tokenize image}

for t in $targets; do
	mkdir -p "$b/out"
	# Keep the last run of each target.
	rm -rf "$b/out/$t.prev"
	[ ! -d "$b/out/$t" ] || mv "$b/out/$t" "$b/out/$t.prev"
	nice -n 19 afl-fuzz -i "$src/fuzz/corpus/$t" -o "$b/out/$t" -x "$src/fuzz/dict/$t.dict" \
	    -V "$secs" -- "$b/lfuzz" "$src/fuzz/$t.lua" >"$b/out/$t.log" 2>&1 &
done
wait
for t in $targets; do
	n=$(ls "$b/out/$t/default/crashes" 2>/dev/null | grep -vc README || true)
	h=$(ls "$b/out/$t/default/hangs" 2>/dev/null | wc -l)
	e=$(awk -F': *' '/^execs_done/ {print $2}' "$b/out/$t/default/fuzzer_stats" 2>/dev/null)
	echo "$t: execs ${e:-0}, crashes $n, hangs $h"
done
