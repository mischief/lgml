rockspec_format = "3.0"
package = "lgml"
version = "scm-1"
source = {
	url = "git+https://git.offblast.org/mischief/lgml.git",
}
description = {
	summary = "Lua inference stack on ggml",
	license = "ISC",
	homepage = "https://git.offblast.org/mischief/lgml",
}
dependencies = {
	"lua >= 5.4",
	"luv",
	"lua-cjson",
	"picohttpparser",
	"imsg",
}
build = {
	type = "command",
	-- werror off: other compilers warn on other lines.
	build_command = "rm -rf build-rock && meson setup build-rock --prefix=$(PREFIX) --bindir=$(BINDIR) " ..
		"-Dlua_dir=$(LUADIR) -Dlua_cdir=$(LIBDIR) -Dwerror=false && meson compile -C build-rock",
	install_command = "meson install -C build-rock",
}
