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
	build_command = "meson setup build-rock --prefix=$(PREFIX) --buildtype=release && meson compile -C build-rock",
	install_command = "meson install -C build-rock",
}
