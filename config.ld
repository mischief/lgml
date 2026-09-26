-- LDoc configuration: ldoc . writes HTML to build/doc.
project = "lgml"
title = "lgml reference"
description = "Torch-shaped inference on ggml, driven from Lua."
file = { "src", "lua" }
dir = "build/doc"
format = "markdown"
all = false
no_space_before_args = true
sort_modules = true
-- The C files each declare module lgml; merge joins them into one page.
merge = true
