--- Image preprocessing for the vision encoders, as llama.cpp mtmd.
-- The sizes use f32 arithmetic, as the C++ code does, so edge cases
-- round the same way.
-- @module lgml.vision.preprocess
local lgml = require("lgml")

local M = {}

-- Round x to the nearest f32.
local function f32(x)
	return (string.unpack("<f", string.pack("<f", x)))
end

-- C++ std::round: halves go away from zero.
local function round(x)
	return x >= 0 and math.floor(x + 0.5) or -math.floor(-x + 0.5)
end

--- Find the resized size that keeps the aspect ratio.
-- Each side becomes a multiple of align, and the pixel count stays
-- within min_pixels and max_pixels, as "smart resize" in transformers.
-- @tparam integer w the image width
-- @tparam integer h the image height
-- @tparam integer align the side multiple (patch size times merge)
-- @tparam integer min_pixels the smallest pixel count
-- @tparam integer max_pixels the largest pixel count
-- @treturn integer the new width
-- @treturn integer the new height
function M.smart_size(w, h, align, min_pixels, max_pixels)
	local fa = f32(align)
	local function by(fn)
		return function(x)
			return math.tointeger(fn(f32(x / fa))) * align
		end
	end
	local rnd, ceil, floor = by(round), by(math.ceil), by(math.floor)
	local hb = math.max(align, rnd(h))
	local wb = math.max(align, rnd(w))
	if hb * wb > max_pixels then
		local beta = f32(math.sqrt(f32(f32(h * w) / f32(max_pixels))))
		hb = math.max(align, floor(f32(h / beta)))
		wb = math.max(align, floor(f32(w / beta)))
	elseif hb * wb < min_pixels then
		local beta = f32(math.sqrt(f32(f32(min_pixels) / f32(h * w))))
		hb = ceil(f32(h * beta))
		wb = ceil(f32(w * beta))
	end
	return wb, hb
end

--- Find where an image goes in a canvas when it keeps its aspect ratio.
-- It scales to touch the canvas on one axis, rounds sides up, and sits
-- in the center, as llama.cpp's PAD_CEIL resize.
-- @tparam integer w the image width
-- @tparam integer h the image height
-- @tparam integer tw the canvas width
-- @tparam integer th the canvas height
-- @treturn integer the scaled width
-- @treturn integer the scaled height
-- @treturn integer the left column
-- @treturn integer the top row
function M.fit(w, h, tw, th)
	local s = math.min(f32(tw / w), f32(th / h))
	local nw = math.min(math.tointeger(math.ceil(f32(w * s))), tw)
	local nh = math.min(math.tointeger(math.ceil(f32(h * s))), th)
	return nw, nh, (tw - nw) // 2, (th - nh) // 2
end

--- Decode an image and give it the size a vision encoder takes.
-- The image becomes RGB (alpha dropped). With `opts.fixed`, it has
-- exactly that size; else `smart_size` picks the size. When the size
-- changes, a bilinear resize and `fit` place it on a black canvas.
-- @tparam string|userdata img file bytes (PNG or JPEG) or an `lgml.image`
-- @tparam table opts options:
--
-- - `align`: the side multiple
-- - `min_pixels`, `max_pixels`: the pixel count range
-- - `fixed`: `{ w, h }` for a fixed size (default nil)
-- @treturn[1] userdata the RGB `lgml.image`
-- @treturn[2] nil
-- @treturn[2] string an error message
function M.prepare(img, opts)
	if type(img) == "string" then
		local err
		img, err = lgml.image(img)
		if not img then
			return nil, err
		end
	end
	local rgb <close> = img:rgb()
	local tw, th
	if opts.fixed then
		tw, th = opts.fixed[1], opts.fixed[2]
	else
		tw, th = M.smart_size(rgb.width, rgb.height, opts.align, opts.min_pixels,
		    opts.max_pixels)
	end
	if tw == rgb.width and th == rgb.height then
		return rgb:crop(0, 0, tw, th)
	end
	local nw, nh, x, y = M.fit(rgb.width, rgb.height, tw, th)
	local small <close> = rgb:resize(nw, nh, "bilinear")
	return small:place(tw, th, x, y)
end

return M
