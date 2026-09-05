"""
Fast Psychedelic Tiling Visualizer
--------------------------------

Requirements:
    pip install numpy matplotlib

Run:
    python psychedelic_tiling_visualizer_fast.py

Performance changes:
- Default render resolution reduced to 384x384, then smoothly upscaled.
- Uses float32 coordinate arrays.
- Cached coordinate fields are rebuilt only when quality changes.
- Animation defaults to 18 FPS instead of ~30 FPS.
- Redraws are batched during reset/randomize.
- Quality presets let you trade speed for detail.

Keyboard:
    Space  pause/play
    S      save current image
"""

import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons
from matplotlib.colors import hsv_to_rgb


def triangle_wave(x):
    return 2.0 * np.abs(2.0 * (x - np.floor(x + 0.5))) - 1.0


def kaleido_angle(theta, sectors):
    sectors = max(1, int(sectors))
    wedge = np.float32(2.0 * np.pi / sectors)
    t = np.mod(theta, wedge)
    return np.minimum(t, wedge - t)


def normalize01(x):
    lo = float(x.min())
    hi = float(x.max())
    span = hi - lo
    if span < 1e-8:
        return np.zeros_like(x)
    return (x - lo) / span


class PsychedelicTileEngine:
    QUALITY = {
        "Fast": 288,
        "Balanced": 384,
        "Fine": 512,
        "Ultra": 640,
    }

    def __init__(self, quality="Balanced"):
        self.quality = quality
        self.resolution = None
        self.X = self.Y = self.R = self.TH = None
        self.set_quality(quality)

    def set_quality(self, quality):
        self.quality = quality
        resolution = self.QUALITY[quality]
        if resolution == self.resolution:
            return

        self.resolution = resolution
        lin = np.linspace(-1.0, 1.0, resolution, dtype=np.float32)
        self.X, self.Y = np.meshgrid(lin, lin)

        self.R = np.sqrt(self.X * self.X + self.Y * self.Y).astype(np.float32)
        self.R += np.float32(1e-7)
        self.TH = np.arctan2(self.Y, self.X).astype(np.float32)

    def render(self, p):
        X, Y, R, TH = self.X, self.Y, self.R, self.TH

        density = np.float32(p["density"])
        symmetry = int(p["symmetry"])
        warp = np.float32(p["warp"])
        hue_shift = np.float32(p["hue_shift"])
        color_speed = np.float32(p["color_speed"])
        contrast = np.float32(p["contrast"])
        spiral = np.float32(p["spiral"])
        pulse = np.float32(p["pulse"])
        detail = np.float32(p["detail"])
        glow = np.float32(p["glow"])
        tile_mix = np.float32(p["tile_mix"])
        t = np.float32(p["time"])

        folded = kaleido_angle(TH + spiral * R + np.float32(0.18) * t, symmetry)
        angular = folded * np.float32(symmetry)
        radial = R * density * np.float32(np.pi)

        wx = X + warp * (
            np.float32(0.35) * np.sin(angular * np.float32(3.0) + np.float32(1.7) * t)
            + np.float32(0.20) * np.cos(radial * np.float32(1.4) - np.float32(0.8) * t)
        )
        wy = Y + warp * (
            np.float32(0.35) * np.cos(angular * np.float32(2.0) - np.float32(1.1) * t)
            + np.float32(0.20) * np.sin(radial * np.float32(1.1) + np.float32(0.6) * t)
        )

        a = density * np.float32(2.4) + np.float32(0.2)
        sqx = wx * a
        sqy = wy * a

        q = np.float32(np.sqrt(3.0))
        hx = a * (wx + np.float32(0.5) * wy)
        hy = a * (q / np.float32(2.0)) * wy

        twopi = np.float32(2.0 * np.pi)

        square_tile = (
            np.cos(twopi * sqx)
            + np.cos(twopi * sqy)
            + np.float32(0.55) * np.cos(twopi * (sqx + sqy))
        ) / np.float32(2.55)

        hex_tile = (
            np.cos(twopi * hx)
            + np.cos(twopi * hy)
            + np.cos(twopi * (hx - hy))
        ) / np.float32(3.0)

        tile_field = (np.float32(1.0) - tile_mix) * square_tile + tile_mix * hex_tile

        rings = np.sin(radial * (np.float32(1.5) + detail) - pulse * t)
        petals = np.cos(
            angular * (np.float32(2.0) + detail * np.float32(3.0))
            + np.float32(0.7) * t
        )

        mode = p["mode"]

        if mode == "Mandala":
            base = (
                np.float32(0.55) * rings
                + np.float32(0.48) * petals
                + np.float32(0.38) * tile_field
            )

        elif mode == "Tunnel":
            base = (
                np.float32(0.68) * np.sin(
                    radial * (np.float32(2.0) + detail * np.float32(1.5))
                    - np.float32(1.8) * t
                )
                + np.float32(0.35) * np.cos(
                    angular * np.float32(3 + symmetry // 2) + t
                )
                + np.float32(0.27) * tile_field
            )

        elif mode == "Hex Bloom":
            base = (
                np.float32(0.60) * hex_tile
                + np.float32(0.38) * np.cos(
                    angular * np.float32(symmetry + 1) - pulse * t
                )
                + np.float32(0.42) * rings
            )

        else:
            recursive_like = np.sin(
                radial * (np.float32(2.4) + np.float32(1.3) * detail)
                + np.float32(1.8) * np.sin(
                    angular * (np.float32(2.0) + detail * np.float32(2.0)) - t
                )
            )

            wavefold = triangle_wave(
                np.float32(0.6) * radial / np.float32(np.pi)
                + np.float32(0.22) * petals
            )

            base = (
                np.float32(0.48) * recursive_like
                + np.float32(0.36) * wavefold
                + np.float32(0.32) * tile_field
                + np.float32(0.22) * petals
            )

        shaped = np.tanh(
            base
            * (np.float32(1.0) + np.float32(3.5) * contrast)
            * (np.float32(0.9) + np.float32(0.5) * contrast)
        )
        shaped01 = np.float32(0.5) + np.float32(0.5) * shaped

        glow_field = np.exp(-glow * np.float32(1.8) * R) * (
            np.float32(0.6)
            + np.float32(0.4) * np.cos(
                angular * np.float32(symmetry / 2.0) - t
            )
        )
        glow_field = normalize01(glow_field)

        hue = (
            hue_shift
            + color_speed * np.float32(0.07) * t
            + np.float32(0.20) * shaped01
            + np.float32(0.13) * np.sin(angular * np.float32(1.5) + np.float32(0.5) * t)
            + np.float32(0.10) * np.cos(radial * np.float32(0.9) - t)
        ) % np.float32(1.0)

        saturation = np.clip(
            np.float32(0.72)
            + np.float32(0.30) * np.sin(twopi * shaped01 + angular)
            + np.float32(0.16) * glow_field,
            0.0, 1.0
        )

        value = np.clip(
            np.float32(0.20)
            + np.float32(0.66) * shaped01
            + np.float32(0.52) * glow_field,
            0.0, 1.0
        )

        hsv = np.dstack((hue, saturation, value))
        return hsv_to_rgb(hsv)


class FastPsychedelicTilingApp:
    def __init__(self):
        self.engine = PsychedelicTileEngine("Balanced")
        self.animating = True
        self.start_time = time.perf_counter()
        self.frozen_time = 0.0
        self.suppress_updates = False
        self.render_busy = False

        self.params = {
            "mode": "Mandala",
            "density": 5.5,
            "symmetry": 8,
            "warp": 0.28,
            "hue_shift": 0.07,
            "color_speed": 1.0,
            "contrast": 0.70,
            "spiral": 1.15,
            "pulse": 1.2,
            "detail": 1.1,
            "glow": 1.2,
            "tile_mix": 0.45,
            "time": 0.0,
        }

        self.build_ui()
        self.redraw()
        self.install_timer()

    def current_time(self):
        if self.animating:
            return time.perf_counter() - self.start_time
        return self.frozen_time

    def build_ui(self):
        self.fig = plt.figure(figsize=(13, 9))
        self.fig.canvas.manager.set_window_title("Fast Psychedelic Tiling Visualizer")

        self.ax_img = self.fig.add_axes([0.045, 0.20, 0.62, 0.74])
        self.ax_img.set_xticks([])
        self.ax_img.set_yticks([])

        blank = np.zeros((self.engine.resolution, self.engine.resolution, 3), dtype=np.float32)
        self.im = self.ax_img.imshow(
            blank,
            interpolation="bilinear",
            origin="lower"
        )

        left, width, h, gap, top = 0.75, 0.20, 0.022, 0.033, 0.86
        axes = [self.fig.add_axes([left, top - i * gap, width, h]) for i in range(11)]

        self.s_density = Slider(axes[0], "Density", 1.0, 14.0, valinit=5.5, valstep=0.05)
        self.s_symmetry = Slider(axes[1], "Symmetry", 3, 24, valinit=8, valstep=1)
        self.s_warp = Slider(axes[2], "Warp", 0.0, 1.2, valinit=0.28, valstep=0.01)
        self.s_hue = Slider(axes[3], "Hue shift", 0.0, 1.0, valinit=0.07, valstep=0.001)
        self.s_color = Slider(axes[4], "Color speed", 0.0, 4.0, valinit=1.0, valstep=0.01)
        self.s_contrast = Slider(axes[5], "Contrast", 0.0, 1.5, valinit=0.70, valstep=0.01)
        self.s_spiral = Slider(axes[6], "Spiral", -4.0, 4.0, valinit=1.15, valstep=0.01)
        self.s_pulse = Slider(axes[7], "Pulse", 0.0, 4.0, valinit=1.2, valstep=0.01)
        self.s_detail = Slider(axes[8], "Detail", 0.0, 3.0, valinit=1.1, valstep=0.01)
        self.s_glow = Slider(axes[9], "Glow", 0.1, 4.0, valinit=1.2, valstep=0.01)
        self.s_tile = Slider(axes[10], "Tile mix", 0.0, 1.0, valinit=0.45, valstep=0.01)

        self.all_sliders = [
            self.s_density, self.s_symmetry, self.s_warp, self.s_hue,
            self.s_color, self.s_contrast, self.s_spiral, self.s_pulse,
            self.s_detail, self.s_glow, self.s_tile
        ]
        for slider in self.all_sliders:
            slider.on_changed(self.on_slider)

        self.ax_mode = self.fig.add_axes([0.715, 0.07, 0.13, 0.11])
        self.radio_mode = RadioButtons(
            self.ax_mode,
            ("Mandala", "Tunnel", "Hex Bloom", "Fractal Wave"),
            active=0
        )
        self.radio_mode.on_clicked(self.on_mode_change)

        self.ax_quality = self.fig.add_axes([0.855, 0.07, 0.11, 0.11])
        self.radio_quality = RadioButtons(
            self.ax_quality,
            ("Fast", "Balanced", "Fine", "Ultra"),
            active=1
        )
        self.radio_quality.on_clicked(self.on_quality_change)

        self.ax_play = self.fig.add_axes([0.70, 0.015, 0.08, 0.04])
        self.ax_reset = self.fig.add_axes([0.79, 0.015, 0.08, 0.04])
        self.ax_rand = self.fig.add_axes([0.88, 0.015, 0.08, 0.04])

        self.b_play = Button(self.ax_play, "Pause")
        self.b_reset = Button(self.ax_reset, "Reset")
        self.b_rand = Button(self.ax_rand, "Randomize")

        self.b_play.on_clicked(self.toggle_animation)
        self.b_reset.on_clicked(self.reset)
        self.b_rand.on_clicked(self.randomize)

        self.ax_fps = self.fig.add_axes([0.08, 0.115, 0.48, 0.025])
        self.s_fps = Slider(
            self.ax_fps, "Target FPS", 5, 30, valinit=18, valstep=1
        )

        self.status = self.fig.text(
            0.05, 0.065,
            "Balanced: 384x384 render, bilinear upscale",
            fontsize=10
        )

        self.fig.text(
            0.05, 0.025,
            "Tip: if it still feels slow, choose Fast or lower Target FPS. "
            "Press S to save; Space pauses.",
            fontsize=9
        )

        self.fig.canvas.mpl_connect("key_press_event", self.on_key)

    def collect_params(self):
        self.params["density"] = float(self.s_density.val)
        self.params["symmetry"] = int(self.s_symmetry.val)
        self.params["warp"] = float(self.s_warp.val)
        self.params["hue_shift"] = float(self.s_hue.val)
        self.params["color_speed"] = float(self.s_color.val)
        self.params["contrast"] = float(self.s_contrast.val)
        self.params["spiral"] = float(self.s_spiral.val)
        self.params["pulse"] = float(self.s_pulse.val)
        self.params["detail"] = float(self.s_detail.val)
        self.params["glow"] = float(self.s_glow.val)
        self.params["tile_mix"] = float(self.s_tile.val)
        self.params["time"] = self.current_time()

    def redraw(self):
        if self.suppress_updates or self.render_busy:
            return

        self.render_busy = True
        try:
            self.collect_params()
            t0 = time.perf_counter()
            img = self.engine.render(self.params)
            render_ms = (time.perf_counter() - t0) * 1000.0

            self.im.set_data(img)
            self.ax_img.set_title(
                f"{self.params['mode']} | "
                f"{self.engine.resolution}x{self.engine.resolution} | "
                f"{render_ms:.0f} ms render"
            )
            self.status.set_text(
                f"{self.engine.quality}: {self.engine.resolution}x"
                f"{self.engine.resolution} render | {render_ms:.0f} ms/frame"
            )
            self.fig.canvas.draw_idle()
        finally:
            self.render_busy = False

    def on_slider(self, _val):
        self.redraw()

    def on_mode_change(self, label):
        self.params["mode"] = label
        self.redraw()

    def on_quality_change(self, label):
        self.engine.set_quality(label)
        self.redraw()

    def toggle_animation(self, _event):
        if self.animating:
            self.frozen_time = self.current_time()
            self.animating = False
            self.b_play.label.set_text("Play")
        else:
            self.start_time = time.perf_counter() - self.frozen_time
            self.animating = True
            self.b_play.label.set_text("Pause")
        self.redraw()

    def reset(self, _event):
        defaults = [
            (self.s_density, 5.5),
            (self.s_symmetry, 8),
            (self.s_warp, 0.28),
            (self.s_hue, 0.07),
            (self.s_color, 1.0),
            (self.s_contrast, 0.70),
            (self.s_spiral, 1.15),
            (self.s_pulse, 1.2),
            (self.s_detail, 1.1),
            (self.s_glow, 1.2),
            (self.s_tile, 0.45),
        ]

        self.suppress_updates = True
        try:
            for slider, value in defaults:
                slider.set_val(value)

            self.radio_mode.set_active(0)
            self.radio_quality.set_active(1)
            self.s_fps.set_val(18)

            self.params["mode"] = "Mandala"
            self.engine.set_quality("Balanced")
            self.frozen_time = 0.0
            self.start_time = time.perf_counter()
            self.animating = True
            self.b_play.label.set_text("Pause")
        finally:
            self.suppress_updates = False

        self.redraw()

    def randomize(self, _event):
        rng = np.random.default_rng()

        values = [
            (self.s_density, rng.uniform(1.8, 12.5)),
            (self.s_symmetry, int(rng.integers(4, 18))),
            (self.s_warp, rng.uniform(0.05, 0.9)),
            (self.s_hue, rng.uniform(0.0, 1.0)),
            (self.s_color, rng.uniform(0.2, 3.0)),
            (self.s_contrast, rng.uniform(0.3, 1.3)),
            (self.s_spiral, rng.uniform(-3.5, 3.5)),
            (self.s_pulse, rng.uniform(0.1, 3.5)),
            (self.s_detail, rng.uniform(0.1, 2.8)),
            (self.s_glow, rng.uniform(0.2, 3.5)),
            (self.s_tile, rng.uniform(0.0, 1.0)),
        ]

        self.suppress_updates = True
        try:
            for slider, value in values:
                slider.set_val(value)

            idx = int(rng.integers(0, 4))
            self.radio_mode.set_active(idx)
            self.params["mode"] = (
                "Mandala", "Tunnel", "Hex Bloom", "Fractal Wave"
            )[idx]

            self.frozen_time = 0.0
            self.start_time = time.perf_counter()
        finally:
            self.suppress_updates = False

        self.redraw()

    def install_timer(self):
        self.timer = self.fig.canvas.new_timer(interval=40)
        self._last_frame = 0.0

        def tick():
            if not self.animating:
                return

            target_fps = max(1.0, float(self.s_fps.val))
            now = time.perf_counter()

            if now - self._last_frame >= 1.0 / target_fps:
                self._last_frame = now
                self.redraw()

        self.timer.add_callback(tick)
        self.timer.start()

    def on_key(self, event):
        if event.key in ("s", "S"):
            stamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"psychedelic_tile_{stamp}.png"
            self.collect_params()
            img = self.engine.render(self.params)
            plt.imsave(filename, img)
            print(f"Saved {filename}")

        elif event.key == " ":
            self.toggle_animation(None)


def main():
    FastPsychedelicTilingApp()
    plt.show()


if __name__ == "__main__":
    main()
