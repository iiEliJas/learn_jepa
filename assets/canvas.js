/* Small dependency-free canvas plotting helpers for the learning demos.
 *
 * Classic script (no ES modules) so the pages work when opened directly from
 * disk. The module attaches `CanvasKit` to the global object.
 */
(function (root, factory) {
  if (typeof module === "object" && module.exports) {
    module.exports = factory();
  } else {
    root.CanvasKit = factory();
  }
})(typeof globalThis !== "undefined" ? globalThis : this, function () {
  "use strict";

  const COLORS = {
    ink: "#18202a",
    muted: "#5c6775",
    faint: "#8a95a3",
    grid: "#e8ecf2",
    axis: "#b9c3cf",
    accent: "#2458d6",
    accentSoft: "#9db7f3",
    teal: "#0f766e",
    orange: "#c2610a",
    red: "#b42318",
    green: "#16794a",
    purple: "#6d4bc3",
    points: ["#2458d6", "#0f766e", "#c2610a", "#b42318", "#6d4bc3", "#16794a"],
  };

  function fitCanvas(canvas) {
    const rect = canvas.getBoundingClientRect();
    const dpr = Math.max(1, window.devicePixelRatio || 1);
    const width = Math.max(1, Math.round(rect.width));
    const height = Math.max(1, Math.round(rect.height));
    if (canvas.width !== Math.round(width * dpr) || canvas.height !== Math.round(height * dpr)) {
      canvas.width = Math.round(width * dpr);
      canvas.height = Math.round(height * dpr);
    }
    const ctx = canvas.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    return { ctx, w: width, h: height };
  }

  function niceTicks(min, max, count) {
    if (!isFinite(min) || !isFinite(max) || max <= min) {
      return [min];
    }
    const span = max - min;
    const rough = span / Math.max(1, count);
    const magnitude = Math.pow(10, Math.floor(Math.log10(rough)));
    const normalized = rough / magnitude;
    let step;
    if (normalized < 1.5) step = 1;
    else if (normalized < 3) step = 2;
    else if (normalized < 7) step = 5;
    else step = 10;
    step *= magnitude;
    const ticks = [];
    const start = Math.ceil(min / step) * step;
    for (let value = start; value <= max + step * 1e-6; value += step) {
      ticks.push(Math.abs(value) < step * 1e-9 ? 0 : value);
    }
    return ticks;
  }

  function formatTick(value) {
    if (value === 0) return "0";
    const abs = Math.abs(value);
    if (abs >= 1000 || abs < 0.001) return value.toExponential(1);
    if (abs >= 100) return value.toFixed(0);
    if (abs >= 10) return value.toFixed(1).replace(/\.0$/, "");
    return value.toFixed(2).replace(/0+$/, "").replace(/\.$/, "");
  }

  class Plot {
    constructor(canvas, options) {
      this.canvas = canvas;
      this.options = Object.assign(
        {
          padding: { l: 50, r: 16, t: 28, b: 36 },
          xmin: 0,
          xmax: 1,
          ymin: 0,
          ymax: 1,
          xlabel: "",
          ylabel: "",
          title: "",
          grid: true,
          zeroLines: false,
          xTicks: null,
          yTicks: null,
          xFormat: formatTick,
          yFormat: formatTick,
          background: "#ffffff",
        },
        options || {}
      );
    }

    draw() {
      const { ctx, w, h } = fitCanvas(this.canvas);
      const o = this.options;
      const pad = o.padding;
      const box = {
        x0: pad.l,
        y0: pad.t,
        x1: Math.max(pad.l + 1, w - pad.r),
        y1: Math.max(pad.t + 1, h - pad.b),
      };
      this.ctx = ctx;
      this.w = w;
      this.h = h;
      this.box = box;
      const o2 = this.options;
      this.sx = (box.x1 - box.x0) / (o2.xmax - o2.xmin || 1);
      this.sy = (box.y1 - box.y0) / (o2.ymax - o2.ymin || 1);
      this.x = (value) => box.x0 + (value - o2.xmin) * this.sx;
      this.y = (value) => box.y1 - (value - o2.ymin) * this.sy;

      ctx.clearRect(0, 0, w, h);
      ctx.fillStyle = o.background;
      ctx.fillRect(0, 0, w, h);

      const xTicks = o.xTicks || niceTicks(o.xmin, o.xmax, 5);
      const yTicks = o.yTicks || niceTicks(o.ymin, o.ymax, 5);

      ctx.save();
      ctx.font = "11px ui-sans-serif, system-ui, -apple-system, Segoe UI, sans-serif";
      ctx.textBaseline = "middle";
      if (o.grid) {
        ctx.strokeStyle = COLORS.grid;
        ctx.lineWidth = 1;
        ctx.beginPath();
        for (const tick of xTicks) {
          const px = Math.round(this.x(tick)) + 0.5;
          ctx.moveTo(px, box.y0);
          ctx.lineTo(px, box.y1);
        }
        for (const tick of yTicks) {
          const py = Math.round(this.y(tick)) + 0.5;
          ctx.moveTo(box.x0, py);
          ctx.lineTo(box.x1, py);
        }
        ctx.stroke();
      }

      if (o.zeroLines) {
        ctx.strokeStyle = COLORS.axis;
        ctx.lineWidth = 1;
        ctx.beginPath();
        if (o.ymin < 0 && o.ymax > 0) {
          const py = Math.round(this.y(0)) + 0.5;
          ctx.moveTo(box.x0, py);
          ctx.lineTo(box.x1, py);
        }
        if (o.xmin < 0 && o.xmax > 0) {
          const px = Math.round(this.x(0)) + 0.5;
          ctx.moveTo(px, box.y0);
          ctx.lineTo(px, box.y1);
        }
        ctx.stroke();
      }

      ctx.strokeStyle = COLORS.axis;
      ctx.lineWidth = 1;
      ctx.strokeRect(
        Math.round(box.x0) + 0.5,
        Math.round(box.y0) + 0.5,
        Math.round(box.x1 - box.x0),
        Math.round(box.y1 - box.y0)
      );

      ctx.fillStyle = COLORS.muted;
      ctx.textAlign = "center";
      for (const tick of xTicks) {
        ctx.fillText(o.xFormat(tick), this.x(tick), box.y1 + 14);
      }
      ctx.textAlign = "right";
      for (const tick of yTicks) {
        ctx.fillText(o.yFormat(tick), box.x0 - 7, this.y(tick));
      }

      ctx.fillStyle = COLORS.ink;
      ctx.font = "600 11.5px ui-sans-serif, system-ui, -apple-system, Segoe UI, sans-serif";
      if (o.title) {
        ctx.textAlign = "left";
        ctx.fillText(o.title, box.x0, 12);
      }
      ctx.fillStyle = COLORS.muted;
      ctx.font = "11px ui-sans-serif, system-ui, -apple-system, Segoe UI, sans-serif";
      if (o.xlabel) {
        ctx.textAlign = "center";
        ctx.fillText(o.xlabel, (box.x0 + box.x1) / 2, h - 7);
      }
      if (o.ylabel) {
        ctx.save();
        ctx.translate(11, (box.y0 + box.y1) / 2);
        ctx.rotate(-Math.PI / 2);
        ctx.textAlign = "center";
        ctx.fillText(o.ylabel, 0, 0);
        ctx.restore();
      }
      ctx.restore();
      return this;
    }

    clip(fn) {
      const { ctx, box } = this;
      ctx.save();
      ctx.beginPath();
      ctx.rect(box.x0, box.y0, box.x1 - box.x0, box.y1 - box.y0);
      ctx.clip();
      try {
        fn(this);
      } finally {
        ctx.restore();
      }
    }

    line(points, options) {
      const o = Object.assign(
        { stroke: COLORS.accent, width: 1.8, dash: null, alpha: 1 },
        options || {}
      );
      if (!points || points.length < 2) return this;
      const ctx = this.ctx;
      ctx.save();
      ctx.strokeStyle = o.stroke;
      ctx.lineWidth = o.width;
      ctx.globalAlpha = o.alpha;
      if (o.dash) ctx.setLineDash(o.dash);
      ctx.beginPath();
      points.forEach((p, index) => {
        const px = this.x(p[0]);
        const py = this.y(p[1]);
        if (index === 0) ctx.moveTo(px, py);
        else ctx.lineTo(px, py);
      });
      ctx.stroke();
      ctx.restore();
      return this;
    }

    area(points, baseline, options) {
      const o = Object.assign({ fill: "rgba(36, 88, 214, 0.12)", stroke: null }, options || {});
      if (!points || points.length < 2) return this;
      const ctx = this.ctx;
      ctx.save();
      ctx.fillStyle = o.fill;
      ctx.beginPath();
      ctx.moveTo(this.x(points[0][0]), this.y(baseline));
      for (const p of points) ctx.lineTo(this.x(p[0]), this.y(p[1]));
      ctx.lineTo(this.x(points[points.length - 1][0]), this.y(baseline));
      ctx.closePath();
      ctx.fill();
      if (o.stroke) {
        ctx.strokeStyle = o.stroke;
        ctx.lineWidth = 1;
        ctx.stroke();
      }
      ctx.restore();
      return this;
    }

    scatter(points, options) {
      const o = Object.assign(
        { r: 2.5, fill: COLORS.accent, stroke: null, alpha: 1, lineWidth: 1 },
        options || {}
      );
      const ctx = this.ctx;
      ctx.save();
      ctx.globalAlpha = o.alpha;
      for (const p of points) {
        const fill = Array.isArray(o.fill) ? o.fill[p.index || 0] : o.fill;
        ctx.beginPath();
        ctx.arc(this.x(p[0]), this.y(p[1]), o.r, 0, Math.PI * 2);
        ctx.fillStyle = fill;
        ctx.fill();
        if (o.stroke) {
          ctx.strokeStyle = o.stroke;
          ctx.lineWidth = o.lineWidth;
          ctx.stroke();
        }
      }
      ctx.restore();
      return this;
    }

    arrow(x0, y0, x1, y1, options) {
      const o = Object.assign(
        { stroke: COLORS.accent, width: 1.3, head: 5, alpha: 1 },
        options || {}
      );
      const ctx = this.ctx;
      const ax = this.x(x0);
      const ay = this.y(y0);
      const bx = this.x(x1);
      const by = this.y(y1);
      const angle = Math.atan2(by - ay, bx - ax);
      ctx.save();
      ctx.strokeStyle = o.stroke;
      ctx.fillStyle = o.stroke;
      ctx.globalAlpha = o.alpha;
      ctx.lineWidth = o.width;
      ctx.beginPath();
      ctx.moveTo(ax, ay);
      ctx.lineTo(bx, by);
      ctx.stroke();
      if (o.head > 0) {
        ctx.beginPath();
        ctx.moveTo(bx, by);
        ctx.lineTo(
          bx - o.head * Math.cos(angle - Math.PI / 6),
          by - o.head * Math.sin(angle - Math.PI / 6)
        );
        ctx.lineTo(
          bx - o.head * Math.cos(angle + Math.PI / 6),
          by - o.head * Math.sin(angle + Math.PI / 6)
        );
        ctx.closePath();
        ctx.fill();
      }
      ctx.restore();
      return this;
    }

    hline(value, options) {
      const o = Object.assign(
        { stroke: COLORS.faint, width: 1, dash: [4, 4], alpha: 1 },
        options || {}
      );
      return this.line(
        [
          [this.options.xmin, value],
          [this.options.xmax, value],
        ],
        o
      );
    }

    vline(value, options) {
      const o = Object.assign(
        { stroke: COLORS.faint, width: 1, dash: [4, 4], alpha: 1 },
        options || {}
      );
      return this.line(
        [
          [value, this.options.ymin],
          [value, this.options.ymax],
        ],
        o
      );
    }

    text(x, y, text, options) {
      const o = Object.assign(
        {
          fill: COLORS.ink,
          font: "11px ui-sans-serif, system-ui, -apple-system, Segoe UI, sans-serif",
          align: "left",
          baseline: "middle",
          background: null,
        },
        options || {}
      );
      const ctx = this.ctx;
      ctx.save();
      ctx.font = o.font;
      ctx.fillStyle = o.fill;
      ctx.textAlign = o.align;
      ctx.textBaseline = o.baseline;
      const px = this.x(x);
      const py = this.y(y);
      if (o.background) {
        const metrics = ctx.measureText(text);
        ctx.fillStyle = o.background;
        ctx.fillRect(px - 3, py - 8, metrics.width + 6, 16);
        ctx.fillStyle = o.fill;
      }
      ctx.fillText(text, px, py);
      ctx.restore();
      return this;
    }

    legend(entries, options) {
      const o = Object.assign(
        {
          x: null,
          y: null,
          align: "left",
          font: "11px ui-sans-serif, system-ui, -apple-system, Segoe UI, sans-serif",
        },
        options || {}
      );
      const ctx = this.ctx;
      const box = this.box;
      const widths = entries.map((entry) => {
        ctx.font = o.font;
        return ctx.measureText(entry.label).width + 26;
      });
      const width = Math.max(...widths, 40);
      const height = entries.length * 15 + 6;
      let x = o.x;
      let y = o.y;
      if (x === null) x = o.align === "right" ? box.x1 - width - 6 : box.x0 + 6;
      if (y === null) y = box.y0 + 6;
      ctx.save();
      ctx.font = o.font;
      ctx.fillStyle = "rgba(255,255,255,0.82)";
      ctx.fillRect(x - 4, y - 4, width + 8, height);
      entries.forEach((entry, index) => {
        const rowY = y + index * 15 + 3;
        ctx.strokeStyle = entry.color;
        ctx.fillStyle = entry.color;
        ctx.lineWidth = 2.2;
        if (entry.dash) ctx.setLineDash(entry.dash);
        else ctx.setLineDash([]);
        if (entry.kind === "line") {
          ctx.beginPath();
          ctx.moveTo(x, rowY);
          ctx.lineTo(x + 12, rowY);
          ctx.stroke();
        } else {
          ctx.beginPath();
          ctx.arc(x + 6, rowY, 3.5, 0, Math.PI * 2);
          ctx.fill();
        }
        ctx.fillStyle = COLORS.muted;
        ctx.textAlign = "left";
        ctx.textBaseline = "middle";
        ctx.fillText(entry.label, x + 18, rowY);
      });
      ctx.restore();
      return this;
    }
  }

  function histogram(values, bins, min, max) {
    const counts = new Array(bins).fill(0);
    const width = (max - min) / bins;
    for (const value of values) {
      if (!isFinite(value)) continue;
      const index = Math.floor((value - min) / width);
      if (index >= 0 && index < bins) counts[index] += 1;
    }
    const total = values.length || 1;
    return {
      counts,
      density: counts.map((count) => count / (total * width)),
      centers: counts.map((_, index) => min + (index + 0.5) * width),
      width,
      edges: counts.map((_, index) => min + index * width).concat([max]),
    };
  }

  function stepPoints(hist) {
    const points = [];
    for (let i = 0; i < hist.centers.length; i += 1) {
      const left = hist.edges[i];
      const right = hist.edges[i + 1];
      const value = hist.density[i];
      points.push([left, value], [right, value]);
    }
    return points;
  }

  function normalPdf(x, mean, std) {
    const m = mean || 0;
    const s = std || 1;
    return Math.exp(-0.5 * ((x - m) / s) ** 2) / (s * Math.sqrt(2 * Math.PI));
  }

  return {
    COLORS,
    Plot,
    fitCanvas,
    niceTicks,
    formatTick,
    histogram,
    stepPoints,
    normalPdf,
  };
});
