/* Shared DOM/canvas helpers for the CIFAR atlas and per-image card. */
(function (root) {
  "use strict";

  const CLASS_COLORS = [
    "#2458d6",
    "#0f766e",
    "#c2610a",
    "#b42318",
    "#6d4bc3",
    "#8a4b08",
    "#16794a",
    "#0e7490",
    "#9d174d",
    "#4b5563",
  ];

  const CLASS_SHAPES = [
    "circle",
    "square",
    "triangle",
    "diamond",
    "plus",
    "cross",
    "triangle-down",
    "hexagon",
    "star",
    "square-outline",
  ];

  function classColor(index) {
    return CLASS_COLORS[index % CLASS_COLORS.length];
  }

  function classShape(index) {
    return CLASS_SHAPES[index % CLASS_SHAPES.length];
  }

  function fmt(value, digits) {
    const d = digits === undefined ? 3 : digits;
    if (!isFinite(value)) return "–";
    const abs = Math.abs(value);
    if (abs >= 10000 || (abs > 0 && abs < 0.0005)) return value.toExponential(2);
    return value.toFixed(d);
  }

  function clamp(value, low, high) {
    return Math.max(low, Math.min(high, value));
  }

  function loadImage(src, callback) {
    const image = new Image();
    image.onload = () => callback(null, image);
    image.onerror = () => callback(new Error(`could not load ${src}`));
    image.src = src;
  }

  function loadSprites(data, callback) {
    const base = (data.meta && data.meta.sprite_dir) || "../assets/atlas";
    const output = {};
    let remaining = 2;
    let failed = null;
    const done = () => {
      remaining -= 1;
      if (remaining === 0) callback(failed, output);
    };
    for (const key of ["images", "views"]) {
      const spec = data.sprites[key];
      const src = `${base}/${spec.src}`;
      loadImage(src, (error, image) => {
        if (error) failed = error;
        else output[key] = image;
        done();
      });
    }
  }

  function paintTile(canvas, image, spec, index, size) {
    if (!canvas || !image || !spec) return;
    const dpr = Math.max(1, window.devicePixelRatio || 1);
    const pixels = Math.max(1, Math.round(size * dpr));
    if (canvas.width !== pixels || canvas.height !== pixels) {
      canvas.width = pixels;
      canvas.height = pixels;
    }
    const ctx = canvas.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.imageSmoothingEnabled = false;
    ctx.clearRect(0, 0, size, size);
    const column = index % spec.cols;
    const row = Math.floor(index / spec.cols);
    if (row >= spec.rows) return;
    ctx.drawImage(
      image,
      column * spec.tile,
      row * spec.tile,
      spec.tile,
      spec.tile,
      0,
      0,
      size,
      size
    );
  }

  function markerPath(ctx, x, y, radius, shape) {
    ctx.beginPath();
    if (shape === "square" || shape === "square-outline") {
      ctx.rect(x - radius, y - radius, radius * 2, radius * 2);
      return;
    }
    if (shape === "triangle") {
      ctx.moveTo(x, y - radius * 1.2);
      ctx.lineTo(x + radius * 1.1, y + radius * 0.8);
      ctx.lineTo(x - radius * 1.1, y + radius * 0.8);
      ctx.closePath();
      return;
    }
    if (shape === "triangle-down") {
      ctx.moveTo(x, y + radius * 1.2);
      ctx.lineTo(x + radius * 1.1, y - radius * 0.8);
      ctx.lineTo(x - radius * 1.1, y - radius * 0.8);
      ctx.closePath();
      return;
    }
    if (shape === "diamond") {
      ctx.moveTo(x, y - radius * 1.25);
      ctx.lineTo(x + radius * 1.1, y);
      ctx.lineTo(x, y + radius * 1.25);
      ctx.lineTo(x - radius * 1.1, y);
      ctx.closePath();
      return;
    }
    if (shape === "plus") {
      const arm = radius * 0.42;
      ctx.rect(x - arm, y - radius, arm * 2, radius * 2);
      ctx.rect(x - radius, y - arm, radius * 2, arm * 2);
      return;
    }
    if (shape === "cross") {
      const arm = radius * 0.38;
      ctx.moveTo(x - radius, y - radius);
      ctx.lineTo(x + radius, y + radius);
      ctx.moveTo(x + radius, y - radius);
      ctx.lineTo(x - radius, y + radius);
      ctx.lineWidth = Math.max(1, arm);
      return;
    }
    if (shape === "hexagon") {
      for (let index = 0; index < 6; index += 1) {
        const angle = (Math.PI / 3) * index - Math.PI / 6;
        const px = x + radius * Math.cos(angle);
        const py = y + radius * Math.sin(angle);
        if (index === 0) ctx.moveTo(px, py);
        else ctx.lineTo(px, py);
      }
      ctx.closePath();
      return;
    }
    if (shape === "star") {
      for (let index = 0; index < 10; index += 1) {
        const angle = (Math.PI / 5) * index - Math.PI / 2;
        const length = index % 2 === 0 ? radius * 1.2 : radius * 0.52;
        const px = x + length * Math.cos(angle);
        const py = y + length * Math.sin(angle);
        if (index === 0) ctx.moveTo(px, py);
        else ctx.lineTo(px, py);
      }
      ctx.closePath();
      return;
    }
    ctx.arc(x, y, radius, 0, Math.PI * 2);
  }

  function drawMarker(ctx, x, y, radius, shape, fill, stroke, lineWidth) {
    markerPath(ctx, x, y, radius, shape);
    if (fill) {
      ctx.fillStyle = fill;
      ctx.fill();
    }
    if (stroke) {
      ctx.strokeStyle = stroke;
      ctx.lineWidth = lineWidth || 1;
      ctx.stroke();
    }
  }

  function thumbNode(sprite, spec, index, size, className) {
    const canvas = document.createElement("canvas");
    canvas.className = className || "thumb";
    canvas.style.width = `${size}px`;
    canvas.style.height = `${size}px`;
    canvas.setAttribute("aria-hidden", "true");
    paintTile(canvas, sprite, spec, index, size);
    return canvas;
  }

  root.AtlasUI = {
    CLASS_COLORS,
    CLASS_SHAPES,
    classColor,
    classShape,
    fmt,
    clamp,
    loadSprites,
    paintTile,
    drawMarker,
    thumbNode,
  };
})(typeof globalThis !== "undefined" ? globalThis : this);
