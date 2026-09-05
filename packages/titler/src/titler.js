// @aparte/titler — a conversation titler that runs anywhere JavaScript runs.
//
// One message in, a title of 3 to 6 words out — words copied verbatim from the
// message (extractive, never invented). Reads the `.bin` files published at
// https://huggingface.co/apartejs/aparte-titler (format 1): a JSON header, the
// BPE merges, and the quantized weights. No dependency, ~6 KB minified.
//
//   import { Titler } from "@aparte/titler";
//   const titler = new Titler(await fetch(modelUrl).then((r) => r.arrayBuffer()));
//   titler.title("Write me a cover letter for a junior data analyst position at a bank");
//   // -> "cover letter junior analyst position bank"
//
// The Python reference implementation (python/aparte_titler/reader.py) gives
// exactly the same titles; the test in test/ checks it on 1,496 messages.

const PUNCT = ".,;:!?\"'()[]{}<>*`“”‘’«»…-–—";   // stripped at the EDGES of a word only ("-Don't" -> "Don't"; "rendez-vous" and "c++" untouched)
const WORD_CHAR = /[\p{L}\p{N}]/u;
// GPT-2's pre-tokenization (HF ByteLevel, use_regex = true)
const SPLIT = /'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+/gu;
// Python's str.isspace(): JS's \s adds U+FEFF and lacks \x1c-\x1f and \x85
const SPACE = "\\t\\n\\v\\f\\r \\x1c-\\x1f\\x85\\xa0\\u1680\\u2000-\\u200a\\u2028\\u2029\\u202f\\u205f\\u3000";
const ONE_SPACE = new RegExp("^[" + SPACE + "]$", "u");
const EDGES = new RegExp("^[" + SPACE + "]+|[" + SPACE + "]+$", "gu");

// ------------------------------------------------------------ file reading

function fp16(u) {
  const s = u & 0x8000 ? -1 : 1, e = (u >> 10) & 0x1f, m = u & 0x3ff;
  if (e === 0) return s * m * 2 ** -24;
  if (e === 31) return m ? NaN : s * Infinity;
  return s * (1 + m / 1024) * 2 ** (e - 15);
}

// bit stream, least significant bit first: n signed integers of `bits` bits
function unpackBits(bytes, n, bits) {
  const out = new Int32Array(n), offset = 1 << (bits - 1);
  if (bits === 8) { for (let i = 0; i < n; i++) out[i] = bytes[i] - 128; return out; }
  let pos = 0;
  for (let i = 0; i < n; i++) {
    let v = 0;
    for (let k = 0; k < bits; k++, pos++) if (bytes[pos >> 3] & (1 << (pos & 7))) v |= 1 << k;
    out[i] = v - offset;
  }
  return out;
}

/** Parse a `.bin` model file. Returns { header, merges, weights }. */
export function readModel(buffer) {
  const bytes = new Uint8Array(buffer), view = new DataView(buffer);
  if (String.fromCharCode(...bytes.subarray(0, 4)) !== "LMAP") throw new Error("not an aparte-titler model file");
  let p = 4;
  const headerLength = view.getUint32(p, true); p += 4;
  const header = JSON.parse(new TextDecoder().decode(bytes.subarray(p, p + headerLength))); p += headerLength;
  if ((header.format ?? 1) !== 1) throw new Error("model format " + header.format + " is not supported by this runtime (format 1)");
  const mergesLength = view.getUint32(p, true); p += 4;
  const merges = new Uint16Array(buffer.slice(p, p + mergesLength)); p += mergesLength;
  const weights = {};
  for (const t of header.tensors) {
    const shape = t.shape, count = shape.reduce((a, b) => a * b, 1);
    const w = new Float32Array(count);
    if (t.type === "f32") {
      for (let i = 0; i < count; i++) w[i] = view.getFloat32(p + 4 * i, true);
    } else if (t.type === "f16") {
      for (let i = 0; i < count; i++) w[i] = fp16(view.getUint16(p + 2 * i, true));
    } else {
      // quantized rows: packed integers, then one fp16 scale per row
      const bits = parseInt(t.type.slice(1)), rows = shape[0], cols = count / rows;
      const packedLength = Math.ceil((count * bits) / 8);
      const q = unpackBits(bytes.subarray(p, p + packedLength), count, bits);
      for (let r = 0; r < rows; r++) {
        const scale = fp16(view.getUint16(p + packedLength + 2 * r, true));
        for (let c = 0; c < cols; c++) w[r * cols + c] = q[r * cols + c] * scale;
      }
    }
    weights[t.name] = { data: w, shape };
    p += t.bytes;
  }
  return { header, merges, weights };
}

// -------------------------------------------------------------- tokenizer

class Tokenizer {
  constructor(header, merges) {
    this.byteIds = header.byte_ids;          // token id of each of the 256 bytes
    this.base = header.merges_base;          // merge k produces token id base + k
    this.rank = new Map();
    for (let k = 0; k < merges.length / 2; k++) this.rank.set(merges[2 * k] * 65536 + merges[2 * k + 1], k);
    this.encoder = new TextEncoder();
  }

  /** Token ids and [start, end) offsets in UTF-16 units, as HF with trim_offsets = false. */
  encode(text) {
    const ids = [], offsets = [];
    for (const m of text.matchAll(SPLIT)) {
      const start = m.index, piece = m[0];
      const charOfByte = [], endOfByte = [], tokens = [];
      let i = 0;
      for (const ch of piece) {                 // per code point
        const n = ch.length;                    // 1 or 2 UTF-16 units
        for (const b of this.encoder.encode(ch)) { tokens.push([this.byteIds[b], 1]); charOfByte.push(i); endOfByte.push(i + n); }
        i += n;
      }
      for (;;) {                                // merge the lowest-ranked pair first
        let best = -1, at = -1;
        for (let j = 0; j < tokens.length - 1; j++) {
          const r = this.rank.get(tokens[j][0] * 65536 + tokens[j + 1][0]);
          if (r !== undefined && (best < 0 || r < best)) { best = r; at = j; }
        }
        if (best < 0) break;
        tokens.splice(at, 2, [this.base + best, tokens[at][1] + tokens[at + 1][1]]);
      }
      let pos = 0;
      for (const [id, n] of tokens) {
        ids.push(id);
        offsets.push([start + charOfByte[pos], start + endOfByte[pos + n - 1]]);
        pos += n;
      }
    }
    return { ids, offsets };
  }
}

// ------------------------------------------------------------------ model

function layerNorm(x, L, D, g, b, out) {
  for (let t = 0; t < L; t++) {
    let mean = 0; for (let d = 0; d < D; d++) mean += x[t * D + d]; mean /= D;
    let v = 0; for (let d = 0; d < D; d++) { const e = x[t * D + d] - mean; v += e * e; } v /= D;
    const inv = 1 / Math.sqrt(v + 1e-5);
    for (let d = 0; d < D; d++) out[t * D + d] = (x[t * D + d] - mean) * inv * g[d] + b[d];
  }
  return out;
}

function erf(x) {  // Abramowitz-Stegun 7.1.26, error < 1.5e-7
  const s = Math.sign(x); x = Math.abs(x);
  const t = 1 / (1 + 0.3275911 * x);
  const y = 1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-x * x);
  return s * y;
}
const gelu = (x) => 0.5 * x * (1 + erf(x / Math.SQRT2));

// y[L x N] = x[L x K] · W[N x K]^T + b[N]   (torch layout: rows are outputs)
function linear(x, L, K, W, b, N, out) {
  for (let t = 0; t < L; t++) for (let n = 0; n < N; n++) {
    let s = b ? b[n] : 0;
    for (let k = 0; k < K; k++) s += x[t * K + k] * W[n * K + k];
    out[t * N + n] = s;
  }
  return out;
}

class Model {
  constructor(header, weights) {
    this.h = header; this.w = Object.fromEntries(Object.entries(weights).map(([k, v]) => [k, v.data]));
    this.D = header.dim; this.H = header.heads; this.embDim = header.dim_emb;
  }

  /** One logit per token: positive = keep. */
  logits(ids) {
    const { w, D, H, h } = this, L = ids.length, Dh = D / H;
    const x = new Float32Array(L * D);
    if (this.embDim) {                        // factorized embeddings (ALBERT style)
      const e = new Float32Array(L * this.embDim);
      for (let t = 0; t < L; t++) for (let d = 0; d < this.embDim; d++) e[t * this.embDim + d] = w["emb_tok.weight"][ids[t] * this.embDim + d];
      linear(e, L, this.embDim, w["proj_emb.weight"], null, D, x);
    } else {
      for (let t = 0; t < L; t++) for (let d = 0; d < D; d++) x[t * D + d] = w["emb_tok.weight"][ids[t] * D + d];
    }
    for (let t = 0; t < L; t++) {
      const pos = Math.min(t, h.max_pos - 1);
      if (h.positions === "sinus") {
        for (let d = 0; d < D; d += 2) {
          const f = Math.exp(d * (-Math.log(10000) / D));
          x[t * D + d] += Math.sin(pos * f); x[t * D + d + 1] += Math.cos(pos * f);
        }
      } else {
        for (let d = 0; d < D; d++) x[t * D + d] += w["emb_pos.weight"][pos * D + d];
      }
    }
    const y = new Float32Array(L * D), qkv = new Float32Array(L * 3 * D), att = new Float32Array(L * D);
    const f1 = new Float32Array(L * 4 * D), f2 = new Float32Array(L * D), s = new Float32Array(L);
    for (let c = 0; c < h.layers; c++) {
      const k = "blocks." + c + ".";
      layerNorm(x, L, D, w[k + "n1.weight"], w[k + "n1.bias"], y);
      linear(y, L, D, w[k + "att.qkv.weight"], w[k + "att.qkv.bias"], 3 * D, qkv);
      att.fill(0);
      for (let head = 0; head < H; head++) {    // every token attends to every token (encoder: no causal mask)
        const o = head * Dh;
        for (let i = 0; i < L; i++) {
          let max = -Infinity;
          for (let j = 0; j < L; j++) {
            let d = 0;
            for (let e = 0; e < Dh; e++) d += qkv[i * 3 * D + o + e] * qkv[j * 3 * D + D + o + e];
            s[j] = d / Math.sqrt(Dh); if (s[j] > max) max = s[j];
          }
          let sum = 0;
          for (let j = 0; j < L; j++) { s[j] = Math.exp(s[j] - max); sum += s[j]; }
          for (let j = 0; j < L; j++) {
            const p = s[j] / sum;
            for (let e = 0; e < Dh; e++) att[i * D + o + e] += p * qkv[j * 3 * D + 2 * D + o + e];
          }
        }
      }
      linear(att, L, D, w[k + "att.output.weight"], w[k + "att.output.bias"], D, f2);
      for (let i = 0; i < L * D; i++) x[i] += f2[i];
      layerNorm(x, L, D, w[k + "n2.weight"], w[k + "n2.bias"], y);
      linear(y, L, D, w[k + "ff.net.0.weight"], w[k + "ff.net.0.bias"], 4 * D, f1);
      for (let i = 0; i < L * 4 * D; i++) f1[i] = gelu(f1[i]);
      linear(f1, L, 4 * D, w[k + "ff.net.2.weight"], w[k + "ff.net.2.bias"], D, f2);
      for (let i = 0; i < L * D; i++) x[i] += f2[i];
    }
    layerNorm(x, L, D, w["norm.weight"], w["norm.bias"], y);
    const z = new Float32Array(L);
    linear(y, L, D, w["head.weight"], w["head.bias"], 1, z);
    return z;
  }
}

// --------------------------------------------------------------- decoding

const isSpace = (c) => ONE_SPACE.test(c);
const isWordChar = (c) => WORD_CHAR.test(c);

// Python's message.strip()[:maxChars]: Python's whitespace set, cut in code points
function prepare(message, maxChars) {
  const s = message.replace(EDGES, "");
  let n = 0, end = 0;
  for (const ch of s) { if (n === maxChars) break; n++; end += ch.length; }
  return s.slice(0, end);
}

// a new word starts on a token whose first code point is a space or not a letter/digit —
// except a hyphen or an apostrophe glued between two letters ("Pourrais-tu", "l'école"),
// which keeps the word whole: a title must never start with a fragment like "-tu"
const JOINERS = new Set(["-", "'", "’"]);
function wordIds(text, offsets) {
  const out = []; let num = -1, prevEnd = -1;
  offsets.forEach(([a, b], i) => {
    const c = b > a ? String.fromCodePoint(text.codePointAt(a)) : "";
    let starts = i === 0 || isSpace(c) || !isWordChar(c);
    if (starts && i > 0 && JOINERS.has(c) && a === prevEnd && a > 0 && isWordChar(String.fromCodePoint(text.codePointAt(a - 1)))) {
      const next = a + c.length;
      if (next < text.length && isWordChar(String.fromCodePoint(text.codePointAt(next)))) starts = false;
    }
    if (starts) num++;
    out.push(num);
    prevEnd = b;
  });
  return out;
}

function trimPunct(m) {
  let a = 0, b = m.length;
  while (a < b && PUNCT.includes(m[a])) a++;
  while (b > a && PUNCT.includes(m[b - 1])) b--;
  return m.slice(a, b);
}

// kept tokens, merged into character intervals, sliced from the text
function recompose(text, offsets, keep) {
  const spans = [];
  offsets.forEach(([a, b], i) => {
    if (!keep[i] || b <= a) return;
    if (spans.length && a <= spans[spans.length - 1][1]) spans[spans.length - 1][1] = Math.max(spans[spans.length - 1][1], b);
    else spans.push([a, b]);
  });
  return spans.map(([a, b]) => trimPunct(text.slice(a, b).replace(EDGES, ""))).filter(Boolean).join(" ");
}

export class Titler {
  /** @param {ArrayBuffer} buffer the contents of a `.bin` model file */
  constructor(buffer) {
    const { header, merges, weights } = readModel(buffer);
    this.header = header; this.tokenizer = new Tokenizer(header, merges); this.model = new Model(header, weights);
    // a 1.0 file carries no `decoding` block and decodes with exactly these values
    const h = header.decoding ?? {};
    this.decoding = {
      threshold: h.threshold ?? 0.5, minWords: h.min_words ?? 3,
      maxWords: h.max_words ?? 6, keepAllUpTo: h.keep_all_up_to ?? 4,
    };
  }

  /** Score every word of the message: [{ word, start, end, score }] in text order. */
  words(message) {
    const text = prepare(message, this.header.max_chars);
    const { ids, offsets } = this.tokenizer.encode(text);
    if (!ids.length) return { text, words: [] };
    const z = this.model.logits(ids);
    const groups = new Map();
    wordIds(text, offsets).forEach((wid, i) => { if (!groups.has(wid)) groups.set(wid, []); groups.get(wid).push(i); });
    const words = [];
    for (const tokens of groups.values()) {
      const start = offsets[tokens[0]][0], end = offsets[tokens[tokens.length - 1]][1];
      if (!WORD_CHAR.test(text.slice(start, end))) continue;   // punctuation-only groups are never a title word
      let sum = 0; for (const i of tokens) sum += z[i];            // the decision is per word: mean of its token logits
      // `word` is the clean word (no leading space, no edge punctuation); start/end are the raw token offsets
      words.push({ word: trimPunct(text.slice(start, end).replace(EDGES, "")), start, end, score: 1 / (1 + Math.exp(-sum / tokens.length)), tokens });
    }
    return { text, words, offsets, count: ids.length };
  }

  /**
   * The title.
   *
   * Two decodings. A number keeps that many best-scored words: what version
   * 1.0 shipped, kept for compatibility. Otherwise the hybrid decoding of
   * version 1.1 — among the six best, those the model is at least half sure
   * of, never fewer than three, and everything when the message is barely
   * longer than a title. Measured on the seventeen gold sets: +3 points on
   * average, nothing lost in any language. It corrects a length, not a
   * ranking: a message of ten words used to be answered with six.
   *
   * @param {string} message
   * @param {number | {mode?: "hybrid" | "budget", budget?: number, stopWords?: Iterable<string>}} [options]
   */
  title(message, options) {
    const d = this.decoding;
    const budget = typeof options === "number" ? options
      : options?.mode === "budget" ? (options.budget ?? d.maxWords) : null;
    let { text, words } = this.words(message);
    if (!words.length) return "";
    if (options?.stopWords) {
      const banned = new Set([...options.stopWords].map((w) => w.toLowerCase()));
      const kept = words.filter((w) => !banned.has(w.word.toLowerCase()));
      if (kept.length >= d.minWords) words = kept;
    }
    const byScore = [...words].sort((a, b) => b.score - a.score);   // stable sort: ties keep text order
    let chosen;
    if (budget !== null) chosen = byScore.slice(0, budget);
    else if (words.length <= d.keepAllUpTo) chosen = words;
    else {
      const top = byScore.slice(0, d.maxWords);
      chosen = top.filter((w) => w.score >= d.threshold);
      if (chosen.length < d.minWords) chosen = top.slice(0, d.minWords);
    }
    // one span per kept word, in text order: punctuation between two kept words is never carried over
    return chosen.sort((a, b) => a.start - b.start).map((w) => w.word).filter(Boolean).join(" ");
  }
}

/** Convenience: build a Titler from an ArrayBuffer (or a Promise of one). */
export async function loadTitler(buffer) { return new Titler(await buffer); }
