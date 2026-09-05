/** A scored word of the message, in text order. */
export interface ScoredWord {
  /** the word as it appears in the message (leading space excluded) */
  word: string;
  /** character offsets [start, end) in the prepared text (UTF-16 units) */
  start: number;
  end: number;
  /** probability that the word belongs to the title */
  score: number;
  /** indices of the tokens that make up the word */
  tokens: number[];
}

export interface ModelHeader {
  /** file format version (this runtime reads 1) */
  format: number;
  /** identity of the file: "aparte-titler", its semantic version, the model scope ("latin", "fr", …), its languages and precision */
  name: string;
  version: string;
  model: string;
  languages: string[];
  precision: "fp32" | "int8" | "int4" | "int3" | string;
  license: string;
  url: string;
  dim: number;
  layers: number;
  heads: number;
  positions: "learned" | "sinus";
  dim_emb: number;
  vocab: number;
  n_merges: number;
  merges_base: number;
  /** token id of each of the 256 bytes */
  byte_ids: number[];
  pad: number;
  max_chars: number;
  max_pos: number;
  bits: number;
  /** training seed, 0 when it was not fixed */
  seed?: number;
  /** the decoding this file was built for; absent on 1.0 files, which use the same defaults */
  decoding?: {
    default?: "hybrid" | "budget";
    threshold?: number;
    min_words?: number;
    max_words?: number;
    keep_all_up_to?: number;
  };
  tensors: { name: string; shape: number[]; type: "f32" | "f16" | string; bytes: number }[];
}

export interface TitleOptions {
  /** "hybrid" (default) or "budget", the 1.0 decoding */
  mode?: "hybrid" | "budget";
  /** how many words in budget mode (default 6) */
  budget?: number;
  /** words never kept unless dropping them would leave fewer than three */
  stopWords?: Iterable<string>;
}

export interface ParsedModel {
  header: ModelHeader;
  merges: Uint16Array;
  weights: Record<string, { data: Float32Array; shape: number[] }>;
}

/** Parse a `.bin` model file (format 1). */
export function readModel(buffer: ArrayBuffer): ParsedModel;

export class Titler {
  /** @param buffer the contents of a `.bin` model file */
  constructor(buffer: ArrayBuffer);
  readonly header: ModelHeader;
  /** Score every word of the message. */
  words(message: string): { text: string; words: ScoredWord[]; offsets?: [number, number][]; count?: number };
  /** The decoding in force, from the header or the defaults. */
  readonly decoding: { threshold: number; minWords: number; maxWords: number; keepAllUpTo: number };
  /**
   * The title, in message order. A number keeps that many best-scored words
   * (the 1.0 decoding). Otherwise: among the six best, those scored at least
   * `threshold`, never fewer than `minWords`, and every word when the message
   * has at most `keepAllUpTo` of them.
   */
  title(message: string, options?: number | TitleOptions): string;
}

/** Build a Titler from an ArrayBuffer or a Promise of one. */
export function loadTitler(buffer: ArrayBuffer | Promise<ArrayBuffer>): Promise<Titler>;
