// What every runtime shares: the titler class, where the model file sits, and
// how to turn whatever the caller hands us into bytes.
//
// Reading a file from disk is Node-only and lives in index.node.js, picked by
// the "node" condition of package.json. The portable entry (index.js) is the
// default on purpose: a tool that knows nothing about export conditions gets
// the one that builds everywhere, instead of failing on a Node built-in.
export { Titler } from "./titler.js";
import { Titler } from "./titler.js";

/** URL of the bundled model file, as resolved from this module. */
export const modelUrl = new URL("../model/titler-v1.1-latin-mini-int3.bin", import.meta.url);

/** Languages covered by this model (ISO 639-1). */
export const languages = ["en", "fr", "es", "de", "pt", "it", "nl", "pl", "sv", "da", "fi", "cs", "ro", "no", "hu", "hr", "lt"];

/** Fetch a model file over the network. */
export async function fetchBytes(url) {
  const response = await fetch(url);
  if (!response.ok) throw new Error("could not load the titler model from " + url + " (" + response.status + ")");
  return response.arrayBuffer();
}

/** Anything a caller may reasonably hand us, as an ArrayBuffer. */
export async function toBytes(source) {
  const value = await source;
  if (value instanceof ArrayBuffer) return value;
  if (ArrayBuffer.isView(value)) return value.buffer.slice(value.byteOffset, value.byteOffset + value.byteLength);
  if (typeof Response !== "undefined" && value instanceof Response) return value.arrayBuffer();
  if (value instanceof URL || typeof value === "string") return fetchBytes(value);
  throw new TypeError("loadTitler() takes a URL, a string, an ArrayBuffer, a typed array or a Response");
}

/** Builds the package's loadTitler: no argument = the bundled model, cached. */
export function loader(readBundled) {
  let cached;
  return async function loadTitler(source) {
    if (source === undefined) return (cached ??= readBundled().then((bytes) => new Titler(bytes)));
    return new Titler(await toBytes(source));
  };
}
