import { Titler } from "./titler.js";

export { Titler } from "./titler.js";

/** URL of the bundled model file, as resolved from the package. */
export const modelUrl: URL;

/** Languages covered by this model (ISO 639-1 codes). */
export const languages: string[];

/**
 * Load a model and return a ready Titler.
 *
 * With no argument, the model bundled in this package: read from disk in Node,
 * fetched from `modelUrl` elsewhere. Once your app is bundled that URL no
 * longer points at the package, so pass the model in yourself — a URL of the
 * file you serve, or its bytes.
 */
export function loadTitler(source?: URL | string | ArrayBuffer | ArrayBufferView | Response | Promise<ArrayBuffer | Response>): Promise<Titler>;
