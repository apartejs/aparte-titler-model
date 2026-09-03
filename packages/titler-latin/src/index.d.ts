import { Titler } from "./titler.js";

export { Titler } from "./titler.js";

/** URL of the bundled model file. */
export const modelUrl: URL;

/** Languages covered by this model (ISO 639-1 codes). */
export const languages: string[];

/** Load the bundled model once and return a ready Titler. */
export function loadTitler(): Promise<Titler>;
