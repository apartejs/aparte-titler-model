# Security

aparte-titler runs entirely on the user's device: the runtime is a single dependency-free file, the model is a static data file, and no message ever leaves the browser or the process.

The model is extractive: its output is always a subset of the words of its input. It cannot execute instructions found in a message; the worst an injected instruction can do is a poor title.

If you find a vulnerability in the runtime (for instance a malformed model file crashing a page), please report it privately through [apartejs.dev](https://apartejs.dev/) rather than in a public issue. We aim to answer within a week.
