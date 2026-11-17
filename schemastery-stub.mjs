/**
 * Stand-in for `@deepseek-ai/schemastery`, used only by `smoke.mjs`.
 *
 * The plugin imports the real package to declare its Loader `Config`; a plain
 * `node smoke.mjs` run cannot resolve packages from the DSH installation, so
 * the self-test maps the specifier here. Only the builder API is needed —
 * nothing in the pipeline reads the schema.
 */
const chain = () => {
  const api = {
    default: () => api,
    description: () => api,
    step: () => api,
    min: () => api,
    max: () => api,
    required: () => api,
    hidden: () => api,
    volatile: () => api,
  }
  return api
}

const builder = () => chain()

export default {
  object: () => chain(),
  string: builder,
  number: builder,
  boolean: builder,
  union: builder,
  const: builder,
  any: builder,
  array: builder,
}
