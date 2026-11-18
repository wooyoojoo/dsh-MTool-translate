/**
 * Browser half of dsh-mtool-translate: the Plugins page configuration section.
 *
 * The slot is declared by the plugin manager and keyed by the bundle's package
 * name, so this page appears on the bundle's own page without the manager
 * naming any plugin. The form is generated from the field table the host
 * reports, so a setting added on the host side shows up here without a second
 * edit — the host is the single source of truth for keys, kinds, ranges, and
 * descriptions.
 *
 * It is written in the artifact format the client module loader executes: a
 * classic script calling `window.__ModuleLoader__.load({ id, factory })`, so
 * the bundle needs no build step. Only `react` is requested from the module
 * table; the styling is inline and every colour is a `--dsw-*` token, so both
 * themes match the shell.
 */

window.__ModuleLoader__.load({
  id: 'dsh-mtool-translate',
  factory: (require) => {
    var module = { exports: {} }
    var exports = module.exports
    Object.defineProperty(exports, Symbol.toStringTag, { value: 'Module' })
    const React = require('react')
    const h = React.createElement

    /** Package name this bundle is installed as; it keys the Plugins-page slot. */
    const BUNDLE = 'dsh-mtool-translate'

    /** Locale namespace owned by this bundle. */
    const NS = 'mtool-translate'

    /** How long an in-place notice stays before it clears itself. */
    const NOTICE_MS = 6000

    const zh = {
      title: 'MTool 文本翻译设置',
      loading: '正在读取设置…',
      loadFailed: '读取设置失败：{detail}',
      save: '保存',
      saving: '保存中…',
      saved: '已保存，下次翻译生效',
      nothingToSave: '没有改动',
      reset: '恢复默认',
      resetting: '恢复中…',
      resetDone: '已恢复默认值',
      saveFailed: '保存失败：{detail}',
      note: '这些设置保存在 ~/.dsh/mtool-translate/settings.json，改完立即生效，不需要重启。命令行里也可以让助手改。',
      unset: '未设置',
    }

    const en = {
      title: 'MTool text translation settings',
      loading: 'Reading settings…',
      loadFailed: 'Could not read the settings: {detail}',
      save: 'Save',
      saving: 'Saving…',
      saved: 'Saved; the next translation uses it',
      nothingToSave: 'Nothing changed',
      reset: 'Reset to defaults',
      resetting: 'Resetting…',
      resetDone: 'Defaults restored',
      saveFailed: 'Could not save: {detail}',
      note: 'Stored in ~/.dsh/mtool-translate/settings.json; changes apply immediately, no restart. You can also ask the agent to change them.',
      unset: 'unset',
    }

    /** Substitute `{name}` placeholders without the locale service. */
    const interpolate = (template, params) => params === undefined
      ? template
      : template.replace(/\{(\w+)\}/g, (match, name) => (name in params ? String(params[name]) : match))

    const S = {
      stack: { display: 'flex', flexDirection: 'column', gap: '14px' },
      field: { display: 'flex', flexDirection: 'column', gap: '4px' },
      label: { fontSize: '13px', fontWeight: 500 },
      note: { fontSize: '12px', opacity: 0.6 },
      row: { display: 'flex', alignItems: 'center', gap: '8px', flexWrap: 'wrap' },
      muted: { fontSize: '13px', opacity: 0.7 },
      problem: { fontSize: '13px', color: 'var(--dsw-alias-state-error-primary, #d9544d)' },
      heading: { fontSize: '14px', fontWeight: 600, margin: 0 },
      input: {
        boxSizing: 'border-box',
        height: '30px',
        padding: '0 8px',
        fontSize: '13px',
        color: 'var(--dsw-alias-label-primary)',
        background: 'var(--dsw-alias-bg-layer-1)',
        border: '0.5px solid var(--dsw-alias-border-l4)',
        borderRadius: '6px',
        outline: 'none',
      },
      button: {
        height: '28px',
        padding: '0 10px',
        fontSize: '12px',
        borderRadius: '6px',
        border: 'none',
        cursor: 'pointer',
        color: 'var(--dsw-alias-label-primary-foreground)',
        background: 'var(--dsw-alias-button-primary-fill)',
      },
      ghost: {
        height: '28px',
        padding: '0 10px',
        fontSize: '12px',
        borderRadius: '6px',
        border: '0.5px solid var(--dsw-alias-border-l3)',
        cursor: 'pointer',
        color: 'var(--dsw-alias-label-primary)',
        background: 'transparent',
      },
      checkRow: { display: 'flex', alignItems: 'center', gap: '8px' },
    }

    /** The message of a thrown value, whatever was thrown. */
    const detailOf = (error) => (error instanceof Error ? error.message : String(error))

    /** Absolute URL of one host route, resolved against the deployment base. */
    function routeUrl(path) {
      return new URL(`api/mtool-translate/${path}`, document.baseURI).href
    }

    /**
     * Read one JSON route, turning the host's `{ ok: false, error }` envelope
     * and any transport failure into one thrown Error the caller can show.
     */
    async function readJson(url, init) {
      const response = await fetch(url, init)
      let payload = null
      try {
        payload = await response.json()
      } catch {
        throw new Error(`HTTP ${response.status}`)
      }
      if (payload !== null && typeof payload === 'object' && payload.ok === false) {
        throw new Error(String(payload.error ?? `HTTP ${response.status}`))
      }
      if (!response.ok) throw new Error(`HTTP ${response.status}`)
      return payload
    }

    /** POST one JSON body to a host route. */
    const postJson = (path, body) => readJson(routeUrl(path), {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify(body),
    })

    /** An in-place notice that clears itself; repeated text restarts the hold. */
    function useNotice() {
      const [notice, setNotice] = React.useState(null)
      React.useEffect(() => {
        if (notice === null) return undefined
        const timer = setTimeout(() => { setNotice(null) }, NOTICE_MS)
        return () => { clearTimeout(timer) }
      }, [notice])
      const notify = React.useCallback((text) => { setNotice({ seq: Date.now(), text }) }, [])
      return { notice, notify }
    }

    /** The control one field kind needs. */
    function Control({ field, value, onChange, id }) {
      if (field.kind === 'boolean') {
        return h('div', { style: S.checkRow },
          h('input', {
            id,
            type: 'checkbox',
            checked: value === true,
            onChange: event => { onChange(event.target.checked) },
          }),
          h('span', { style: S.note }, value === true ? 'true' : 'false'))
      }
      if (field.kind === 'enum') {
        return h('select', {
          id,
          style: S.input,
          value: typeof value === 'string' ? value : '',
          onChange: event => { onChange(event.target.value) },
        }, ...field.values.map(option => h('option', { key: option, value: option }, option)))
      }
      if (field.kind === 'number' || field.kind === 'integer') {
        return h('input', {
          id,
          style: S.input,
          type: 'number',
          min: field.min,
          max: field.max,
          step: 1,
          value: value === undefined || value === null ? '' : String(value),
          onChange: event => {
            const raw = event.target.value
            onChange(raw === '' ? undefined : Number(raw))
          },
        })
      }
      return h('input', {
        id,
        style: S.input,
        type: 'text',
        spellCheck: false,
        value: value === undefined || value === null ? '' : String(value),
        onChange: event => { onChange(event.target.value) },
      })
    }

    /**
     * The bundle's configuration section: what the host reports now, the drafts
     * this page holds, and the one write that carries the difference.
     */
    function SettingsSection(props) {
      const t = props.t
      const [status, setStatus] = React.useState(null)
      const [drafts, setDrafts] = React.useState({})
      const [problem, setProblem] = React.useState(null)
      const [busy, setBusy] = React.useState(null)
      const { notice, notify } = useNotice()

      const adopt = React.useCallback((next) => {
        setStatus(next)
        setDrafts({ ...next.settings })
        setProblem(null)
      }, [])

      React.useEffect(() => {
        let live = true
        readJson(routeUrl('status'))
          .then(payload => { if (live) adopt(payload) })
          .catch(error => { if (live) setProblem(detailOf(error)) })
        return () => { live = false }
      }, [adopt])

      const edit = React.useCallback((key, value) => {
        setDrafts(previous => ({ ...previous, [key]: value }))
      }, [])

      const save = React.useCallback(async () => {
        if (status === null) return
        const changes = {}
        for (const [key, value] of Object.entries(drafts)) {
          if (value === undefined) continue
          if (JSON.stringify(value) !== JSON.stringify(status.settings[key])) changes[key] = value
        }
        if (Object.keys(changes).length === 0) {
          notify(t('nothingToSave'))
          return
        }
        setBusy('save')
        try {
          adopt(await postJson('update', { settings: changes }))
          notify(t('saved'))
        } catch (error) {
          notify(t('saveFailed', { detail: detailOf(error) }))
        } finally {
          setBusy(null)
        }
      }, [status, drafts, adopt, notify, t])

      const reset = React.useCallback(async () => {
        setBusy('reset')
        try {
          await postJson('update', { unset: status.fields.map(field => field.key) })
          adopt(await readJson(routeUrl('status')))
          notify(t('resetDone'))
        } catch (error) {
          notify(t('saveFailed', { detail: detailOf(error) }))
        } finally {
          setBusy(null)
        }
      }, [status, adopt, notify, t])

      if (problem !== null) return h('div', { style: S.problem }, t('loadFailed', { detail: problem }))
      if (status === null) return h('div', { style: S.muted }, t('loading'))

      return h('div', { style: S.stack },
        h('h3', { style: S.heading }, t('title')),
        notice === null ? null : h('div', { role: 'status', style: S.note }, notice.text),
        ...status.fields.map(field => h('div', { key: field.key, style: S.field },
          h('label', { htmlFor: `mtool-${field.key}`, style: S.label }, field.key),
          h(Control, {
            field,
            id: `mtool-${field.key}`,
            value: drafts[field.key],
            onChange: value => { edit(field.key, value) },
          }),
          field.description === undefined || field.description === ''
            ? null
            : h('span', { style: S.note }, field.description))),
        h('div', { style: S.row },
          h('button', { style: S.button, disabled: busy !== null, onClick: () => { void save() } },
            busy === 'save' ? t('saving') : t('save')),
          h('button', { style: S.ghost, disabled: busy !== null, onClick: () => { void reset() } },
            busy === 'reset' ? t('resetting') : t('reset'))),
        h('div', { style: S.note }, t('note')),
      )
    }

    /**
     * Claim the Plugins page's configuration section for this bundle.
     *
     * @param ctx - client plugin context.
     */
    function apply(ctx) {
      const locale = ctx.get('locale')
      const fallback = (navigator.language ?? '').toLowerCase().startsWith('zh') ? zh : en
      let t = (key, params) => interpolate(fallback[key] ?? en[key] ?? key, params)
      if (locale !== undefined && typeof locale.bind === 'function' && typeof locale.register === 'function') {
        ctx.effect(() => locale.register(NS, { zh, en }), 'mtool-translate: dictionaries')
        t = locale.bind(NS)
      }
      const inject = () => ({ t })
      ctx.slots.inject('plugins.bundle.config', () => ctx.slots.register(
        { name: 'plugins.bundle.config', key: BUNDLE, inject },
        SettingsSection,
      ))
    }

    const inject = ['slots']
    exports.apply = apply
    exports.inject = inject
    return module.exports
  },
})
