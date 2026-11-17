# dsh-mtool-translate

把 MTool 导出的 `ManualTransFile.json` 交给当前配置的模型翻译，产出 MTool 能直接加载回来的 JSON。
不装 AiNiee、不配第二套 API Key、不跑 Python——**只要 MTool + DSH**。

MTool 的收费翻译链路本来就是：导出原文 → 送翻 → 加载译文。这个插件只替换中间那一步。

## 安装

```bash
git clone https://github.com/wooyoojoo/dsh-MTool-translate.git
cd dsh-MTool-translate
pnpm install --prod     # 配置表单依赖 @deepseek-ai/schemastery
```

然后在 DSH 里「插件 → 添加插件」选择该目录。`pnpm install` 不能省：profile 安装 linked 包时不会替它装依赖，缺了 schemastery 插件会加载失败。

## 用法

1. MTool 里打开游戏，`翻译` → `导出游戏原文`，游戏根目录出现 `ManualTransFile.json`。
2. 把该文件拖进 DSH 对话，说「翻译这个」。
3. 助手调用 `translate_mtool_file`，产出 `ManualTransFile.translated.json`（与原文同目录，`indent=4`、不转义非 ASCII，和 AiNiee 的写出格式一致）。
4. 回到 MTool，`翻译` → `加载翻译文件`，选 `ManualTransFile.translated.json`。

再翻一次同样的文件不会重复花钱：已翻译的行、以及缓存里有的行都会跳过。

没有现成游戏想先试试：`samples/ManualTransFile.json` 是一份带控制码的小样。

> 从聊天窗口拖进来的文件会落在 DSH 的附件目录，产出也写在那儿。想让它落在工作目录里，调用时传 `outputPath`。

## 工具参数

| 参数 | 说明 |
| --- | --- |
| `path` | 必填，`ManualTransFile.json` 的路径 |
| `target` | 目标语言，默认 `简体中文` |
| `glossary` | 术语表，`"原文=>译文"` 数组，也可以传 JSON 文件路径 |
| `outputPath` | 输出路径，默认 `<输入>.translated.json` |
| `limit` | 本次最多翻多少行，用来先跑个小样 |
| `maxBatches` | 本次最多发多少个请求，剩下的下次再跑 |
| `force` | 连已有译文的行也重翻 |
| `dryRun` | 只统计待翻/缓存/已完成/跳过多少行，不调用模型 |

## 从 AiNiee 搬过来的部分

代码是按文件逐个搬的行为，不是重写：

| 文件 | 搬了什么 |
| --- | --- |
| `Backend/resources/regex/regex.json` | 50 多条 RPG Maker / Wolf / KRKR 控制码正则，顺序照抄 |
| `Backend/domain/text/text_processor.py` | `[Pn]` 占位符（整批共享计数）、逆序还原、非日文字符前后缀剥离 |
| `Backend/domain/text/filters/general_text_filter.py` | 跳过规则：纯数字、空行、纯标点、文件后缀、代码前缀 |
| `Backend/domain/text/filters/code_text_filter.py` | 代码/着色器片段不送翻；规则按 GLSL/JS/CSS 标记和标点密度重写（MTool 会把插件内嵌 shader 也导出来，必须挡） |
| `Backend/domain/responses/response_extractor.py` | `<textarea>` + `N.译文` 回包解析、丢弃序号前的废话 |
| `Backend/domain/responses/advanced_checks.py` | 占位符是否齐全、换行符数是否一致、返回原文 / 翻译残留判定 |
| `Backend/domain/prompts/glossary_helper.py` | 术语按有无空格/间隔点拆词匹配，不区分大小写 |
| `Backend/domain/text/text_symbol_repair.py` | 引号成对替换、内部引号、省略号/破折号/问号修复 |
| `Backend/resources/prompts/Translate/common_system_zh.txt` | 系统提示词结构与 `<textarea>` 输出约定 |

自己写的部分只有壳：DSH 工具注册、模型线路、分批并发、缓存、断点落盘、文件读写。

## 和 AiNiee 的差异

已经对齐的（这些以前不一样，现在一致）：

- 系统提示词用 `common_system_zh.txt` 原文；用户消息是 `###上文内容 + ###这是你接下来的翻译任务，原文文本如下 + <textarea>` 外壳。
- 多行条目用 `N.[ "N.k.,行", … ]` 子编号块发出、按块收回。
- 数字序号 `1.` 先转 `【1】` 再还原，避免和多行协议的序号打架。
- 回包序号必须是 1..n 连续递增（同 `check_dict_order`），错行直接重试而不是硬塞。
- 控制码正则表与 `regex.json` 逐条一致（54 条，脚本比对过）。
- 译前/译后替换支持正则行（`{src, dst, regex: true}`），也兼容纯字符串表。
- 禁翻表支持整行精确匹配和正则，且只把当前批次实际命中的词条发给模型。
- 角色表 / 背景设定 / 翻译风格 / 翻译示例按 AiNiee 的段落格式拼进系统提示词。
- `promptPreset: cot` 用 `cot_system_zh.txt`；`targetPlatform: sakura` 时占位符改成 `↓` 连写。
- 多行块里模型给每行都加引号时，按原文首尾字符决定保留或去掉（stage 4）。

还没对齐 / 不打算搬的：

- **回包兜底**：AiNiee 只认 `textarea + N.`；这里多留了一个 JSON 数组兜底（模型不听话时救急）。
- **占位符上限**：AiNiee 单请求最多替换 50 个占位符，超过就不再遮蔽；这里不设上限（长批次更安全）。
- **批次划分**：AiNiee 由任务调度器切块；这里按「40 行 / 1600 字」贪心打包，默认 8 并发。
- **校验口径**：「返回原文」在 AiNiee 里是整批 Jaccard ≥ 0.85，这里是逐行按原文非汉字字符覆盖率 ≥ 0.9（因为这里逐行重试）。「翻译残留」同理。这两项都默认关。
- **没搬**：术语表自动抽取（AiNiee 的 analysis 任务）、由分析任务生成的项目级角色表/术语表/禁翻表、EPUB 标记、`local` 提示词、`{p}` 复合占位的 padding 变体（套 `\s*…\s*` 会把换行一起吞掉，实测会破坏换行校验）。
- **没有 GUI**：AiNiee 的 Qt 界面、配置面板、术语表编辑页都由 DSH 对话取代。

## 配置

**在界面里改**：设置 → 插件 → `dsh-mtool-translate`（点进 bundle 页面，表单在描述和插件行之间）→ 改完点「保存」，**下次翻译立即生效，不用重启**。

页面上只有两项：

- **推理强度**：`off` / `low` / `high` / `max`（默认 `low`）
- **增量缓存**：保留 / 关闭（默认保留）

其余都是代码里的固定值，故意不给旋钮：

| 项 | 固定行为 |
| --- | --- |
| 源语言 | 从文件内容自动判断（假名→日语、谚文→韩语、西里尔→俄语、汉字→中文、否则英语） |
| 目标语言 | 由你在对话里点名（「日译中」「翻成繁中」「译成英文」），不点名默认简中 |
| 模型 | 跟随 DSH「设置 → 模型」里的默认线路，当前是 `deepseek-official / deepseek-flash` |
| 批次 | 每请求 40 行 / 1600 字，8 并发，重试 2 次，回复上限 8192 token |
| 输出位置 | 永远写在输入文件旁边 |

保存的值落在 `~/.dsh/mtool-translate/settings.json`，改完下次翻译立即生效；点「恢复默认」清掉覆盖。也可以在对话里说「推理开 high」，助手用 `mtool_translate_config` 写同一个文件。

YAML 里只有这两项可配：

```yaml
- id: mtool-translate
  config:
    reasoningEffort: low
    cacheEnabled: true
```

### 增量缓存

`.mtool-translate-cache.json` 是逐行对照表（`sha1(目标语言+原文) → 译文`），写在导出文件旁边。它只影响「下次再翻同一份/同一目录」，不影响 MTool 加载；关掉 `cacheEnabled` 后不再读写。删掉也安全，只是下次会重新付费翻译。注意它保存了译文原文，分享游戏目录时会一起带出去。

## 限制

- 只处理 MTool 导出的 JSON 格式（扁平 `原文: 译文`）。T++ 的 `.trans`、Ren'Py 的 `.rpy` 不在范围内。
- 运行时抓取（MTool 的 hook）不在范围内：本插件只做「文件进、文件出」，游戏仍需用 MTool 挂载译本。
- 译文长度不会自动适配窗口宽度；某行溢出就在 MTool 的翻译修正里改。
- 模型仍可能译错；插件保证的是**格式、控制码、行数不出错**，不是语义正确。

## 自检

```bash
node smoke.mjs
```

用桩 LLM 覆盖控制码遮蔽/还原、跳过规则、回包解析、逐行校验、符号修复、缓存、重试、断点续跑，不需要网络。
