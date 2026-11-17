# dsh-mtool-translate

把 MTool 导出的 `ManualTransFile.json` 交给当前配置的模型翻译，产出 MTool 能直接加载回来的 JSON。
不装 AiNiee、不配第二套 API Key、不跑 Python——**只要 MTool + DSH**。

MTool 的收费翻译链路本来就是：导出原文 → 送翻 → 加载译文。这个插件只替换中间那一步。

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
| `Backend/domain/responses/response_extractor.py` | `<textarea>` + `N.译文` 回包解析、丢弃序号前的废话 |
| `Backend/domain/responses/advanced_checks.py` | 占位符是否齐全、换行符数是否一致、返回原文 / 翻译残留判定 |
| `Backend/domain/prompts/glossary_helper.py` | 术语按有无空格/间隔点拆词匹配，不区分大小写 |
| `Backend/domain/text/text_symbol_repair.py` | 引号成对替换、内部引号、省略号/破折号/问号修复 |
| `Backend/resources/prompts/Translate/common_system_zh.txt` | 系统提示词结构与 `<textarea>` 输出约定 |

自己写的部分只有壳：DSH 工具注册、模型线路、分批并发、缓存、断点落盘、文件读写。

## 和 AiNiee 的差异

- **回包协议**：主用 `textarea + N.`（同 AiNiee），另外保留 JSON 数组兜底；AiNiee 没有 JSON 兜底。
- **占位符上限**：AiNiee 单请求最多替换 50 个占位符，超过就原样保留、不再遮蔽；这里不设上限。
- **批次划分**：AiNiee 由它的任务调度器切块；这里按「30 行 / 1200 字」贪心打包。
- **校验口径**：「返回原文」检查在 AiNiee 里是整批 Jaccard 相似度 ≥ 0.85，这里是逐行按原文非汉字字符集的覆盖率判定（因为这里逐行重试）。且这两项都默认关闭，和 AiNiee 的开关默认值一致。
- **没有搬**：译前/译后替换的正则行（只支持纯字符串表）、禁翻表的正则行、Sakura/本地模型的 `↓` 占位、EPUB 标记、数字序号 `1.` 预处理、多行块的 `N.N.` 子编号、译后段落级引号修复（stage 4）。
- **没有搬 GUI**：AiNiee 的 Qt 界面、配置面板、术语表编辑页都在插件里由 DSH 对话取代。

## 配置

`cordis.patch.yml` 里给插件行传 `config`，各项都可选：

```yaml
- id: mtool-translate
  config:
    target: 简体中文
    sourceLanguage: 日语
    model: deepseek-flash      # 不填则跟随「设置」里的默认模型
    reasoningEffort: low       # off | low | high | max
    batchLines: 30
    batchChars: 1200
    concurrency: 3
    maxRetries: 2
    symbolRepair: true
    checkReturnedOriginal: false
    checkResidualOriginal: false
    exclusion: []              # 禁翻表，整行匹配就跳过
    preReplace: {}             # 译前替换 {"原文": "替换"}
    postReplace: {}            # 译后替换
    glossary: {"エルフ": "精灵"}  # 也可以用工具参数传
```

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
