# TeamMember

TeamMember 是一个面向“团队成员协作 + 知识融合 + 记忆与画像”的自托管系统。

它解决两个核心问题：

1) 团队知识：把任何数据源（SQL / OData / 手工粘贴等）的内容，通过大模型按语义拆成“知识点”种子（seed），写入向量库，按需参与 RAG。  
2) 团队协作：每个成员独立账号登录；会话可共享并设置权限；系统会把对话自动提炼为“用户画像”和“长期记忆”，在每次对话中自动融合（同时尽量不拖慢响应）。
3) 语义状态：系统会异步维护线程级的“未闭合事项(open loops)”与“主题收敛程度(topic entropy)”，用于减少“这个/那个”这类指代歧义，并在必要时自动反问收敛问题。
4) 纠错回写：支持把对话中的纠错/沉淀提交为 Teaching，管理员审核通过后写入知识库；检索时会对 `source_kind=teaching` 做轻量加权（可配置）。
5) 管理审计：管理员对系统的关键变更（配置、权限、审核操作）会写入 append-only 审计日志。应用层不提供修改接口，数据库层也禁止对审计表做 UPDATE/DELETE（只允许 INSERT）。
6) 按需 Web Search：当路由器判断需要外部公开线索时，可调用 Bing Search API 获取“标题/摘要/URL”（不抓正文），并与 RAG 一起用于回答（可开关）。
7) 子 Agent 拆分：当路由器判断任务复杂度较高时，可把任务拆成最多 5 个子任务，分别由子 Agent（同一个模型）处理，再综合成最终答复（可调“更少拆分 / 更多拆分”）。
8) AI Trace（30 天）：每次对话可记录“路由决策 / Web Search 摘要 / 子任务拆分 / 子 Agent 结果”等调试信息，普通用户可在 UI 查看（可开关，自动清理）。

## 一键启动

前置：

- Docker Desktop（Windows）或 Docker Engine（Linux）
- API Key：
  - 火山方舟：`ARK_API_KEY`
  - 阿里 DashScope：`DASHSCOPE_API_KEY`（用于 embedding + 识图）

步骤：

1) 将 `.env.example` 复制为 `.env`，填入 `ARK_API_KEY` 和 `DASHSCOPE_API_KEY`  
2) 运行：`docker compose up -d --build`  
3) 打开：`http://localhost:8080`  

后端 Swagger：`http://localhost:8000/docs`

提示：管理员账号默认是“第一个注册的用户”。管理员在右上角可以打开 `Admin` 面板，配置全局参数、管理用户、审核 Teaching、配置数据源（SQL / OData / 手工粘贴）。

## Web Search（Bing，摘要模式）

用途：补充“外部公开线索/链接/最新信息”，仅使用 Bing 返回的 `title/snippet/url` 摘要，不抓取网页正文。

启用方式：

1) 在 `.env` 中填写 `BING_SEARCH_API_KEY`（没有 Key 时，即使 UI 勾选也不会实际搜索）  
2) 前端右上角 `Admin` 面板里勾选 `Web Search` 并保存  

相关运行时配置（Admin 面板）：

- `web_search_enabled`：是否允许按需 Web Search
- `web_search_top_k`：每个 query 返回的摘要条数（1~10）
- `web_search_max_queries`：每次最多发起的 query 数（1~5）

说明：

- Web Search 的结果可能过时/不完整：后端系统提示词已要求模型“不要被 Web 摘要绑架”，必须给出验证步骤或反问收敛。

## 子 Agent 拆分（最多 5 个子任务）

用途：当问题涉及多个目标/多系统/多分支时，先拆分成子任务（<=5），分别得到结果，再综合输出。

相关运行时配置（Admin 面板）：

- `agent_decompose_policy`：`auto | force_on | force_off`
- `agent_decompose_bias`：0..100（越高越倾向拆分）
- `agent_max_subtasks`：子任务上限（默认 5）

## AI Trace（普通用户可见，30 天自动清理）

用途：用于展示“系统到底调用了什么工具/怎么拆分”，便于 debug 和复盘。它不是安全审计（安全审计请看 Audit Log）。

查看方式：

- UI：对话中每条 AI 消息右上角会出现 `工具/Trace` 按钮（打开后可查看路由决策、Web 摘要、拆分与子 Agent 结果）。
- API：`GET /api/threads/{thread_id}/messages/{message_id}/trace`

保留策略：

- 默认保留 30 天；后端维护线程每天清理过期记录。
- 可通过 Admin 面板修改 `ai_trace_retention_days`。

## 管理审计（Audit Log）

审计的目的：记录管理员对系统做过的关键变更，便于追溯与复盘。

特性：

- 只记录管理员变更（目前覆盖：`/admin/config`、`/admin/users/*`、Teaching 审核 approve/reject）。
- append-only：应用层无修改/删除接口；数据库触发器阻断对审计表的 UPDATE/DELETE。
- 哈希链：每条记录包含 `prev_hash` 与 `event_hash`，可用于发现“中间被篡改/被删”的痕迹。

查看方式：

- 前端：右上角 `Admin` 面板里有 “审计日志（只增不改）”。
- API：`GET /api/admin/audit?limit=200`（仅管理员可访问）。

边界说明：

- 该机制能防止“应用内管理员”篡改审计记录。
- 但如果有人拥有 Postgres 超级用户权限或能直接进容器执行 SQL，仍可能绕过数据库触发器。
  若需要更强的不可抵赖性，建议把每日的 `event_hash` 锚定到外部系统（Power Automate/Webhook、对象存储不可变策略等）。

## 被墙/网络问题处理（仍用 Docker）

如果你遇到类似 `failed to fetch anonymous token`（拉取基础镜像失败），优先用下面两种方式解决。

### 方式 A：配置 Docker 镜像加速（推荐）

在 Docker Desktop 的设置里配置 `registry-mirrors`（或你公司/自建的镜像仓库），让 `docker pull` 能稳定成功。

### 方式 B：在 `.env` 指定镜像源/基础镜像

本项目支持通过 `.env` 覆盖镜像与构建基础镜像：

- `POSTGRES_IMAGE`
- `QDRANT_IMAGE`
- `PYTHON_BASE_IMAGE`
- `NODE_BASE_IMAGE`
- `NGINX_BASE_IMAGE`

示例（仅示意，按你可用的镜像仓库替换）：

- `PYTHON_BASE_IMAGE=m.daocloud.io/library/python:3.11-slim`
- `NODE_BASE_IMAGE=m.daocloud.io/library/node:20-alpine`
- `NGINX_BASE_IMAGE=m.daocloud.io/library/nginx:1.27-alpine`

### 包管理镜像（可选）

如果 `pip/npm` 下载慢或失败，也可以在 `.env` 里设置：

- `PIP_INDEX_URL` / `PIP_TRUSTED_HOST`
- `NPM_REGISTRY`

## 架构图（Mermaid）

```mermaid
flowchart LR
  %% TeamMember - System Architecture (V1+)

  subgraph Client["Client (Browser)"]
    UI["React/Vite Web UI<br/>Threads • Chat(Stream) • Admin • Trace"]
  end

  subgraph Edge["Edge / Reverse Proxy (Docker Compose)"]
    Nginx["Nginx (serves UI + /api reverse proxy)"]
  end

  subgraph Backend["Backend (FastAPI)"]
    API["HTTP API<br/>Auth • Threads • Shares • Chat(Stream)<br/>Admin Config • Teaching Review • Trace"]
    Guards["Guardrails<br/>anti prompt-leak • anti persona/decision disclosure"]

    Router["Intent Router / Tool Router<br/>RAG/Web/Decompose decision"]
    Decomp["Task Decomposer<br/><= 5 subtasks"]
    SubAgents["Sub-agent Workers<br/><= 5 (same model)"]

    Rag["RAG Orchestrator<br/>Retrieve • Optional Rerank • Self-check"]
    Web["Web Search Tool (Bing)<br/>summary-only: title/snippet/url"]
    Gen["LLM Generation (Ark/DeepSeek)<br/>Answer • Rewrite • Clarify Questions • Synthesis"]

    Chunker["Semantic Chunking (LLM)<br/>any source -> knowledge-point seeds"]
    Memory["Memory Service<br/>Memory snippets • Persona"]
    Decision["Decision Profile Builder<br/>(idle refresh every 5 days)"]
    TState["Thread Semantic State<br/>open loops • topic entropy"]
    Vision["Image Understanding (Qwen-VL)<br/>screenshot -> structured JSON"]
    Embed["Embedding (DashScope)<br/>text-embedding-v2 (1536d)"]

    Trace["AI Trace (debug)<br/>router/web/decompose/subagent<br/>(30-day retention)"]
    Audit["Admin Audit Log<br/>append-only + hash chain"]
    Maint["Maintenance Thread<br/>03:15 daily cleanup/refresh"]
    Notify["Webhook Notifier (optional)<br/>POST to Power Automate"]
  end

  subgraph Storage["Storage"]
    PG["PostgreSQL<br/>users • threads • chat_messages<br/>thread_state • app_config<br/>audit_logs • ai_trace_runs • teaching queue..."]
    Qdrant["Qdrant Vector DB<br/>knowledge seeds • user memory"]
  end

  subgraph External["External Services"]
    Ark["Volcengine Ark (OpenAI-compatible)<br/>DeepSeek model"]
    DashScope["DashScope (Alibaba)<br/>Embedding + Qwen-VL"]
    Bing["Bing Search API"]
    Sources["Data Sources<br/>SQL • OData • Manual Paste"]
  end

  %% Client -> Edge -> Backend
  UI --> Nginx --> API

  %% Chat pipeline
  API --> Guards --> Router
  Router -->|need RAG?| Rag --> Embed --> DashScope
  Rag --> Qdrant
  Rag -->|optional rerank| Gen --> Ark
  Router -->|need Web?| Web --> Bing
  Router -->|complex?| Decomp --> SubAgents --> Ark
  Rag --> Gen
  Web --> Gen
  SubAgents --> Gen

  %% Context & background analyzers
  API --> Memory --> PG
  API -. async .-> TState --> PG
  Maint -. idle .-> Decision --> PG
  Vision --> DashScope

  %% Trace & audit
  API --> Trace --> PG
  API --> Audit --> PG
  Maint -->|purge expired traces| PG

  %% Knowledge ingestion
  Sources --> API --> Chunker --> Ark
  Chunker --> Embed --> Qdrant
  API --> Qdrant
  API -. optional .-> Notify
```
