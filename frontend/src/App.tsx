import { useEffect, useMemo, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import * as api from "./api";

type View = "login" | "app";

export default function App() {
  const [view, setView] = useState<View>("login");
  const [me, setMe] = useState<api.User | null>(null);
  const [threads, setThreads] = useState<api.Thread[]>([]);
  const [activeThreadId, setActiveThreadId] = useState<string>("");
  const [activeThreadUpdatedAt, setActiveThreadUpdatedAt] = useState<string>("");
  const [lastSeen, setLastSeen] = useState<Record<string, string>>({});
  const [thread, setThread] = useState<any>(null);
  const [showCanvas, setShowCanvas] = useState(false);
  const [messages, setMessages] = useState<api.ChatMessage[]>([]);
  const [chatInput, setChatInput] = useState("");
  const [streaming, setStreaming] = useState(false);
  const [error, setError] = useState("");
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [adminOpen, setAdminOpen] = useState(false);
  const [teachingOpen, setTeachingOpen] = useState(false);
  const [teachingInstruction, setTeachingInstruction] = useState("");
  const [teachingResult, setTeachingResult] = useState("");
  const [teachingSubmitting, setTeachingSubmitting] = useState(false);
  const [reviewDetailOpen, setReviewDetailOpen] = useState(false);
  const [reviewDetail, setReviewDetail] = useState<any>(null);
  const [traceOpen, setTraceOpen] = useState(false);
  const [traceLoading, setTraceLoading] = useState(false);
  const [traceData, setTraceData] = useState<api.AiTrace | null>(null);

  const [profileName, setProfileName] = useState("");
  const [profileText, setProfileText] = useState("");

  const [adminUsers, setAdminUsers] = useState<api.AdminUser[]>([]);
  const [adminConfig, setAdminConfig] = useState<any>(null);
  const [adminAudit, setAdminAudit] = useState<api.AdminAuditLog[]>([]);
  const [teachingReviews, setTeachingReviews] = useState<api.AdminTeachingReview[]>([]);
  const [teachingReviewStatus, setTeachingReviewStatus] = useState<"pending" | "approved" | "rejected" | "all">("pending");
  const [cfgRagPolicy, setCfgRagPolicy] = useState("auto");
  const [cfgRagTopK, setCfgRagTopK] = useState("10");
  const [cfgRagMaxContext, setCfgRagMaxContext] = useState("12");
  const [cfgRagTeachingScoreBoost, setCfgRagTeachingScoreBoost] = useState("0.05");
  const [cfgRagTeachingCandidates, setCfgRagTeachingCandidates] = useState("40");
  const [cfgWebSearchEnabled, setCfgWebSearchEnabled] = useState(false);
  const [cfgWebSearchTopK, setCfgWebSearchTopK] = useState("5");
  const [cfgWebSearchMaxQueries, setCfgWebSearchMaxQueries] = useState("2");
  const [cfgAgentDecomposePolicy, setCfgAgentDecomposePolicy] = useState("auto");
  const [cfgAgentDecomposeBias, setCfgAgentDecomposeBias] = useState("30");
  const [cfgAgentMaxSubtasks, setCfgAgentMaxSubtasks] = useState("5");
  const [cfgAiTraceEnabled, setCfgAiTraceEnabled] = useState(true);
  const [cfgAiTraceRetentionDays, setCfgAiTraceRetentionDays] = useState("30");
  const [cfgMemoryEnabled, setCfgMemoryEnabled] = useState(true);
  const [cfgMemoryTopK, setCfgMemoryTopK] = useState("5");
  const [cfgTopicGuardEnabled, setCfgTopicGuardEnabled] = useState(false);
  const [cfgTopicAllowedTopics, setCfgTopicAllowedTopics] = useState("");
  const [cfgPersonaDisclosureEnabled, setCfgPersonaDisclosureEnabled] = useState(false);
  const [cfgThreadStateEnabled, setCfgThreadStateEnabled] = useState(true);
  const [cfgThreadStateWindowMsgs, setCfgThreadStateWindowMsgs] = useState("60");
  const [cfgThreadStateCooldownSeconds, setCfgThreadStateCooldownSeconds] = useState("90");
  const [cfgDecisionProfileEnabled, setCfgDecisionProfileEnabled] = useState(true);
  const [cfgDecisionProfileRefreshHours, setCfgDecisionProfileRefreshHours] = useState("24");
  const [cfgProactiveEnabled, setCfgProactiveEnabled] = useState(false);
  const [cfgProactiveMinMsgs, setCfgProactiveMinMsgs] = useState("500");
  const [cfgProactiveWeekdayOnly, setCfgProactiveWeekdayOnly] = useState(true);
  const [cfgProactiveWorkStart, setCfgProactiveWorkStart] = useState("09:00");
  const [cfgProactiveWorkEnd, setCfgProactiveWorkEnd] = useState("18:00");
  const [cfgProactiveTimezone, setCfgProactiveTimezone] = useState("Asia/Shanghai");

  const [kstats, setKstats] = useState<any>(null);
  const [sources, setSources] = useState<api.DataSource[]>([]);
  const [pasteSourceName, setPasteSourceName] = useState("manual");
  const [pasteText, setPasteText] = useState("");
  const [pasteResult, setPasteResult] = useState("");
  const [sourceResult, setSourceResult] = useState("");
  const [shareAccount, setShareAccount] = useState("");
  const [sharePermission, setSharePermission] = useState<"read" | "write">("read");
  const [shareRows, setShareRows] = useState<api.ThreadShare[]>([]);

  const [srcEditingId, setSrcEditingId] = useState<string>("");
  const [srcKind, setSrcKind] = useState<"sql" | "odata">("sql");
  const [srcName, setSrcName] = useState("");
  const [srcSqlDbUrl, setSrcSqlDbUrl] = useState("");
  const [srcSqlQuery, setSrcSqlQuery] = useState("SELECT id, title, content FROM your_table");
  const [srcOdataUrl, setSrcOdataUrl] = useState("");
  const [srcOdataHeaders, setSrcOdataHeaders] = useState("{\n  \"Authorization\": \"Bearer <token>\"\n}");
  const [srcMaxItems, setSrcMaxItems] = useState("200");

  const [loginAccount, setLoginAccount] = useState("");
  const [loginPassword, setLoginPassword] = useState("");
  const [regPhone, setRegPhone] = useState("");
  const [regPassword, setRegPassword] = useState("");
  const [regName, setRegName] = useState("");

  const chatEndRef = useRef<HTMLDivElement | null>(null);

  const canEdit = useMemo(() => {
    if (!thread) return false;
    return thread.permission === "owner" || thread.permission === "write";
  }, [thread]);
  const canChat = canEdit;

  function lastSeenStorageKey(userId: string) {
    return `tm_last_seen_${userId}`;
  }

  function markThreadSeen(threadId: string, updatedAt: string) {
    if (!me?.id) return;
    const ts = updatedAt || new Date().toISOString();
    setLastSeen((prev) => {
      const next = { ...prev, [threadId]: ts };
      try {
        localStorage.setItem(lastSeenStorageKey(me.id), JSON.stringify(next));
      } catch { }
      return next;
    });
  }

  async function boot() {
    try {
      const u = await api.me();
      setMe(u);
      setView("app");
      await refreshThreads();
    } catch {
      setView("login");
    }
  }

  async function refreshThreads() {
    const ts = await api.listThreads();
    setThreads(ts);
    if (!activeThreadId && ts.length) {
      await openThread(ts[0].id, ts[0].updated_at);
    }
  }

  async function openThread(id: string, updatedAt?: string) {
    setActiveThreadId(id);
    setActiveThreadUpdatedAt(updatedAt || "");
    const t = await api.getThread(id);
    setThread(t);
    setShowCanvas(Boolean(String(t?.canvas_md || "").trim()));
    const ms = await api.listMessages(id);
    setMessages(ms);
    await refreshShares(id, String(t?.permission || ""));
    markThreadSeen(id, updatedAt || new Date().toISOString());
    setTimeout(() => chatEndRef.current?.scrollIntoView({ behavior: "smooth" }), 80);
  }

  useEffect(() => {
    boot();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    setTimeout(() => chatEndRef.current?.scrollIntoView({ behavior: "smooth" }), 60);
  }, [messages.length, streaming]);

  useEffect(() => {
    if (!me?.id) {
      setLastSeen({});
      return;
    }
    try {
      const raw = localStorage.getItem(lastSeenStorageKey(me.id)) || "{}";
      const obj = JSON.parse(raw);
      if (obj && typeof obj === "object") setLastSeen(obj);
    } catch {
      setLastSeen({});
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [me?.id]);

  useEffect(() => {
    if (!me?.id) return;
    if (!threads.length) return;
    if (Object.keys(lastSeen).length) return;
    // On a new device / cleared storage, avoid marking every existing thread as unread.
    const base: Record<string, string> = {};
    for (const t of threads) base[t.id] = t.updated_at;
    setLastSeen(base);
    try {
      localStorage.setItem(lastSeenStorageKey(me.id), JSON.stringify(base));
    } catch { }
  }, [me?.id, threads.length, lastSeen]);

  const unreadCount = useMemo(() => {
    const unseen = threads.filter((t) => {
      const seen = lastSeen[t.id];
      if (!seen) return true;
      return new Date(t.updated_at).getTime() > new Date(seen).getTime();
    });
    return unseen.length;
  }, [threads, lastSeen]);

  useEffect(() => {
    if (view !== "app" || !me?.id) return;
    let cancelled = false;
    const timer = window.setInterval(async () => {
      if (cancelled) return;
      if (streaming) return;
      try {
        const ts = await api.listThreads();
        if (cancelled) return;
        setThreads(ts);

        // active thread auto-refresh (only when updated_at changes)
        const activeId = activeThreadId;
        if (activeId) {
          const cur = ts.find((x) => x.id === activeId);
          if (cur && cur.updated_at && cur.updated_at !== activeThreadUpdatedAt) {
            const ms = await api.listMessages(activeId);
            if (cancelled) return;
            setMessages(ms);
            setActiveThreadUpdatedAt(cur.updated_at);
            markThreadSeen(activeId, cur.updated_at);
          }
        }
      } catch { }
    }, 3000);
    return () => {
      cancelled = true;
      window.clearInterval(timer);
    };
  }, [view, me?.id, activeThreadId, activeThreadUpdatedAt, streaming]);

  async function doLogin() {
    setError("");
    try {
      const u = await api.login(loginAccount, loginPassword);
      setMe(u);
      setProfileName(u.name || "");
      setView("app");
      await refreshThreads();
    } catch (e: any) {
      setError(String(e?.message || e));
    }
  }

  async function doRegister() {
    setError("");
    try {
      const u = await api.register(regPhone, regPassword, regName);
      setMe(u);
      setProfileName(u.name || "");
      setView("app");
      await refreshThreads();
    } catch (e: any) {
      setError(String(e?.message || e));
    }
  }

  async function newThread() {
    setError("");
    try {
      const t = await api.createThread("新会话");
      await refreshThreads();
      await openThread(t.id);
    } catch (e: any) {
      setError(String(e?.message || e));
    }
  }

  async function saveThread() {
    if (!thread) return;
    setError("");
    try {
      await api.updateThread(thread.id, { title: thread.title, canvas_md: thread.canvas_md });
      await refreshThreads();
    } catch (e: any) {
      setError(String(e?.message || e));
    }
  }

  async function refreshChat() {
    if (!activeThreadId) return;
    const ms = await api.listMessages(activeThreadId);
    setMessages(ms);
  }

  async function openTrace(messageId: string) {
    if (!activeThreadId) return;
    setError("");
    setTraceLoading(true);
    try {
      const tr = await api.getMessageTrace(activeThreadId, messageId);
      setTraceData(tr);
      setTraceOpen(true);
    } catch (e: any) {
      setError(String(e?.message || e));
    } finally {
      setTraceLoading(false);
    }
  }

  function openTeaching() {
    setTeachingInstruction("");
    setTeachingResult("");
    setTeachingOpen(true);
  }

  async function submitTeaching() {
    if (!activeThreadId) return;
    setTeachingSubmitting(true);
    setTeachingResult("");
    setError("");
    try {
      const res: any = await api.submitTeaching(activeThreadId, teachingInstruction, 80);
      setTeachingResult(`已提交审核：points=${res.points} review_id=${res.review_id}`);
      if (me?.is_admin) {
        await refreshTeachingReviews();
      }
    } catch (e: any) {
      setError(String(e?.message || e));
    } finally {
      setTeachingSubmitting(false);
    }
  }

  async function refreshShares(threadId: string, permission: string) {
    if (permission !== "owner") {
      setShareRows([]);
      return;
    }
    try {
      const rows = await api.listThreadShares(threadId);
      setShareRows(rows);
    } catch (e: any) {
      setShareRows([]);
      setError(String(e?.message || e));
    }
  }

  async function addShare() {
    if (!activeThreadId || !shareAccount.trim()) return;
    setError("");
    try {
      await api.shareThread(activeThreadId, shareAccount.trim(), sharePermission);
      setShareAccount("");
      await refreshShares(activeThreadId, "owner");
    } catch (e: any) {
      setError(String(e?.message || e));
    }
  }

  async function revokeShare(sharedUserId: string) {
    if (!activeThreadId) return;
    setError("");
    try {
      await api.deleteThreadShare(activeThreadId, sharedUserId);
      await refreshShares(activeThreadId, "owner");
    } catch (e: any) {
      setError(String(e?.message || e));
    }
  }

  function isWriteIntent(text: string): boolean {
    const t = (text || "").trim();
    if (!t) return false;
    // very small heuristic: only trigger when user clearly asks to write/generate/draft something
    return /(^|[\s，。！？])((帮我|请|麻烦)?(写|生成|起草|撰写|整理|输出))(一篇|一份|个|份|篇)?/.test(t) ||
      /(写一|生成一|起草一|撰写一|写个|写份|写篇|给我写)/.test(t) ||
      /(方案|文档|文章|邮件|公告|说明)(.*)(写|生成|起草|输出)/.test(t);
  }

  async function sendChat() {
    if (!activeThreadId) return;
    if (!canChat) return;
    const text = chatInput.trim();
    if (!text) return;
    setChatInput("");
    setError("");
    setStreaming(true);

    const writeIntent = isWriteIntent(text);
    if (writeIntent) {
      setShowCanvas(true);
    }

    // optimistic add user msg
    setMessages((m) => [
      ...m,
      { id: `tmp_${Date.now()}`, role: "user", content: text, created_at: new Date().toISOString(), user_id: me?.id || "" },
      { id: `tmp_a_${Date.now()}`, role: "assistant", content: "", created_at: new Date().toISOString(), user_id: me?.id || "" },
    ]);

    let acc = "";
    try {
      await api.chatStream(activeThreadId, text, (d) => {
        acc += d;
        setMessages((m) => {
          const copy = [...m];
          const last = copy[copy.length - 1];
          if (last && last.role === "assistant") {
            copy[copy.length - 1] = { ...last, content: acc };
          }
          return copy;
        });
      });
      if (writeIntent && acc.trim() && thread && (thread.permission === "owner" || thread.permission === "write")) {
        await api.updateThread(activeThreadId, { canvas_md: acc });
        setThread((prev: any) => (prev ? { ...prev, canvas_md: acc } : prev));
      }
      await refreshThreads();
    } catch (e: any) {
      setError(String(e?.message || e));
    } finally {
      setStreaming(false);
      // reload from server to get real ids
      await refreshChat();
    }
  }

  function onChatKeyDown(e: React.KeyboardEvent<HTMLTextAreaElement>) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      if (!streaming) sendChat();
    }
  }

  async function uploadImage(file: File) {
    setError("");
    try {
      const vision = await api.visionDescribe(file);
      const text = "用户上传了图片。识别结果(JSON)：\n" + JSON.stringify(vision, null, 2);
      setChatInput((prev) => (prev ? prev + "\n\n" + text : text));
    } catch (e: any) {
      setError(String(e?.message || e));
    }
  }

  async function openSettings() {
    setError("");
    try {
      const u = await api.me();
      setMe(u);
      setProfileName(u.name || "");
      setProfileText(u.profile || "");
      setSettingsOpen(true);
    } catch (e: any) {
      setError(String(e?.message || e));
    }
  }

  async function saveProfile() {
    setError("");
    try {
      await api.updateProfile(profileName, profileText);
      const u = await api.me();
      setMe(u);
      setSettingsOpen(false);
    } catch (e: any) {
      setError(String(e?.message || e));
    }
  }

  async function openAdmin() {
    setError("");
    try {
      const users = await api.adminListUsers();
      const cfg = await api.adminGetConfig();
      const aud = await api.adminListAudit({ limit: 200 }).catch(() => []);
      const reviews = await api.adminListTeachingReviews(teachingReviewStatus).catch(() => []);
      const ks = await api.knowledgeStats().catch(() => null);
      const ss = await api.listSources().catch(() => []);
      setAdminUsers(users);
      setAdminConfig(cfg);
      setAdminAudit(aud as any);
      setTeachingReviews(reviews as any);
      setCfgRagPolicy(String(cfg?.effective?.rag_policy || "auto"));
      setCfgRagTopK(String(cfg?.effective?.rag_top_k ?? "10"));
      setCfgRagMaxContext(String(cfg?.effective?.rag_max_context ?? "12"));
      setCfgRagTeachingScoreBoost(String(cfg?.effective?.rag_teaching_score_boost ?? "0.05"));
      setCfgRagTeachingCandidates(String(cfg?.effective?.rag_teaching_candidates ?? "40"));
      setCfgWebSearchEnabled(Boolean(cfg?.effective?.web_search_enabled ?? false));
      setCfgWebSearchTopK(String(cfg?.effective?.web_search_top_k ?? "5"));
      setCfgWebSearchMaxQueries(String(cfg?.effective?.web_search_max_queries ?? "2"));
      setCfgAgentDecomposePolicy(String(cfg?.effective?.agent_decompose_policy || "auto"));
      setCfgAgentDecomposeBias(String(cfg?.effective?.agent_decompose_bias ?? "30"));
      setCfgAgentMaxSubtasks(String(cfg?.effective?.agent_max_subtasks ?? "5"));
      setCfgAiTraceEnabled(Boolean(cfg?.effective?.ai_trace_enabled ?? true));
      setCfgAiTraceRetentionDays(String(cfg?.effective?.ai_trace_retention_days ?? "30"));
      setCfgMemoryEnabled(Boolean(cfg?.effective?.memory_enabled));
      setCfgMemoryTopK(String(cfg?.effective?.memory_top_k ?? "5"));
      setCfgTopicGuardEnabled(Boolean(cfg?.effective?.topic_guard_enabled));
      setCfgTopicAllowedTopics(String(cfg?.effective?.topic_allowed_topics || ""));
      setCfgPersonaDisclosureEnabled(Boolean(cfg?.effective?.persona_disclosure_enabled ?? false));
      setCfgThreadStateEnabled(Boolean(cfg?.effective?.thread_state_enabled ?? true));
      setCfgThreadStateWindowMsgs(String(cfg?.effective?.thread_state_window_msgs ?? "60"));
      setCfgThreadStateCooldownSeconds(String(cfg?.effective?.thread_state_cooldown_seconds ?? "90"));
      setCfgDecisionProfileEnabled(Boolean(cfg?.effective?.decision_profile_enabled ?? true));
      setCfgDecisionProfileRefreshHours(String(cfg?.effective?.decision_profile_refresh_hours ?? "24"));
      setCfgProactiveEnabled(Boolean(cfg?.effective?.proactive_enabled));
      setCfgProactiveMinMsgs(String(cfg?.effective?.proactive_min_user_msgs ?? "500"));
      setCfgProactiveWeekdayOnly(Boolean(cfg?.effective?.proactive_weekday_only ?? true));
      setCfgProactiveWorkStart(String(cfg?.effective?.proactive_work_start ?? "09:00"));
      setCfgProactiveWorkEnd(String(cfg?.effective?.proactive_work_end ?? "18:00"));
      setCfgProactiveTimezone(String(cfg?.effective?.proactive_timezone ?? "Asia/Shanghai"));
      setKstats(ks);
      setSources(ss as any);
      setAdminOpen(true);
    } catch (e: any) {
      setError(String(e?.message || e));
    }
  }

  async function refreshAudit() {
    setError("");
    try {
      const aud = await api.adminListAudit({ limit: 200 });
      setAdminAudit(aud);
    } catch (e: any) {
      setError(String(e?.message || e));
    }
  }

  async function refreshTeachingReviews(status?: "pending" | "approved" | "rejected" | "all") {
    setError("");
    try {
      const st = status || teachingReviewStatus;
      const rows = await api.adminListTeachingReviews(st);
      setTeachingReviews(rows);
    } catch (e: any) {
      setError(String(e?.message || e));
    }
  }

  async function approveTeachingReview(reviewId: string) {
    setError("");
    try {
      await api.adminApproveTeachingReview(reviewId);
      await refreshTeachingReviews();
      await refreshKnowledgeAdmin();
    } catch (e: any) {
      setError(String(e?.message || e));
    }
  }

  async function rejectTeachingReview(reviewId: string) {
    const comment = window.prompt("拒绝原因（可选）") || "";
    setError("");
    try {
      await api.adminRejectTeachingReview(reviewId, comment);
      await refreshTeachingReviews();
    } catch (e: any) {
      setError(String(e?.message || e));
    }
  }

  async function openReviewDetail(reviewId: string) {
    setError("");
    try {
      const d = await api.getTeachingSubmission(reviewId);
      setReviewDetail(d);
      setReviewDetailOpen(true);
    } catch (e: any) {
      setError(String(e?.message || e));
    }
  }

  async function setUserAdmin(userId: string, isAdmin: boolean) {
    await api.adminUpdateUser(userId, { is_admin: isAdmin });
    const users = await api.adminListUsers();
    setAdminUsers(users);
  }

  async function setUserAnyTopic(userId: string, allow: boolean) {
    await api.adminUpdateUser(userId, { allow_any_topic: allow });
    const users = await api.adminListUsers();
    setAdminUsers(users);
  }

  async function saveAdminConfig() {
    setError("");
    try {
      await api.adminSetConfig("rag_policy", cfgRagPolicy);
      await api.adminSetConfig("rag_top_k", cfgRagTopK);
      await api.adminSetConfig("rag_max_context", cfgRagMaxContext);
      await api.adminSetConfig("rag_teaching_score_boost", cfgRagTeachingScoreBoost);
      await api.adminSetConfig("rag_teaching_candidates", cfgRagTeachingCandidates);
      await api.adminSetConfig("web_search_enabled", cfgWebSearchEnabled ? "true" : "false");
      await api.adminSetConfig("web_search_top_k", cfgWebSearchTopK);
      await api.adminSetConfig("web_search_max_queries", cfgWebSearchMaxQueries);
      await api.adminSetConfig("agent_decompose_policy", cfgAgentDecomposePolicy);
      await api.adminSetConfig("agent_decompose_bias", cfgAgentDecomposeBias);
      await api.adminSetConfig("agent_max_subtasks", cfgAgentMaxSubtasks);
      await api.adminSetConfig("ai_trace_enabled", cfgAiTraceEnabled ? "true" : "false");
      await api.adminSetConfig("ai_trace_retention_days", cfgAiTraceRetentionDays);
      await api.adminSetConfig("memory_enabled", cfgMemoryEnabled ? "true" : "false");
      await api.adminSetConfig("memory_top_k", cfgMemoryTopK);
      await api.adminSetConfig("topic_guard_enabled", cfgTopicGuardEnabled ? "true" : "false");
      await api.adminSetConfig("topic_allowed_topics", cfgTopicAllowedTopics);
      await api.adminSetConfig("persona_disclosure_enabled", cfgPersonaDisclosureEnabled ? "true" : "false");
      await api.adminSetConfig("thread_state_enabled", cfgThreadStateEnabled ? "true" : "false");
      await api.adminSetConfig("thread_state_window_msgs", cfgThreadStateWindowMsgs);
      await api.adminSetConfig("thread_state_cooldown_seconds", cfgThreadStateCooldownSeconds);
      await api.adminSetConfig("decision_profile_enabled", cfgDecisionProfileEnabled ? "true" : "false");
      await api.adminSetConfig("decision_profile_refresh_hours", cfgDecisionProfileRefreshHours);
      await api.adminSetConfig("proactive_enabled", cfgProactiveEnabled ? "true" : "false");
      await api.adminSetConfig("proactive_min_user_msgs", cfgProactiveMinMsgs);
      await api.adminSetConfig("proactive_weekday_only", cfgProactiveWeekdayOnly ? "true" : "false");
      await api.adminSetConfig("proactive_work_start", cfgProactiveWorkStart);
      await api.adminSetConfig("proactive_work_end", cfgProactiveWorkEnd);
      await api.adminSetConfig("proactive_timezone", cfgProactiveTimezone);
      const cfg = await api.adminGetConfig();
      setAdminConfig(cfg);
    } catch (e: any) {
      setError(String(e?.message || e));
    }
  }

  async function refreshKnowledgeAdmin() {
    const ks = await api.knowledgeStats().catch(() => null);
    const ss = await api.listSources().catch(() => []);
    setKstats(ks);
    setSources(ss as any);
  }

  async function doPasteIngest() {
    setError("");
    setPasteResult("");
    try {
      const res = await api.ingestPaste(pasteText, pasteSourceName || "manual");
      setPasteResult(`ok: points=${res.points} upserted=${res.upserted}`);
      setPasteText("");
      await refreshKnowledgeAdmin();
    } catch (e: any) {
      setError(String(e?.message || e));
    }
  }

  function newSource(kind: "sql" | "odata") {
    setSourceResult("");
    setSrcEditingId("");
    setSrcKind(kind);
    setSrcName(kind === "sql" ? "SQL 数据源" : "OData 数据源");
    setSrcSqlDbUrl("");
    setSrcSqlQuery("SELECT id, title, content FROM your_table");
    setSrcOdataUrl("");
    setSrcOdataHeaders("{\n  \"Authorization\": \"Bearer <token>\"\n}");
  }

  async function editSource(sourceId: string) {
    setError("");
    setSourceResult("");
    try {
      const s: any = await api.getSource(sourceId);
      setSrcEditingId(String(s.id || sourceId));
      const k = String(s.kind || "sql") as any;
      setSrcKind(k === "odata" ? "odata" : "sql");
      setSrcName(String(s.name || ""));
      const cfg = (s.config && typeof s.config === "object") ? s.config : {};
      setSrcSqlDbUrl(String(cfg.database_url || ""));
      setSrcSqlQuery(String(cfg.query || "SELECT id, title, content FROM your_table"));
      setSrcOdataUrl(String(cfg.url || ""));
      setSrcOdataHeaders(JSON.stringify(cfg.headers || {}, null, 2) || "{\n  \"Authorization\": \"Bearer <token>\"\n}");
    } catch (e: any) {
      setError(String(e?.message || e));
    }
  }

  async function saveSource() {
    setError("");
    setSourceResult("");
    try {
      const name = (srcName || "").trim();
      if (!name) throw new Error("name is required");
      let config: any = {};
      if (srcKind === "sql") {
        config = { database_url: (srcSqlDbUrl || "").trim(), query: (srcSqlQuery || "").trim() };
        if (!config.database_url) throw new Error("database_url is required");
        if (!config.query) throw new Error("query is required");
      } else {
        config = { url: (srcOdataUrl || "").trim() };
        if (!config.url) throw new Error("url is required");
        const hs = (srcOdataHeaders || "").trim();
        if (hs) {
          try {
            const obj = JSON.parse(hs);
            if (obj && typeof obj === "object") config.headers = obj;
          } catch {
            throw new Error("headers 必须是合法 JSON");
          }
        }
      }

      if (srcEditingId) await api.updateSource(srcEditingId, srcKind as any, name, config);
      else await api.createSource(srcKind as any, name, config);
      setSourceResult("已保存");
      await refreshKnowledgeAdmin();
    } catch (e: any) {
      setError(String(e?.message || e));
    }
  }

  async function removeSource(sourceId: string) {
    setError("");
    setSourceResult("");
    try {
      await api.deleteSource(sourceId);
      setSourceResult("已删除");
      if (srcEditingId === sourceId) setSrcEditingId("");
      await refreshKnowledgeAdmin();
    } catch (e: any) {
      setError(String(e?.message || e));
    }
  }

  async function runSourceIngest(sourceId: string) {
    setError("");
    setSourceResult("");
    try {
      const n = parseInt(srcMaxItems || "200", 10);
      await api.ingestSource(sourceId, Number.isFinite(n) ? n : 200);
      setSourceResult("已触发后台导入（可在“知识库统计”里观察数量增长）");
    } catch (e: any) {
      setError(String(e?.message || e));
    }
  }

  if (view === "login") {
    return (
      <div className="login">
        <div className="left">
          <h1 style={{ marginTop: 0 }}>TeamMember</h1>
          <div style={{ opacity: 0.95, lineHeight: 1.7 }}>
            <div>一个可协作、可共享、带长期记忆的团队助手。</div>
            <ul>
              <li>对话驱动：你说怎么改，它就怎么改</li>
              <li>知识库按需介入：不是每次都检索</li>
              <li>每个账号独立隔离，可按线程共享</li>
            </ul>
            <div className="muted" style={{ color: "rgba(255,255,255,.85)" }}>
              提示：密码建议 8 位以上，避免超过 72 字节（bcrypt 限制）。
            </div>
          </div>
        </div>
        <div className="right">
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12 }}>
            <div>
              <h3 style={{ marginTop: 0 }}>登录</h3>
              <div className="field">
                <label>手机号或邮箱</label>
                <input value={loginAccount} onChange={(e) => setLoginAccount(e.target.value)} placeholder="13800138000 或 name@company.com" />
              </div>
              <div className="field">
                <label>密码</label>
                <input type="password" value={loginPassword} onChange={(e) => setLoginPassword(e.target.value)} />
              </div>
              <button className="btn primary" onClick={doLogin} style={{ width: "100%" }}>
                登录
              </button>
            </div>
            <div>
              <h3 style={{ marginTop: 0 }}>注册</h3>
              <div className="field">
                <label>姓名（可选）</label>
                <input value={regName} onChange={(e) => setRegName(e.target.value)} placeholder="张三" />
              </div>
              <div className="field">
                <label>手机号（中国 11 位或国际 +国家码）</label>
                <input value={regPhone} onChange={(e) => setRegPhone(e.target.value)} placeholder="13800138000 或 +8613800138000" />
              </div>
              <div className="field">
                <label>密码（至少 8 位）</label>
                <input type="password" value={regPassword} onChange={(e) => setRegPassword(e.target.value)} />
              </div>
              <button className="btn" onClick={doRegister} style={{ width: "100%" }}>
                注册并登录
              </button>
            </div>
          </div>
          {error ? <div className="error" style={{ marginTop: 10 }}>{error}</div> : null}
        </div>
      </div>
    );
  }

  return (
    <div>
      <div className="topbar">
        <div className="brand">TeamMember</div>
        <div className="row">
          {unreadCount ? <div className="badge" title="有新消息">新消息 {unreadCount}</div> : null}
          <div className="muted">{me ? (me.phone || me.email) : ""}</div>
          <button className="btn" onClick={openSettings}>
            设置
          </button>
          {me?.is_admin ? (
            <button className="btn" onClick={openAdmin}>
              Admin
            </button>
          ) : null}
          <button className="btn" onClick={() => { api.setToken(null); setMe(null); setView("login"); }}>
            退出
          </button>
        </div>
      </div>

      <div className={`layout ${showCanvas ? "withCanvas" : "noCanvas"}`}>
        <div className="panel">
          <h3>线程</h3>
          <div className="list">
            <button className="btn primary" style={{ width: "100%", marginBottom: 10 }} onClick={newThread}>
              新建线程
            </button>
            {threads.map((t) => (
              <div key={t.id} className={`item ${t.id === activeThreadId ? "active" : ""}`} onClick={() => openThread(t.id, t.updated_at)} style={{ position: "relative" }}>
                <div style={{ fontWeight: 600, marginBottom: 4 }}>{t.title || "未命名"}</div>
                <div className="muted">
                  {t.permission} · {new Date(t.updated_at).toLocaleString()}
                </div>
                {(() => {
                  const seen = lastSeen[t.id];
                  const unread = !seen || new Date(t.updated_at).getTime() > new Date(seen).getTime();
                  return unread ? <div className="unreadDot" title="有新消息" /> : null;
                })()}
              </div>
            ))}
          </div>
        </div>

        {showCanvas ? (
          <div className="panel">
            <h3>Canvas（仅在你明确要求写/生成时出现）</h3>
            <div className="canvas">
              {thread ? (
                <>
                  <div className="row" style={{ marginBottom: 10 }}>
                    <input
                      value={thread.title || ""}
                      onChange={(e) => setThread({ ...thread, title: e.target.value })}
                      style={{ flex: 1, padding: 10, borderRadius: 12, border: "1px solid var(--border)" }}
                      disabled={!canEdit}
                    />
                    <button className="btn primary" onClick={saveThread} disabled={!canEdit}>
                      保存
                    </button>
                  </div>
                  <textarea
                    value={thread.canvas_md || ""}
                    onChange={(e) => setThread({ ...thread, canvas_md: e.target.value })}
                    placeholder="这里是正文（Markdown）。当你在对话里明确要求“写/生成/起草”，AI 的输出会自动写入这里。"
                    disabled={!canEdit}
                  />
                  <div style={{ marginTop: 10 }}>
                    <div className="muted" style={{ marginBottom: 6 }}>
                      预览
                    </div>
                    <div className="msg">
                      <ReactMarkdown remarkPlugins={[remarkGfm]}>{thread.canvas_md || ""}</ReactMarkdown>
                    </div>
                  </div>
                </>
              ) : (
                <div className="muted">请选择或创建一个线程</div>
              )}
            </div>
          </div>
        ) : null}

        <div className="panel">
          <h3>对话</h3>
          {thread && thread.permission === "owner" ? (
            <div style={{ padding: "10px 12px", borderBottom: "1px solid var(--border)", background: "#fff" }}>
              <div className="muted" style={{ marginBottom: 8 }}>共享此线程（仅 owner 可管理）</div>
              <div className="row" style={{ marginBottom: 8 }}>
                <input
                  value={shareAccount}
                  onChange={(e) => setShareAccount(e.target.value)}
                  placeholder="同事手机号或邮箱"
                  style={{ flex: 1, padding: 10, borderRadius: 10, border: "1px solid var(--border)" }}
                />
                <select
                  value={sharePermission}
                  onChange={(e) => setSharePermission(e.target.value as "read" | "write")}
                  style={{ width: 110, padding: 10, borderRadius: 10, border: "1px solid var(--border)" }}
                >
                  <option value="read">read</option>
                  <option value="write">write</option>
                </select>
                <button className="btn" onClick={addShare}>共享</button>
              </div>
              <div style={{ display: "grid", gap: 6, maxHeight: 120, overflow: "auto" }}>
                {shareRows.map((row) => (
                  <div key={row.user_id} className="msg" style={{ margin: 0, padding: "8px 10px", display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                    <div>
                      <div style={{ fontWeight: 600 }}>{row.name || row.phone || row.email}</div>
                      <div className="muted">{(row.phone || row.email)} · {row.permission}</div>
                    </div>
                    <button className="btn" onClick={() => revokeShare(row.user_id)}>移除</button>
                  </div>
                ))}
                {!shareRows.length ? <div className="muted">暂无共享对象</div> : null}
              </div>
            </div>
          ) : null}
          <div className="chat">
            {messages.map((m) => (
                <div key={m.id} className="msg">
                  <div className="meta">
                    <div>{m.role === "user" ? "你" : m.role === "assistant" ? "AI" : m.role}</div>
                    <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
                      <div>{new Date(m.created_at).toLocaleString()}</div>
                      {m.role === "assistant" && !String(m.id || "").startsWith("tmp_") ? (
                        <button className="btn" onClick={() => openTrace(m.id)} disabled={traceLoading || streaming}>
                          工具/Trace
                        </button>
                      ) : null}
                    </div>
                  </div>
                  {m.role === "assistant" ? (
                    <ReactMarkdown remarkPlugins={[remarkGfm]}>{m.content || ""}</ReactMarkdown>
                  ) : (
                    <div style={{ whiteSpace: "pre-wrap" }}>{m.content}</div>
                )}
              </div>
            ))}
            <div ref={chatEndRef} />
          </div>
          <div className="inputbar">
            {error ? <div className="error" style={{ marginBottom: 8 }}>{error}</div> : null}
            <textarea
              value={chatInput}
              onChange={(e) => setChatInput(e.target.value)}
              onKeyDown={onChatKeyDown}
              placeholder="Enter 发送，Shift+Enter 换行。可直接粘贴文字，也可上传截图。"
              disabled={!activeThreadId || streaming || !canChat}
            />
            <div className="row" style={{ marginTop: 8, justifyContent: "space-between" }}>
              <div className="row">
                <label className="btn" style={{ display: "inline-flex", alignItems: "center", gap: 6 }}>
                  上传图片
                  <input
                    type="file"
                    accept="image/*"
                    style={{ display: "none" }}
                    onChange={(e) => {
                      const f = e.target.files?.[0];
                      if (f) uploadImage(f);
                      e.currentTarget.value = "";
                    }}
                  />
                </label>
                <button className="btn" onClick={refreshChat} disabled={!activeThreadId || streaming}>
                  刷新
                </button>
                <button className="btn" onClick={openTeaching} disabled={!activeThreadId || streaming || !canChat}>
                  写入知识
                </button>
              </div>
              <button className="btn primary" onClick={sendChat} disabled={!activeThreadId || streaming || !canChat}>
                {streaming ? "生成中…" : "发送"}
              </button>
            </div>
            {!canChat && activeThreadId ? <div className="muted" style={{ marginTop: 8 }}>当前线程是只读共享权限，不能发送新消息。</div> : null}
          </div>
        </div>
      </div>

      {traceOpen ? (
        <div className="modalBackdrop" onClick={() => setTraceOpen(false)}>
          <div className="modalCard" onClick={(e) => e.stopPropagation()}>
            <div className="modalHeader">
              <div style={{ fontWeight: 700 }}>工具 / AI Trace（30 天自动清理）</div>
              <button className="btn" onClick={() => setTraceOpen(false)}>
                关闭
              </button>
            </div>
            <div className="modalBody">
              {traceData ? (
                <>
                  <div className="muted" style={{ marginBottom: 10 }}>
                    trace_id={traceData.id} · {new Date(traceData.created_at).toLocaleString()}
                  </div>
                  {traceData.error ? <div className="error" style={{ marginBottom: 10 }}>{traceData.error}</div> : null}

                  <div style={{ fontWeight: 700, marginBottom: 8 }}>路由决策</div>
                  <pre className="msg" style={{ margin: 0, whiteSpace: "pre-wrap" }}>{JSON.stringify(traceData.router || {}, null, 2)}</pre>

                  <div className="hr" />
                  <div style={{ fontWeight: 700, marginBottom: 8 }}>Web Search（摘要）</div>
                  {(() => {
                    const ws: any = (traceData as any).web_search || {};
                    const results: any = ws.results || {};
                    const qs: string[] = Array.isArray(ws.queries) ? ws.queries : Object.keys(results || {});
                    if (!qs.length) return <div className="muted">本次未使用 Web Search</div>;
                    return (
                      <div style={{ display: "grid", gap: 10 }}>
                        {qs.map((q) => {
                          const rs: any[] = Array.isArray(results?.[q]) ? results[q] : [];
                          return (
                            <div key={q} className="msg" style={{ margin: 0 }}>
                              <div style={{ fontWeight: 700, marginBottom: 6 }}>{q}</div>
                              {!rs.length ? <div className="muted">无结果</div> : null}
                              <div style={{ display: "grid", gap: 6 }}>
                                {rs.slice(0, 8).map((r, idx) => (
                                  <div key={idx} style={{ padding: "6px 0", borderBottom: "1px solid var(--border)" }}>
                                    <div style={{ fontWeight: 600 }}>{r.title || "(no title)"}</div>
                                    {r.url ? (
                                      <div className="muted" style={{ marginTop: 2 }}>
                                        <a href={String(r.url)} target="_blank" rel="noreferrer">
                                          {String(r.url)}
                                        </a>
                                      </div>
                                    ) : null}
                                    {r.snippet ? <div style={{ marginTop: 6, whiteSpace: "pre-wrap" }}>{String(r.snippet)}</div> : null}
                                  </div>
                                ))}
                              </div>
                            </div>
                          );
                        })}
                      </div>
                    );
                  })()}

                  <div className="hr" />
                  <div style={{ fontWeight: 700, marginBottom: 8 }}>子任务拆分</div>
                  <pre className="msg" style={{ margin: 0, whiteSpace: "pre-wrap" }}>{JSON.stringify(traceData.decompose || {}, null, 2)}</pre>

                  <div className="hr" />
                  <div style={{ fontWeight: 700, marginBottom: 8 }}>子Agent 结果</div>
                  <pre className="msg" style={{ margin: 0, whiteSpace: "pre-wrap" }}>{JSON.stringify(traceData.subagent || [], null, 2)}</pre>
                </>
              ) : (
                <div className="muted">暂无 trace 数据</div>
              )}
            </div>
          </div>
        </div>
      ) : null}

      {settingsOpen ? (
        <div className="modalBackdrop" onClick={() => setSettingsOpen(false)}>
          <div className="modalCard" onClick={(e) => e.stopPropagation()}>
            <div className="modalHeader">
              <div style={{ fontWeight: 700 }}>设置</div>
              <button className="btn" onClick={() => setSettingsOpen(false)}>
                关闭
              </button>
            </div>
            <div className="modalBody">
              <div className="grid2">
                <div>
                  <div className="field">
                    <label>姓名（可选）</label>
                    <input value={profileName} onChange={(e) => setProfileName(e.target.value)} />
                  </div>
                  <div className="field">
                    <label>个性化说明（你是谁、角色、擅长什么、对 AI 的要求）</label>
                    <textarea value={profileText} onChange={(e) => setProfileText(e.target.value)} rows={10} />
                  </div>
                  <button className="btn primary" onClick={saveProfile}>
                    保存
                  </button>
                </div>
                <div>
                  <div className="muted" style={{ marginBottom: 8 }}>
                    说明：这段文字会用于生成你的 Persona 与长期记忆，让回答更贴合你。
                  </div>
                  <div className="muted">Admin 设置在右上角的 Admin 按钮（仅管理员可见）。</div>
                </div>
              </div>
            </div>
          </div>
        </div>
      ) : null}

      {teachingOpen ? (
        <div className="modalBackdrop" onClick={() => setTeachingOpen(false)}>
          <div className="modalCard" onClick={(e) => e.stopPropagation()}>
            <div className="modalHeader">
              <div style={{ fontWeight: 700 }}>写入知识（需审核）</div>
              <button className="btn" onClick={() => setTeachingOpen(false)}>
                关闭
              </button>
            </div>
            <div className="modalBody">
              <div className="muted" style={{ marginBottom: 10 }}>
                用于“纠错/沉淀”：系统会基于当前线程的 Canvas 与最近对话生成 Teaching Note，再按语义拆成知识点提交管理员审核。
              </div>
              <div className="field">
                <label>补充说明（可选）</label>
                <textarea
                  rows={6}
                  value={teachingInstruction}
                  onChange={(e) => setTeachingInstruction(e.target.value)}
                  placeholder="例如：重点沉淀 Exchange Online 21V 的域名/端点；把错误理解纠正为…"
                />
              </div>
              <div className="row" style={{ justifyContent: "space-between" }}>
                <button className="btn" onClick={() => { setTeachingInstruction(""); setTeachingResult(""); }} disabled={teachingSubmitting}>
                  清空
                </button>
                <button className="btn primary" onClick={submitTeaching} disabled={!activeThreadId || teachingSubmitting}>
                  {teachingSubmitting ? "提交中…" : "提交审核"}
                </button>
              </div>
              {teachingResult ? (
                <div className="msg" style={{ marginTop: 12 }}>
                  <div style={{ whiteSpace: "pre-wrap" }}>{teachingResult}</div>
                </div>
              ) : null}
              {error ? <div className="error" style={{ marginTop: 10 }}>{error}</div> : null}
            </div>
          </div>
        </div>
      ) : null}

      {adminOpen ? (
        <div className="modalBackdrop" onClick={() => setAdminOpen(false)}>
          <div className="modalCard" onClick={(e) => e.stopPropagation()}>
            <div className="modalHeader">
              <div style={{ fontWeight: 700 }}>Admin</div>
              <button className="btn" onClick={() => setAdminOpen(false)}>
                关闭
              </button>
            </div>
            <div className="modalBody">
              {error ? <div className="error" style={{ marginBottom: 10 }}>{error}</div> : null}
              <div style={{ fontWeight: 700, marginBottom: 8 }}>用户</div>
              <div className="muted" style={{ marginBottom: 10 }}>
                你可以指定哪些用户是管理员。管理员可看到此面板。
              </div>
              <div style={{ display: "grid", gap: 8 }}>
                {adminUsers.map((u) => (
                  <div key={u.id} className="msg" style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                    <div>
                      <div style={{ fontWeight: 700 }}>{u.phone || u.email}</div>
                      <div className="muted">{u.name || "未填写姓名"} · {new Date(u.created_at).toLocaleString()}</div>
                    </div>
                    <div className="row" style={{ gap: 14 }}>
                      <label className="muted" style={{ display: "flex", gap: 8, alignItems: "center" }}>
                        <input type="checkbox" checked={u.is_admin} onChange={(e) => setUserAdmin(u.id, e.target.checked)} />
                        管理员
                      </label>
                      <label className="muted" style={{ display: "flex", gap: 8, alignItems: "center" }}>
                        <input
                          type="checkbox"
                          checked={Boolean(u.allow_any_topic)}
                          onChange={(e) => setUserAnyTopic(u.id, e.target.checked)}
                        />
                        话题例外
                      </label>
                    </div>
                  </div>
                ))}
              </div>

              <div className="hr" />
              <div style={{ fontWeight: 700, marginBottom: 8 }}>全局参数（运行时覆盖）</div>
              <div className="muted" style={{ marginBottom: 10 }}>
                这里的修改会写入数据库并即时生效（不需要改 .env）。
              </div>

              <div className="grid2">
                <div className="field">
                  <label>RAG 策略</label>
                  <select
                    value={cfgRagPolicy}
                    onChange={(e) => setCfgRagPolicy(e.target.value)}
                    style={{ width: "100%", padding: 10, borderRadius: 12, border: "1px solid var(--border)" }}
                  >
                    <option value="auto">auto（自动判断是否检索）</option>
                    <option value="force_on">force_on（强制检索）</option>
                    <option value="force_off">force_off（强制不检索）</option>
                  </select>
                </div>
                <div className="field">
                  <label>RAG TopK</label>
                  <input value={cfgRagTopK} onChange={(e) => setCfgRagTopK(e.target.value)} />
                </div>
                <div className="field">
                  <label>RAG Max Context</label>
                  <input value={cfgRagMaxContext} onChange={(e) => setCfgRagMaxContext(e.target.value)} />
                </div>
                <div className="field">
                  <label>Teaching Score Boost</label>
                  <input value={cfgRagTeachingScoreBoost} onChange={(e) => setCfgRagTeachingScoreBoost(e.target.value)} />
                </div>
                <div className="field">
                  <label>Teaching Candidates</label>
                  <input value={cfgRagTeachingCandidates} onChange={(e) => setCfgRagTeachingCandidates(e.target.value)} />
                </div>
                <div className="field">
                  <label>Web Search</label>
                  <div className="row">
                    <label className="muted" style={{ display: "flex", gap: 8, alignItems: "center" }}>
                      <input type="checkbox" checked={cfgWebSearchEnabled} onChange={(e) => setCfgWebSearchEnabled(e.target.checked)} />
                      允许按需 Web Search（Bing 摘要）
                    </label>
                  </div>
                </div>
                <div className="field">
                  <label>Web TopK</label>
                  <input value={cfgWebSearchTopK} onChange={(e) => setCfgWebSearchTopK(e.target.value)} />
                </div>
                <div className="field">
                  <label>Web Max Queries</label>
                  <input value={cfgWebSearchMaxQueries} onChange={(e) => setCfgWebSearchMaxQueries(e.target.value)} />
                </div>
                <div className="field">
                  <label>子Agent 策略</label>
                  <select
                    value={cfgAgentDecomposePolicy}
                    onChange={(e) => setCfgAgentDecomposePolicy(e.target.value)}
                    style={{ width: "100%", padding: 10, borderRadius: 12, border: "1px solid var(--border)" }}
                  >
                    <option value="auto">auto（自动拆分）</option>
                    <option value="force_on">force_on（强制拆分）</option>
                    <option value="force_off">force_off（强制不拆分）</option>
                  </select>
                </div>
                <div className="field">
                  <label>子Agent 偏好（0..100）</label>
                  <input value={cfgAgentDecomposeBias} onChange={(e) => setCfgAgentDecomposeBias(e.target.value)} />
                </div>
                <div className="field">
                  <label>子任务上限</label>
                  <input value={cfgAgentMaxSubtasks} onChange={(e) => setCfgAgentMaxSubtasks(e.target.value)} />
                </div>
                <div className="field">
                  <label>AI Trace</label>
                  <div className="row">
                    <label className="muted" style={{ display: "flex", gap: 8, alignItems: "center" }}>
                      <input type="checkbox" checked={cfgAiTraceEnabled} onChange={(e) => setCfgAiTraceEnabled(e.target.checked)} />
                      记录路由/工具/子任务（保留期自动清理）
                    </label>
                  </div>
                </div>
                <div className="field">
                  <label>Trace 保留天数</label>
                  <input value={cfgAiTraceRetentionDays} onChange={(e) => setCfgAiTraceRetentionDays(e.target.value)} />
                </div>
                <div className="field">
                  <label>Memory Enabled</label>
                  <div className="row">
                    <label className="muted" style={{ display: "flex", gap: 8, alignItems: "center" }}>
                      <input type="checkbox" checked={cfgMemoryEnabled} onChange={(e) => setCfgMemoryEnabled(e.target.checked)} />
                      启用用户长期记忆
                    </label>
                  </div>
                </div>
                <div className="field">
                  <label>Memory TopK</label>
                  <input value={cfgMemoryTopK} onChange={(e) => setCfgMemoryTopK(e.target.value)} />
                </div>
                <div className="field">
                  <label>话题范围限制</label>
                  <div className="row">
                    <label className="muted" style={{ display: "flex", gap: 8, alignItems: "center" }}>
                      <input type="checkbox" checked={cfgTopicGuardEnabled} onChange={(e) => setCfgTopicGuardEnabled(e.target.checked)} />
                      启用“允许话题”限制
                    </label>
                  </div>
                </div>
                <div className="field" style={{ gridColumn: "1 / -1" }}>
                  <label>允许的话题（描述/关键词，越具体越好）</label>
                  <textarea
                    rows={4}
                    value={cfgTopicAllowedTopics}
                    onChange={(e) => setCfgTopicAllowedTopics(e.target.value)}
                    placeholder="例如：21V Microsoft 365（世纪互联）、Entra ID、Exchange Online、Teams、SharePoint、OneDrive、Purview、Defender、Power Platform、Dynamics 365"
                  />
                </div>
                <div className="field">
                  <label>画像可披露</label>
                  <div className="row">
                    <label className="muted" style={{ display: "flex", gap: 8, alignItems: "center" }}>
                      <input
                        type="checkbox"
                        checked={cfgPersonaDisclosureEnabled}
                        onChange={(e) => setCfgPersonaDisclosureEnabled(e.target.checked)}
                      />
                      允许线程内互相调侃 Persona/决策画像
                    </label>
                  </div>
                </div>
                <div className="field">
                  <label>Thread State</label>
                  <div className="row">
                    <label className="muted" style={{ display: "flex", gap: 8, alignItems: "center" }}>
                      <input
                        type="checkbox"
                        checked={cfgThreadStateEnabled}
                        onChange={(e) => setCfgThreadStateEnabled(e.target.checked)}
                      />
                      启用“未闭合事项/主题熵”
                    </label>
                  </div>
                </div>
                <div className="field">
                  <label>State Window Msgs</label>
                  <input value={cfgThreadStateWindowMsgs} onChange={(e) => setCfgThreadStateWindowMsgs(e.target.value)} />
                </div>
                <div className="field">
                  <label>State Cooldown (s)</label>
                  <input value={cfgThreadStateCooldownSeconds} onChange={(e) => setCfgThreadStateCooldownSeconds(e.target.value)} />
                </div>
                <div className="field">
                  <label>Decision Profile</label>
                  <div className="row">
                    <label className="muted" style={{ display: "flex", gap: 8, alignItems: "center" }}>
                      <input
                        type="checkbox"
                        checked={cfgDecisionProfileEnabled}
                        onChange={(e) => setCfgDecisionProfileEnabled(e.target.checked)}
                      />
                      启用“决策画像”
                    </label>
                  </div>
                </div>
                <div className="field">
                  <label>Decision Refresh (hours)</label>
                  <input value={cfgDecisionProfileRefreshHours} onChange={(e) => setCfgDecisionProfileRefreshHours(e.target.value)} />
                </div>
                <div className="field">
                  <label>Proactive Nudge</label>
                  <div className="row">
                    <label className="muted" style={{ display: "flex", gap: 8, alignItems: "center" }}>
                      <input
                        type="checkbox"
                        checked={cfgProactiveEnabled}
                        onChange={(e) => setCfgProactiveEnabled(e.target.checked)}
                      />
                      启用“工作日随手一句”
                    </label>
                  </div>
                </div>
                <div className="field">
                  <label>触发门槛（用户消息数）</label>
                  <input value={cfgProactiveMinMsgs} onChange={(e) => setCfgProactiveMinMsgs(e.target.value)} />
                </div>
                <div className="field">
                  <label>仅工作日</label>
                  <div className="row">
                    <label className="muted" style={{ display: "flex", gap: 8, alignItems: "center" }}>
                      <input
                        type="checkbox"
                        checked={cfgProactiveWeekdayOnly}
                        onChange={(e) => setCfgProactiveWeekdayOnly(e.target.checked)}
                      />
                      周一到周五
                    </label>
                  </div>
                </div>
                <div className="field">
                  <label>工作时间（HH:MM）</label>
                  <div className="row" style={{ gap: 10 }}>
                    <input style={{ width: "100%" }} value={cfgProactiveWorkStart} onChange={(e) => setCfgProactiveWorkStart(e.target.value)} />
                    <input style={{ width: "100%" }} value={cfgProactiveWorkEnd} onChange={(e) => setCfgProactiveWorkEnd(e.target.value)} />
                  </div>
                </div>
                <div className="field">
                  <label>时区</label>
                  <input value={cfgProactiveTimezone} onChange={(e) => setCfgProactiveTimezone(e.target.value)} placeholder="Asia/Shanghai" />
                </div>
                <div className="field">
                  <label>每天最多触发</label>
                  <input value="1（固定）" disabled />
                </div>
                <div className="field" style={{ display: "flex", alignItems: "flex-end" }}>
                  <button className="btn primary" onClick={saveAdminConfig}>
                    保存全局参数
                  </button>
                </div>
              </div>

              <div className="msg">
                <div className="muted">effective</div>
                <pre style={{ margin: 0, whiteSpace: "pre-wrap" }}>{JSON.stringify(adminConfig?.effective || {}, null, 2)}</pre>
              </div>
              <div className="msg">
                <div className="muted">overrides</div>
                <pre style={{ margin: 0, whiteSpace: "pre-wrap" }}>{JSON.stringify(adminConfig?.overrides || {}, null, 2)}</pre>
              </div>

              <div className="hr" />
              <div className="row" style={{ justifyContent: "space-between", alignItems: "center", marginBottom: 8 }}>
                <div style={{ fontWeight: 700 }}>审计日志（只增不改）</div>
                <button className="btn" onClick={refreshAudit}>刷新</button>
              </div>
              <div className="muted" style={{ marginBottom: 10 }}>
                记录管理员对系统做的变更操作。应用层不提供修改接口，数据库也禁止 UPDATE/DELETE（append-only）。
              </div>
              <div style={{ display: "grid", gap: 8, maxHeight: 280, overflow: "auto" }}>
                {adminAudit.map((a) => (
                  <details key={a.id} className="msg" style={{ margin: 0 }}>
                    <summary style={{ cursor: "pointer" }}>
                      <span style={{ fontWeight: 700 }}>{a.action}</span>
                      <span className="muted"> · {a.actor_label} · {new Date(a.created_at).toLocaleString()}</span>
                      <span className="muted"> · {a.entity_type}:{a.entity_id}</span>
                    </summary>
                    <div className="muted" style={{ marginTop: 8 }}>
                      {a.request_method} {a.request_path} · ip={a.request_ip || "-"} 
                    </div>
                    <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 10, marginTop: 8 }}>
                      <div>
                        <div className="muted">before</div>
                        <pre style={{ margin: 0, whiteSpace: "pre-wrap" }}>{a.before_json || "{}"}</pre>
                      </div>
                      <div>
                        <div className="muted">after</div>
                        <pre style={{ margin: 0, whiteSpace: "pre-wrap" }}>{a.after_json || "{}"}</pre>
                      </div>
                    </div>
                    <div className="muted" style={{ marginTop: 8 }}>
                      hash={a.event_hash} prev={a.prev_hash || "(genesis)"}
                    </div>
                  </details>
                ))}
                {!adminAudit.length ? <div className="muted">暂无记录</div> : null}
              </div>

              <div className="hr" />
              <div style={{ fontWeight: 700, marginBottom: 8 }}>知识库导入（手工 / SQL / OData）</div>
              <div className="muted" style={{ marginBottom: 10 }}>
                说明：导入会先用 LLM 按语义抽取“知识点”再写入向量库；不会按原始条目硬切。
              </div>

              <div className="msg">
                <div className="row" style={{ justifyContent: "space-between" }}>
                  <div style={{ fontWeight: 700 }}>Teaching 审核（写入知识）</div>
                  <div className="row">
                    <select
                      value={teachingReviewStatus}
                      onChange={(e) => {
                        const v = e.target.value as any;
                        setTeachingReviewStatus(v);
                        refreshTeachingReviews(v);
                      }}
                      style={{ padding: 8, borderRadius: 10, border: "1px solid var(--border)" }}
                    >
                      <option value="pending">pending</option>
                      <option value="approved">approved</option>
                      <option value="rejected">rejected</option>
                      <option value="all">all</option>
                    </select>
                    <button className="btn" onClick={() => refreshTeachingReviews()}>
                      刷新
                    </button>
                  </div>
                </div>
                <div style={{ display: "grid", gap: 8, marginTop: 10, maxHeight: 260, overflow: "auto" }}>
                  {teachingReviews.map((r) => (
                    <div key={r.id} className="msg" style={{ margin: 0 }}>
                      <div className="row" style={{ justifyContent: "space-between", alignItems: "flex-start" }}>
                        <div style={{ minWidth: 0 }}>
                          <div style={{ fontWeight: 700, marginBottom: 4 }}>{r.title || "未命名"}</div>
                          <div className="muted">
                            {r.status} · points={r.points} · {new Date(r.created_at).toLocaleString()}
                          </div>
                          <div className="muted" style={{ marginTop: 4 }}>
                            提交人：{r.submitter_name || r.submitter_phone || r.submitter_email || r.submitter_user_id}
                          </div>
                          {r.admin_comment ? <div className="muted" style={{ marginTop: 4 }}>备注：{r.admin_comment}</div> : null}
                        </div>
                        <div className="row" style={{ gap: 8 }}>
                          <button className="btn" onClick={() => openReviewDetail(r.id)}>查看</button>
                          {r.status === "pending" ? (
                            <>
                              <button className="btn" onClick={() => rejectTeachingReview(r.id)}>拒绝</button>
                              <button className="btn primary" onClick={() => approveTeachingReview(r.id)}>通过并入库</button>
                            </>
                          ) : null}
                        </div>
                      </div>
                    </div>
                  ))}
                  {!teachingReviews.length ? <div className="muted">暂无记录</div> : null}
                </div>
              </div>

              <div className="msg">
                <div className="row" style={{ justifyContent: "space-between" }}>
                  <div style={{ fontWeight: 700 }}>知识库统计</div>
                  <button className="btn" onClick={refreshKnowledgeAdmin}>
                    刷新
                  </button>
                </div>
                <pre style={{ margin: 0, whiteSpace: "pre-wrap" }}>{JSON.stringify(kstats || {}, null, 2)}</pre>
              </div>

              <div className="grid2">
                <div className="msg">
                  <div style={{ fontWeight: 700, marginBottom: 8 }}>手工粘贴导入</div>
                  <div className="field">
                    <label>Source Name（用于标识来源）</label>
                    <input value={pasteSourceName} onChange={(e) => setPasteSourceName(e.target.value)} placeholder="manual" />
                  </div>
                  <div className="field">
                    <label>内容</label>
                    <textarea rows={10} value={pasteText} onChange={(e) => setPasteText(e.target.value)} placeholder="粘贴任意内容，系统会自动按语义抽取知识点并入库。" />
                  </div>
                  <button className="btn primary" onClick={doPasteIngest} disabled={!pasteText.trim()}>
                    解析并入库
                  </button>
                  {pasteResult ? <div className="muted" style={{ marginTop: 8 }}>{pasteResult}</div> : null}
                </div>

                <div className="msg">
                  <div className="row" style={{ justifyContent: "space-between", marginBottom: 8 }}>
                    <div style={{ fontWeight: 700 }}>数据源（SQL / OData）</div>
                    <div className="row">
                      <button className="btn" onClick={() => newSource("sql")}>新建 SQL</button>
                      <button className="btn" onClick={() => newSource("odata")}>新建 OData</button>
                    </div>
                  </div>

                  <div className="field">
                    <label>类型</label>
                    <select value={srcKind} onChange={(e) => setSrcKind(e.target.value as any)} style={{ width: "100%", padding: 10, borderRadius: 12, border: "1px solid var(--border)" }}>
                      <option value="sql">sql</option>
                      <option value="odata">odata</option>
                    </select>
                  </div>
                  <div className="field">
                    <label>名称</label>
                    <input value={srcName} onChange={(e) => setSrcName(e.target.value)} placeholder="数据源名称" />
                  </div>

                  {srcKind === "sql" ? (
                    <>
                      <div className="field">
                        <label>database_url</label>
                        <input value={srcSqlDbUrl} onChange={(e) => setSrcSqlDbUrl(e.target.value)} placeholder="postgresql+psycopg://user:pass@host:5432/db" />
                      </div>
                      <div className="field">
                        <label>query（必须返回 id,title,content 三列）</label>
                        <textarea rows={6} value={srcSqlQuery} onChange={(e) => setSrcSqlQuery(e.target.value)} />
                      </div>
                    </>
                  ) : (
                    <>
                      <div className="field">
                        <label>url（支持 @odata.nextLink 自动翻页）</label>
                        <input value={srcOdataUrl} onChange={(e) => setSrcOdataUrl(e.target.value)} placeholder="https://example.com/api/data/v9.0/..." />
                      </div>
                      <div className="field">
                        <label>headers（可选，JSON）</label>
                        <textarea rows={6} value={srcOdataHeaders} onChange={(e) => setSrcOdataHeaders(e.target.value)} />
                      </div>
                    </>
                  )}

                  <div className="field">
                    <label>导入 max_items</label>
                    <input value={srcMaxItems} onChange={(e) => setSrcMaxItems(e.target.value)} />
                  </div>

                  <div className="row">
                    <button className="btn primary" onClick={saveSource}>保存数据源</button>
                    {srcEditingId ? (
                      <button className="btn" onClick={() => runSourceIngest(srcEditingId)}>
                        触发导入
                      </button>
                    ) : null}
                  </div>
                  {sourceResult ? <div className="muted" style={{ marginTop: 8 }}>{sourceResult}</div> : null}

                  <div className="hr" />
                  <div style={{ fontWeight: 700, marginBottom: 8 }}>已保存的数据源</div>
                  <div style={{ display: "grid", gap: 8, maxHeight: 220, overflow: "auto" }}>
                    {sources.map((s) => (
                      <div key={s.id} className="msg" style={{ margin: 0 }}>
                        <div className="row" style={{ justifyContent: "space-between" }}>
                          <div>
                            <div style={{ fontWeight: 700 }}>{s.name}</div>
                            <div className="muted">{s.kind} · {new Date(s.updated_at).toLocaleString()}</div>
                          </div>
                          <div className="row">
                            <button className="btn" onClick={() => editSource(s.id)}>编辑</button>
                            <button className="btn" onClick={() => runSourceIngest(s.id)}>导入</button>
                            <button className="btn" onClick={() => removeSource(s.id)}>删除</button>
                          </div>
                        </div>
                      </div>
                    ))}
                    {!sources.length ? <div className="muted">暂无数据源</div> : null}
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      ) : null}

      {reviewDetailOpen ? (
        <div className="modalBackdrop" onClick={() => setReviewDetailOpen(false)}>
          <div className="modalCard" onClick={(e) => e.stopPropagation()}>
            <div className="modalHeader">
              <div style={{ fontWeight: 700 }}>Teaching 详情</div>
              <button className="btn" onClick={() => setReviewDetailOpen(false)}>
                关闭
              </button>
            </div>
            <div className="modalBody">
              {!reviewDetail ? (
                <div className="muted">加载中…</div>
              ) : (
                <>
                  <div className="muted" style={{ marginBottom: 10 }}>
                    {reviewDetail.status} · points={(reviewDetail.points || []).length} · {new Date(reviewDetail.created_at).toLocaleString()}
                  </div>
                  <div className="msg">
                    <div style={{ fontWeight: 700, marginBottom: 8 }}>{reviewDetail.title || "未命名"}</div>
                    <div className="muted" style={{ marginBottom: 6 }}>Teaching Note</div>
                    <pre style={{ margin: 0, whiteSpace: "pre-wrap" }}>{reviewDetail.teaching_note || ""}</pre>
                  </div>
                  <div className="msg">
                    <div style={{ fontWeight: 700, marginBottom: 8 }}>知识点</div>
                    <div style={{ display: "grid", gap: 8 }}>
                      {(reviewDetail.points || []).slice(0, 30).map((p: any, idx: number) => (
                        <details key={idx} style={{ border: "1px solid var(--border)", borderRadius: 10, padding: 10, background: "#fff" }}>
                          <summary style={{ cursor: "pointer", fontWeight: 700 }}>{p.title || `#${idx + 1}`}</summary>
                          <div className="muted" style={{ marginTop: 6 }}>{(p.tags || []).join(", ")}</div>
                          <div style={{ whiteSpace: "pre-wrap", marginTop: 8 }}>{p.content || ""}</div>
                        </details>
                      ))}
                      {(reviewDetail.points || []).length > 30 ? <div className="muted">仅展示前 30 条</div> : null}
                    </div>
                  </div>
                  {reviewDetail.admin_comment ? <div className="msg"><div className="muted">Admin 备注</div><div style={{ whiteSpace: "pre-wrap" }}>{reviewDetail.admin_comment}</div></div> : null}

                  {me?.is_admin && reviewDetail.status === "pending" ? (
                    <div className="row" style={{ justifyContent: "space-between" }}>
                      <button
                        className="btn"
                        onClick={async () => {
                          await rejectTeachingReview(String(reviewDetail.id || ""));
                          setReviewDetailOpen(false);
                        }}
                      >
                        拒绝
                      </button>
                      <button
                        className="btn primary"
                        onClick={async () => {
                          await approveTeachingReview(String(reviewDetail.id || ""));
                          setReviewDetailOpen(false);
                        }}
                      >
                        通过并入库
                      </button>
                    </div>
                  ) : null}
                </>
              )}
            </div>
          </div>
        </div>
      ) : null}
    </div>
  );
}
