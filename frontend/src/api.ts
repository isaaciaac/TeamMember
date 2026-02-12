export type User = { id: string; phone: string; email: string; name: string; is_admin: boolean; allow_any_topic?: boolean; profile?: string };

let _token: string | null = localStorage.getItem("tm_token");
const _apiBase = (import.meta as any).env?.VITE_API_BASE ? String((import.meta as any).env.VITE_API_BASE) : "";

export function setToken(t: string | null) {
  _token = t;
  if (t) localStorage.setItem("tm_token", t);
  else localStorage.removeItem("tm_token");
}

async function apiFetch(path: string, init: RequestInit = {}) {
  const headers: Record<string, string> = { ...(init.headers as any) };
  if (_token) headers["Authorization"] = `Bearer ${_token}`;
  if (!(init.body instanceof FormData) && !headers["Content-Type"] && init.method && init.method !== "GET") {
    headers["Content-Type"] = "application/json";
  }
  const base = _apiBase ? _apiBase.replace(/\/$/, "") : "";
  const resp = await fetch(`${base}/api${path}`, { ...init, headers });
  const ct = resp.headers.get("content-type") || "";
  const isJson = ct.includes("application/json");
  const data = isJson ? await resp.json().catch(() => null) : await resp.text().catch(() => "");
  if (!resp.ok) {
    let msg = (isJson && data && (data.detail || data.error)) || `${resp.status} ${resp.statusText}`;
    if (!isJson && typeof data === "string") {
      const t = data.trim().toLowerCase();
      if (t.startsWith("<!doctype html") || t.startsWith("<html")) {
        msg = `${resp.status} ${resp.statusText} (upstream error)`;
      } else if (data.trim()) {
        msg = data.trim().slice(0, 300);
      }
    }
    throw new Error(typeof msg === "string" ? msg : JSON.stringify(msg));
  }
  return data;
}

export async function register(phone: string, password: string, name: string) {
  const data = await apiFetch("/auth/register", { method: "POST", body: JSON.stringify({ phone, password, name }) });
  setToken(data.access_token);
  return data.user as User;
}

export async function login(phone: string, password: string) {
  const data = await apiFetch("/auth/login", { method: "POST", body: JSON.stringify({ phone, password }) });
  setToken(data.access_token);
  return data.user as User;
}

export async function me() {
  return (await apiFetch("/auth/me")) as User;
}

export async function updateProfile(name: string, profile: string) {
  return await apiFetch("/user/profile", { method: "PUT", body: JSON.stringify({ name, profile }) });
}

export type Thread = { id: string; title: string; owner_user_id: string; permission: string; updated_at: string };

export async function listThreads() {
  return (await apiFetch("/threads")) as Thread[];
}

export async function createThread(title: string) {
  return await apiFetch("/threads", { method: "POST", body: JSON.stringify({ title }) });
}

export async function getThread(threadId: string) {
  return await apiFetch(`/threads/${threadId}`);
}

export async function updateThread(threadId: string, patch: { title?: string; canvas_md?: string }) {
  return await apiFetch(`/threads/${threadId}`, { method: "PUT", body: JSON.stringify(patch) });
}

export async function deleteThread(threadId: string) {
  return await apiFetch(`/threads/${threadId}`, { method: "DELETE" });
}

export async function shareThread(threadId: string, account: string, permission: "read" | "write") {
  return await apiFetch(`/threads/${threadId}/shares`, { method: "POST", body: JSON.stringify({ account, permission }) });
}

export type ThreadShare = { user_id: string; phone: string; email: string; name: string; permission: "read" | "write"; created_at: string };

export async function listThreadShares(threadId: string) {
  return (await apiFetch(`/threads/${threadId}/shares`)) as ThreadShare[];
}

export async function deleteThreadShare(threadId: string, sharedUserId: string) {
  return await apiFetch(`/threads/${threadId}/shares/${sharedUserId}`, { method: "DELETE" });
}

export type ChatMessage = { id: string; role: string; content: string; created_at: string; user_id: string };

export async function listMessages(threadId: string) {
  return (await apiFetch(`/threads/${threadId}/messages`)) as ChatMessage[];
}

export async function chatStream(threadId: string, message: string, onDelta: (d: string) => void) {
  const base = _apiBase ? _apiBase.replace(/\/$/, "") : "";
  const resp = await fetch(`${base}/api/threads/${threadId}/chat_stream`, {
    method: "POST",
    headers: { "Content-Type": "application/json", ...( _token ? { Authorization: `Bearer ${_token}` } : {} ) },
    body: JSON.stringify({ message }),
  });
  if (!resp.ok || !resp.body) {
    const text = await resp.text().catch(() => "");
    const t = (text || "").trim().toLowerCase();
    if (t.startsWith("<!doctype html") || t.startsWith("<html")) {
      throw new Error(`${resp.status} ${resp.statusText} (upstream error)`);
    }
    throw new Error(text || `${resp.status} ${resp.statusText}`);
  }
  const reader = resp.body.getReader();
  const decoder = new TextDecoder("utf-8");
  let buf = "";
  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buf += decoder.decode(value, { stream: true });
    while (true) {
      const idx = buf.indexOf("\n\n");
      if (idx < 0) break;
      const chunk = buf.slice(0, idx);
      buf = buf.slice(idx + 2);
      const line = chunk.split("\n").find((l) => l.startsWith("data: "));
      if (!line) continue;
      const jsonStr = line.slice(6);
      try {
        const evt = JSON.parse(jsonStr);
        if (evt.delta) onDelta(String(evt.delta));
        if (evt.error) throw new Error(String(evt.error));
      } catch (e: any) {
        throw e;
      }
    }
  }
}

export async function visionDescribe(file: File) {
  const fd = new FormData();
  fd.append("file", file);
  return await apiFetch("/vision/describe", { method: "POST", body: fd });
}

export async function ingestPaste(text: string, source_name: string) {
  return await apiFetch("/knowledge/ingest/paste", { method: "POST", body: JSON.stringify({ text, source_name }) });
}

export async function knowledgeStats() {
  return await apiFetch("/knowledge/stats");
}

export async function submitTeaching(threadId: string, instruction: string, max_messages: number = 80) {
  return await apiFetch("/knowledge/teaching/submit", { method: "POST", body: JSON.stringify({ thread_id: threadId, instruction, max_messages }) });
}

export type TeachingSubmission = { id: string; thread_id: string; status: string; title: string; points: number; created_at: string; updated_at: string; admin_comment?: string };

export async function listTeachingSubmissions() {
  return (await apiFetch("/knowledge/teaching/submissions")) as TeachingSubmission[];
}

export async function getTeachingSubmission(reviewId: string) {
  return await apiFetch(`/knowledge/teaching/submissions/${reviewId}`);
}

export type DataSource = { id: string; kind: string; name: string; updated_at: string };

export async function listSources() {
  return (await apiFetch("/sources")) as DataSource[];
}

export async function getSource(sourceId: string) {
  return await apiFetch(`/sources/${sourceId}`);
}

export async function createSource(kind: "sql" | "odata" | "paste", name: string, config: any) {
  return await apiFetch("/sources", { method: "POST", body: JSON.stringify({ kind, name, config }) });
}

export async function updateSource(sourceId: string, kind: "sql" | "odata" | "paste", name: string, config: any) {
  return await apiFetch(`/sources/${sourceId}`, { method: "PUT", body: JSON.stringify({ kind, name, config }) });
}

export async function deleteSource(sourceId: string) {
  return await apiFetch(`/sources/${sourceId}`, { method: "DELETE" });
}

export async function ingestSource(sourceId: string, max_items: number) {
  return await apiFetch(`/sources/${sourceId}/ingest`, { method: "POST", body: JSON.stringify({ max_items }) });
}

// Admin
export type AdminUser = { id: string; phone: string; email: string; name: string; is_admin: boolean; allow_any_topic: boolean; created_at: string };

export async function adminListUsers() {
  return (await apiFetch("/admin/users")) as AdminUser[];
}

export async function adminUpdateUser(userId: string, patch: { is_admin?: boolean; allow_any_topic?: boolean }) {
  return await apiFetch(`/admin/users/${userId}`, { method: "PUT", body: JSON.stringify(patch) });
}

export async function adminGetConfig() {
  return await apiFetch("/admin/config");
}

export async function adminSetConfig(key: string, value: string) {
  return await apiFetch("/admin/config", { method: "PUT", body: JSON.stringify({ key, value }) });
}

export type AdminTeachingReview = {
  id: string;
  thread_id: string;
  status: string;
  title: string;
  points: number;
  submitter_user_id: string;
  submitter_phone: string;
  submitter_email: string;
  submitter_name: string;
  created_at: string;
  updated_at: string;
  admin_comment?: string;
};

export async function adminListTeachingReviews(status: "pending" | "approved" | "rejected" | "all" = "pending") {
  return (await apiFetch(`/admin/knowledge/reviews?status=${encodeURIComponent(status)}`)) as AdminTeachingReview[];
}

export async function adminApproveTeachingReview(reviewId: string) {
  return await apiFetch(`/admin/knowledge/reviews/${reviewId}/approve`, { method: "POST" });
}

export async function adminRejectTeachingReview(reviewId: string, comment: string) {
  return await apiFetch(`/admin/knowledge/reviews/${reviewId}/reject`, { method: "POST", body: JSON.stringify({ comment }) });
}
