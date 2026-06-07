// Thin fetch wrapper. All calls go through the Vite proxy at /api.
const BASE = "/api";

function getToken() {
  return localStorage.getItem("token") || "";
}

async function request(path, { method = "GET", body, auth = true } = {}) {
  const headers = { "Content-Type": "application/json" };
  if (auth && getToken()) headers["Authorization"] = `Bearer ${getToken()}`;

  const res = await fetch(`${BASE}${path}`, {
    method,
    headers,
    body: body ? JSON.stringify(body) : undefined,
  });

  if (!res.ok) {
    let detail = `Request failed (${res.status})`;
    try {
      const data = await res.json();
      detail = data.detail || detail;
    } catch (_) {}
    const err = new Error(detail);
    err.status = res.status;
    throw err;
  }
  if (res.status === 204) return null;
  return res.json();
}

export const api = {
  signup: (email, username, password) =>
    request("/auth/signup", { method: "POST", auth: false, body: { email, username, password } }),
  login: (email, password) =>
    request("/auth/login", { method: "POST", auth: false, body: { email, password } }),

  listThreads: () => request("/threads"),
  createThread: (title) => request("/threads", { method: "POST", body: { title } }),
  renameThread: (id, title) => request(`/threads/${id}`, { method: "PATCH", body: { title } }),
  deleteThread: (id) => request(`/threads/${id}`, { method: "DELETE" }),
  getMessages: (id) => request(`/threads/${id}/messages`),

  chat: (thread_id, message) => request("/chat", { method: "POST", body: { thread_id, message } }),
};

export { getToken };