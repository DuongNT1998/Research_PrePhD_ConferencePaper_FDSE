import React, { useEffect, useState, useCallback } from "react";
import Auth from "./components/Auth.jsx";
import Sidebar from "./components/Sidebar.jsx";
import Chat from "./components/Chat.jsx";
import { api, getToken } from "./api.js";

export default function App() {
  const [user, setUser] = useState(() => {
    try { return JSON.parse(localStorage.getItem("user")); } catch { return null; }
  });
  const [authed, setAuthed] = useState(!!getToken());

  const [threads, setThreads] = useState([]);
  const [activeId, setActiveId] = useState(null);
  const [messages, setMessages] = useState([]);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [sidebarOpen, setSidebarOpen] = useState(false);

  const logout = useCallback(() => {
    localStorage.removeItem("token");
    localStorage.removeItem("user");
    setAuthed(false);
    setUser(null);
    setThreads([]);
    setMessages([]);
    setActiveId(null);
  }, []);

  const loadThreads = useCallback(async () => {
    try {
      const t = await api.listThreads();
      setThreads(t);
      return t;
    } catch (e) {
      if (e.status === 401) logout();
      else setError(e.message);
      return [];
    }
  }, [logout]);

  useEffect(() => {
    if (authed) loadThreads();
  }, [authed, loadThreads]);

  async function selectThread(id) {
    setActiveId(id);
    setSidebarOpen(false);
    setError("");
    try {
      const msgs = await api.getMessages(id);
      setMessages(msgs);
    } catch (e) {
      setError(e.message);
    }
  }

  function newChat() {
    setActiveId(null);
    setMessages([]);
    setSidebarOpen(false);
  }

  async function renameThread(id, title) {
    try {
      await api.renameThread(id, title);
      setThreads((ts) => ts.map((t) => (t.id === id ? { ...t, title } : t)));
    } catch (e) { setError(e.message); }
  }

  async function deleteThread(id) {
    try {
      await api.deleteThread(id);
      setThreads((ts) => ts.filter((t) => t.id !== id));
      if (id === activeId) newChat();
    } catch (e) { setError(e.message); }
  }

  async function send(message) {
    setBusy(true);
    setError("");
    // optimistic user bubble
    const tempId = "temp-" + Date.now();
    setMessages((m) => [...m, { id: tempId, role: "user", content: message, meta: {} }]);
    try {
      const res = await api.chat(activeId, message);
      // replace optimistic + append assistant
      setMessages((m) => {
        const withoutTemp = m.filter((x) => x.id !== tempId);
        return [...withoutTemp, res.user_message, res.assistant_message];
      });
      if (!activeId) {
        setActiveId(res.thread_id);
        await loadThreads();
      } else {
        // bump title/updated order
        setThreads((ts) => {
          const found = ts.find((t) => t.id === res.thread_id);
          if (found) found.title = res.thread_title;
          return [...ts];
        });
      }
    } catch (e) {
      setMessages((m) => m.filter((x) => x.id !== tempId));
      setError(e.message);
      if (e.status === 401) logout();
    } finally {
      setBusy(false);
    }
  }

  if (!authed) {
    return <Auth onAuthed={(u) => { setUser(u); setAuthed(true); }} />;
  }

  const activeThread = threads.find((t) => t.id === activeId);

  return (
    <div className="app">
      {sidebarOpen && <div className="scrim" onClick={() => setSidebarOpen(false)} />}
      <Sidebar
        threads={threads}
        activeId={activeId}
        user={user}
        onSelect={selectThread}
        onNew={newChat}
        onRename={renameThread}
        onDelete={deleteThread}
        onLogout={logout}
        open={sidebarOpen}
        onClose={() => setSidebarOpen(false)}
      />
      <div className="main-col">
        {error && <div className="error-banner">{error} <button onClick={() => setError("")}>✕</button></div>}
        <Chat
          messages={messages}
          busy={busy}
          onSend={send}
          onOpenSidebar={() => setSidebarOpen(true)}
          threadTitle={activeThread?.title}
        />
      </div>
    </div>
  );
}
