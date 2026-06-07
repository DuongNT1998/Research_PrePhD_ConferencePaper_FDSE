import React, { useState } from "react";

export default function Sidebar({
  threads,
  activeId,
  user,
  onSelect,
  onNew,
  onRename,
  onDelete,
  onLogout,
  open,
  onClose,
}) {
  const [editingId, setEditingId] = useState(null);
  const [editValue, setEditValue] = useState("");

  function startEdit(t) {
    setEditingId(t.id);
    setEditValue(t.title);
  }
  function commitEdit(id) {
    const v = editValue.trim();
    if (v) onRename(id, v);
    setEditingId(null);
  }

  return (
    <aside className={`sidebar ${open ? "open" : ""}`}>
      <div className="sidebar-top">
        <div className="brand">
          <span className="brand-mark">◆</span> Atlas
        </div>
        <button className="icon-btn close-mobile" onClick={onClose} title="Close">✕</button>
      </div>

      <button className="new-chat" onClick={onNew}>＋ New chat</button>

      <div className="thread-list">
        {threads.length === 0 && <div className="empty-threads">No conversations yet.</div>}
        {threads.map((t) => (
          <div
            key={t.id}
            className={`thread-item ${t.id === activeId ? "active" : ""}`}
            onClick={() => onSelect(t.id)}
          >
            {editingId === t.id ? (
              <input
                className="thread-edit"
                value={editValue}
                autoFocus
                onChange={(e) => setEditValue(e.target.value)}
                onBlur={() => commitEdit(t.id)}
                onKeyDown={(e) => e.key === "Enter" && commitEdit(t.id)}
                onClick={(e) => e.stopPropagation()}
              />
            ) : (
              <span className="thread-title" title={t.title}>{t.title}</span>
            )}
            <span className="thread-actions" onClick={(e) => e.stopPropagation()}>
              <button className="icon-btn" title="Rename" onClick={() => startEdit(t)}>✎</button>
              <button className="icon-btn" title="Delete" onClick={() => onDelete(t.id)}>🗑</button>
            </span>
          </div>
        ))}
      </div>

      <div className="sidebar-footer">
        <div className="user-chip">
          <span className="avatar">{(user?.username || "?")[0]?.toUpperCase()}</span>
          <span className="user-name">{user?.username}</span>
        </div>
        <button className="logout" onClick={onLogout}>Log out</button>
      </div>
    </aside>
  );
}
