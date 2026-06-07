import React, { useEffect, useRef, useState } from "react";

function ReasoningPanel({ meta }) {
  const [open, setOpen] = useState(false);
  const path = meta?.reasoning_path || [];
  const hasMeta = (meta?.hops ?? 0) > 0 || path.length > 0;
  if (!hasMeta) return null;
  return (
    <div className="reasoning">
      <button className="reasoning-toggle" onClick={() => setOpen(!open)}>
        {open ? "▾" : "▸"} reasoning · {meta.hops} hops · {Math.round((meta.confidence || 0) * 100)}% confidence
      </button>
      {open && (
        <div className="reasoning-body">
          {path.length > 0 ? (
            <ol className="reasoning-steps">
              {path.map((step, i) => (
                <li key={i}><code>{step}</code></li>
              ))}
            </ol>
          ) : (
            <div className="muted">No explicit path recorded.</div>
          )}
          {meta.evidence && meta.evidence.length > 0 && (
            <div className="evidence">
              <div className="evidence-title">Evidence</div>
              <ul>{meta.evidence.map((e, i) => <li key={i}>{e}</li>)}</ul>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

function Bubble({ msg }) {
  const isUser = msg.role === "user";
  return (
    <div className={`bubble-row ${isUser ? "user" : "assistant"}`}>
      <div className="bubble-avatar">{isUser ? "you" : "◆"}</div>
      <div className="bubble">
        <div className="bubble-content">{msg.content}</div>
        {!isUser && <ReasoningPanel meta={msg.meta} />}
      </div>
    </div>
  );
}

export default function Chat({ messages, busy, onSend, onOpenSidebar, threadTitle }) {
  const [text, setText] = useState("");
  const scrollRef = useRef(null);
  const taRef = useRef(null);

  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: "smooth" });
  }, [messages, busy]);

  function submit(e) {
    e.preventDefault();
    const v = text.trim();
    if (!v || busy) return;
    onSend(v);
    setText("");
    if (taRef.current) taRef.current.style.height = "auto";
  }

  function autoGrow(e) {
    setText(e.target.value);
    e.target.style.height = "auto";
    e.target.style.height = Math.min(e.target.scrollHeight, 200) + "px";
  }

  return (
    <main className="chat">
      <header className="chat-header">
        <button className="icon-btn open-mobile" onClick={onOpenSidebar}>☰</button>
        <span className="chat-title">{threadTitle || "New chat"}</span>
      </header>

      <div className="messages" ref={scrollRef}>
        {messages.length === 0 && !busy && (
          <div className="welcome">
            <div className="welcome-mark">◆</div>
            <h1>Ask about a product.</h1>
            <p>The agent walks the knowledge graph across multiple hops and shows its reasoning path.</p>
            <div className="examples">
              {[
                "lightweight gaming laptop with good battery",
                "wireless earbuds with strong bass but not bulky",
                "phone with a good camera but no overheating",
              ].map((ex) => (
                <button key={ex} onClick={() => onSend(ex)}>{ex}</button>
              ))}
            </div>
          </div>
        )}

        {messages.map((m) => <Bubble key={m.id} msg={m} />)}

        {busy && (
          <div className="bubble-row assistant">
            <div className="bubble-avatar">◆</div>
            <div className="bubble">
              <div className="thinking">
                <span></span><span></span><span></span> reasoning over the graph…
              </div>
            </div>
          </div>
        )}
      </div>

      <form className="composer" onSubmit={submit}>
        <textarea
          ref={taRef}
          rows={1}
          placeholder="Message Atlas…"
          value={text}
          onChange={autoGrow}
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.shiftKey) submit(e);
          }}
        />
        <button className="send" disabled={busy || !text.trim()} title="Send">↑</button>
      </form>
    </main>
  );
}
