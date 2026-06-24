export default function Sidebar({ setPage, page }) {
  const items = [{ key: "therapy", label: "Practice", icon: "🎤" }];

  return (
    <aside className="sidebar">
      <div className="sidebar-decorations">
        <span className="sidebar-sparkle">✨</span>
        <span className="sidebar-sparkle small">⭐</span>
      </div>

      <div className="sidebar-brand">
        <div className="sidebar-brand-icon">🎵</div>
        <div>
          <p className="sidebar-brand-kicker">Speech Fun</p>
          <h2>VaakSuddhi</h2>
        </div>
      </div>

      <nav className="sidebar-nav">
        {items.map((item) => (
          <button
            key={item.key}
            className={`sidebar-btn ${page === item.key ? "active" : ""}`}
            onClick={() => setPage(item.key)}
          >
            <span className="sidebar-btn-icon">{item.icon}</span>
            <span>{item.label}</span>
          </button>
        ))}
      </nav>

      <div className="sidebar-footer">
        <span>🌈</span>
        <span>Ready to play!</span>
      </div>
    </aside>
  );
}