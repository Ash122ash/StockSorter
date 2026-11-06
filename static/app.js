const state = {
  stocks: [],
  sortKey: "price",
  sortDir: "desc", // "asc" | "desc"
  filter: "",
  universe: "combined",
};

const el = {
  universeSelect: document.getElementById("universeSelect"),
  searchInput: document.getElementById("searchInput"),
  refreshBtn: document.getElementById("refreshBtn"),
  countLabel: document.getElementById("countLabel"),
  updatedLabel: document.getElementById("updatedLabel"),
  tableBody: document.getElementById("tableBody"),
  tickerTrack: document.getElementById("tickerTrack"),
  clock: document.getElementById("clock"),
  marketDot: document.getElementById("marketDot"),
  marketState: document.getElementById("marketState"),
};

// ---------- Clock / market-hours indicator (IST, 9:15–15:30 Mon–Fri) ----------

function updateClock() {
  const now = new Date();
  const istString = now.toLocaleString("en-US", { timeZone: "Asia/Kolkata", hour12: false });
  const ist = new Date(istString);

  const h = ist.getHours();
  const m = ist.getMinutes();
  const day = ist.getDay(); // 0 = Sun, 6 = Sat
  const minutesNow = h * 60 + m;
  const isWeekday = day >= 1 && day <= 5;
  const isOpen = isWeekday && minutesNow >= 9 * 60 + 15 && minutesNow <= 15 * 60 + 30;

  el.clock.textContent = ist.toLocaleTimeString("en-IN", { hour12: false }) + " IST";
  el.marketDot.className = "dot " + (isOpen ? "open" : "closed");
  el.marketState.textContent = isOpen ? "Market open" : "Market closed";
}
updateClock();
setInterval(updateClock, 1000);

// ---------- Data fetching ----------

async function loadStocks(force = false) {
  setLoading();
  try {
    const res = await fetch(`/api/stocks?universe=${state.universe}${force ? "&refresh=1" : ""}`);
    const data = await res.json();
    if (!data.ok) throw new Error(data.error || "Failed to load data");

    state.stocks = data.stocks;
    el.updatedLabel.textContent = "Updated " + new Date(data.fetched_at * 1000).toLocaleTimeString();
    render();
    renderTicker();
  } catch (err) {
    el.tableBody.innerHTML = `<tr><td colspan="6" class="empty-row">Couldn't load data: ${escapeHtml(err.message)}</td></tr>`;
  } finally {
    el.refreshBtn.classList.remove("spinning");
  }
}

function setLoading() {
  el.tableBody.innerHTML = `<tr><td colspan="6" class="loading-row">Fetching prices… this can take a while for the full stock list.</td></tr>`;
  el.countLabel.textContent = "Loading…";
}

// ---------- Rendering ----------

function getFilteredSorted() {
  const q = state.filter.trim().toLowerCase();
  let rows = state.stocks;
  if (q) {
    rows = rows.filter(
      (s) => s.symbol.toLowerCase().includes(q) || s.name.toLowerCase().includes(q)
    );
  }
  const dir = state.sortDir === "asc" ? 1 : -1;
  const key = state.sortKey;
  rows = [...rows].sort((a, b) => {
    const va = a[key], vb = b[key];
    if (typeof va === "string") return va.localeCompare(vb) * dir;
    return (va - vb) * dir;
  });
  return rows;
}

function render() {
  const rows = getFilteredSorted();
  el.countLabel.textContent = `${rows.length} of ${state.stocks.length} stocks`;

  if (rows.length === 0) {
    el.tableBody.innerHTML = `<tr><td colspan="6" class="empty-row">No stocks match your filter.</td></tr>`;
  } else {
    el.tableBody.innerHTML = rows.map(rowHtml).join("");
  }

  document.querySelectorAll("th.sortable").forEach((th) => {
    const active = th.dataset.key === state.sortKey;
    th.classList.toggle("active", active);
    th.querySelector(".arrow").textContent = active ? (state.sortDir === "asc" ? "▲" : "▼") : "";
  });
}

function rowHtml(s) {
  const dirClass = s.change > 0 ? "up" : s.change < 0 ? "down" : "";
  const sign = s.change > 0 ? "+" : "";
  return `
    <tr>
      <td class="symbol">${escapeHtml(s.symbol)}</td>
      <td class="name">${escapeHtml(s.name)}</td>
      <td class="exchange">${escapeHtml(s.exchange || "NSE")}</td>
      <td class="num">₹${s.price.toFixed(2)}</td>
      <td class="num change ${dirClass}">${sign}${s.change.toFixed(2)}</td>
      <td class="num pct ${dirClass}">${sign}${s.change_pct.toFixed(2)}%</td>
    </tr>`;
}

function renderTicker() {
  if (state.stocks.length === 0) return;
  const sample = [...state.stocks].sort((a, b) => Math.abs(b.change_pct) - Math.abs(a.change_pct)).slice(0, 30);
  const items = sample
    .map((s) => {
      const dirClass = s.change >= 0 ? "up" : "down";
      const sign = s.change >= 0 ? "+" : "";
      return `<span class="ticker-item"><span class="sym">${escapeHtml(s.symbol)}</span> ₹${s.price.toFixed(2)} <span class="${dirClass}">${sign}${s.change_pct.toFixed(2)}%</span></span>`;
    })
    .join("");
  el.tickerTrack.innerHTML = items + items; // duplicate for seamless loop
}

function escapeHtml(str) {
  return String(str).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

// ---------- Events ----------

document.querySelectorAll("th.sortable").forEach((th) => {
  th.addEventListener("click", () => {
    const key = th.dataset.key;
    if (state.sortKey === key) {
      state.sortDir = state.sortDir === "asc" ? "desc" : "asc";
    } else {
      state.sortKey = key;
      state.sortDir = key === "symbol" || key === "name" ? "asc" : "desc";
    }
    render();
  });
});

el.searchInput.addEventListener("input", (e) => {
  state.filter = e.target.value;
  render();
});

el.universeSelect.addEventListener("change", (e) => {
  state.universe = e.target.value;
  loadStocks();
});

el.refreshBtn.addEventListener("click", () => {
  el.refreshBtn.classList.add("spinning");
  loadStocks(true);
});

// ---------- Init ----------

loadStocks();
