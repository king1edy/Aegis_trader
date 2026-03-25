"""
Journal Router
==============
FastAPI APIRouter exposing journal REST endpoints and the dashboard UI.

Mount this router onto the main FastAPI app in main.py:

    from journal.router import journal_router
    app.include_router(journal_router)

Routes
------
GET  /                              → dashboard HTML (single-page app)
GET  /api/journal/stats             → summary stats
GET  /api/journal/trades            → paginated trade list
GET  /api/journal/trades/open       → open positions
PATCH /api/journal/trades/{id}      → annotate setup_tag / journal_notes
GET  /api/journal/deals             → raw deal audit log
GET  /api/journal/analysis/sessions → performance by session
GET  /api/journal/analysis/hours    → performance by hour of day
GET  /api/journal/analysis/days     → performance by day of week
GET  /api/journal/analysis/setups   → performance by setup tag
GET  /api/journal/analysis/symbols  → performance by symbol
GET  /api/journal/analysis/direction → BUY vs SELL breakdown
GET  /api/journal/equity            → balance/equity time series
GET  /api/journal/tags              → all setup tags
POST /api/journal/tags              → create new setup tag
"""

from __future__ import annotations

import logging
from typing import Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from sqlalchemy import select

from auth.dependencies import get_tenant_id
from database.models import SetupTag, Trade
from database.repository import get_session
from journal import analyzer

logger = logging.getLogger("journal.router")

journal_router = APIRouter(tags=["Journal"])


# =============================================================================
# Request / Response schemas
# =============================================================================

class TradeAnnotation(BaseModel):
    setup_tag: Optional[str] = None
    journal_notes: Optional[str] = None


class NewTag(BaseModel):
    name: str
    color: str = "#3B82F6"
    description: Optional[str] = None


# =============================================================================
# API endpoints
# =============================================================================

@journal_router.get("/api/journal/stats")
async def get_stats(tenant_id: UUID = Depends(get_tenant_id)):
    """Overall trading statistics across all closed trades."""
    return await analyzer.summary_stats(tenant_id)


@journal_router.get("/api/journal/trades")
async def get_trades(
    tenant_id: UUID = Depends(get_tenant_id),
    page:      int = Query(1, ge=1),
    per_page:  int = Query(50, ge=1, le=500),
    symbol:    Optional[str] = None,
    direction: Optional[str] = None,
    session:   Optional[str] = None,
    setup:     Optional[str] = None,
    source:    Optional[str] = None,
    status:    Optional[str] = None,
):
    """Paginated, filterable list of all trades."""
    return await analyzer.list_trades(
        tenant_id=tenant_id,
        page=page,
        per_page=per_page,
        symbol=symbol,
        direction=direction,
        session_=session,
        setup_tag=setup,
        source=source,
        status=status,
    )


@journal_router.get("/api/journal/trades/open")
async def get_open_trades(tenant_id: UUID = Depends(get_tenant_id)):
    """All currently open (and partially-closed) positions."""
    return await analyzer.open_trades(tenant_id)


@journal_router.patch("/api/journal/trades/{trade_id}")
async def annotate_trade(
    trade_id: UUID,
    body: TradeAnnotation,
    tenant_id: UUID = Depends(get_tenant_id),
):
    """
    Update setup_tag and/or journal_notes on a trade.
    Called inline from the dashboard table rows.
    """
    async with get_session() as session:
        result = await session.execute(
            select(Trade).where(Trade.id == trade_id, Trade.tenant_id == tenant_id)
        )
        trade = result.scalar_one_or_none()
        if trade is None:
            raise HTTPException(status_code=404, detail="Trade not found")

        if body.setup_tag is not None:
            trade.setup_tag = body.setup_tag or None
        if body.journal_notes is not None:
            trade.journal_notes = body.journal_notes or None

        await session.commit()

    return {"status": "ok", "trade_id": str(trade_id)}


@journal_router.get("/api/journal/deals")
async def get_deals(
    tenant_id: UUID = Depends(get_tenant_id),
    page:     int = Query(1, ge=1),
    per_page: int = Query(100, ge=1, le=1000),
):
    """Raw MT5 deal audit log."""
    return await analyzer.list_deals(tenant_id, page=page, per_page=per_page)


# ── Analysis endpoints ────────────────────────────────────────────────────────

@journal_router.get("/api/journal/analysis/sessions")
async def analysis_sessions(tenant_id: UUID = Depends(get_tenant_id)):
    return await analyzer.by_session(tenant_id)


@journal_router.get("/api/journal/analysis/hours")
async def analysis_hours(tenant_id: UUID = Depends(get_tenant_id)):
    return await analyzer.by_hour(tenant_id)


@journal_router.get("/api/journal/analysis/days")
async def analysis_days(tenant_id: UUID = Depends(get_tenant_id)):
    return await analyzer.by_day_of_week(tenant_id)


@journal_router.get("/api/journal/analysis/setups")
async def analysis_setups(tenant_id: UUID = Depends(get_tenant_id)):
    return await analyzer.by_setup_tag(tenant_id)


@journal_router.get("/api/journal/analysis/symbols")
async def analysis_symbols(tenant_id: UUID = Depends(get_tenant_id)):
    return await analyzer.by_symbol(tenant_id)


@journal_router.get("/api/journal/analysis/direction")
async def analysis_direction(tenant_id: UUID = Depends(get_tenant_id)):
    return await analyzer.by_direction(tenant_id)


@journal_router.get("/api/journal/equity")
async def get_equity_curve(
    tenant_id: UUID = Depends(get_tenant_id),
    limit: int = Query(500, ge=10, le=5000),
):
    return await analyzer.equity_curve(tenant_id, limit=limit)


# ── Setup tag endpoints ───────────────────────────────────────────────────────

@journal_router.get("/api/journal/tags")
async def get_tags(tenant_id: UUID = Depends(get_tenant_id)):
    return await analyzer.get_tags(tenant_id)


@journal_router.post("/api/journal/tags", status_code=201)
async def create_tag(
    body: NewTag,
    tenant_id: UUID = Depends(get_tenant_id),
):
    async with get_session() as session:
        existing = await session.execute(
            select(SetupTag).where(
                SetupTag.name == body.name, SetupTag.tenant_id == tenant_id
            )
        )
        if existing.scalar_one_or_none() is not None:
            raise HTTPException(status_code=409, detail="Tag already exists")
        tag = SetupTag(
            name=body.name, color=body.color,
            description=body.description, tenant_id=tenant_id,
        )
        session.add(tag)
        await session.commit()
        await session.refresh(tag)
    return {"id": tag.id, "name": tag.name, "color": tag.color}


# =============================================================================
# Dashboard HTML
# =============================================================================

_DASHBOARD_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Aegis Trade Journal</title>
  <script src="https://cdn.tailwindcss.com"></script>
  <script src="https://cdn.plot.ly/plotly-2.32.0.min.js"></script>
  <style>
    body { background:#0f172a; color:#e2e8f0; font-family:'Inter',sans-serif; }
    .card { background:#1e293b; border:1px solid #334155; border-radius:0.75rem; }
    .tag-badge { display:inline-block; padding:2px 8px; border-radius:9999px; font-size:0.72rem; font-weight:600; }
    select, input[type=text] { background:#0f172a; border:1px solid #334155; color:#e2e8f0; border-radius:0.375rem; padding:4px 8px; }
    select:focus, input[type=text]:focus { outline:1px solid #3b82f6; }
    table { border-collapse:collapse; width:100%; }
    th { background:#0f172a; color:#94a3b8; font-size:0.75rem; text-transform:uppercase; letter-spacing:0.05em; padding:10px 12px; text-align:left; position:sticky; top:0; z-index:1; }
    td { padding:9px 12px; border-bottom:1px solid #1e293b; font-size:0.85rem; }
    tr:hover td { background:#1e293b; }
    .win  { color:#10b981; }
    .loss { color:#ef4444; }
    .be   { color:#94a3b8; }
    .pnl-pos { color:#10b981; font-weight:600; }
    .pnl-neg { color:#ef4444; font-weight:600; }
    .source-ea     { background:#1e3a5f; color:#60a5fa; }
    .source-manual { background:#1f2937; color:#a78bfa; }
    .btn { padding:6px 14px; border-radius:0.375rem; font-size:0.82rem; cursor:pointer; border:none; font-weight:500; }
    .btn-primary { background:#3b82f6; color:#fff; }
    .btn-primary:hover { background:#2563eb; }
    .btn-ghost  { background:#334155; color:#cbd5e1; }
    .btn-ghost:hover  { background:#475569; }
    .plotly-chart { width:100%; height:220px; }
    .equity-chart { width:100%; height:260px; }
    #open-table td, #open-table th { font-size:0.78rem; }
  </style>
</head>
<body>

<!-- ── Header ────────────────────────────────────────────────────────────── -->
<header class="border-b border-slate-700 bg-slate-900 px-6 py-3 flex items-center justify-between sticky top-0 z-50">
  <div class="flex items-center gap-3">
    <span class="text-blue-400 font-bold text-lg tracking-wide">⚡ AEGIS</span>
    <span class="text-slate-400 text-sm">Trade Journal</span>
  </div>
  <div class="flex items-center gap-4 text-sm">
    <span id="hdr-balance" class="text-slate-300">Balance: —</span>
    <span id="hdr-equity"  class="text-slate-300">Equity: —</span>
    <span id="hdr-open"    class="text-slate-400">Open: —</span>
    <span id="hdr-today"   class="text-slate-400">Today: —</span>
    <button class="btn btn-ghost text-xs" onclick="refresh()">↻ Refresh</button>
    <span id="hdr-tier" class="tag-badge text-xs" style="display:none"></span>
    <span id="hdr-user" class="text-slate-400 text-xs"></span>
    <button id="btn-settings" class="btn btn-ghost text-xs" onclick="openSettings()" style="display:none">Settings</button>
    <button id="btn-logout" class="btn btn-ghost text-xs" onclick="logout()" style="display:none">Logout</button>
  </div>
</header>

<!-- ── Auth Modal ────────────────────────────────────────────────────────── -->
<div id="auth-modal" class="fixed inset-0 bg-black/80 z-[100] flex items-center justify-center" style="display:none">
  <div class="card p-8 w-96">
    <h2 class="text-xl font-bold mb-1 text-blue-400">⚡ AEGIS</h2>
    <p class="text-sm text-slate-400 mb-6">Trade Journal</p>
    <div id="auth-error" class="text-red-400 text-sm mb-3" style="display:none"></div>
    <!-- Login form -->
    <div id="login-form">
      <input id="login-email" type="text" placeholder="Email or username" class="w-full mb-3 p-2" />
      <input id="login-pass" type="password" placeholder="Password" class="w-full mb-4 p-2" />
      <button class="btn btn-primary w-full mb-3" onclick="doLogin()">Sign In</button>
      <p class="text-xs text-slate-400 text-center">No account? <a href="#" class="text-blue-400" onclick="showRegister();return false">Register</a></p>
    </div>
    <!-- Register form -->
    <div id="register-form" style="display:none">
      <input id="reg-email" type="text" placeholder="Email" class="w-full mb-3 p-2" />
      <input id="reg-user" type="text" placeholder="Username" class="w-full mb-3 p-2" />
      <input id="reg-pass" type="password" placeholder="Password (min 8 chars)" class="w-full mb-4 p-2" />
      <button class="btn btn-primary w-full mb-3" onclick="doRegister()">Create Account</button>
      <p class="text-xs text-slate-400 text-center">Have an account? <a href="#" class="text-blue-400" onclick="showLogin();return false">Sign In</a></p>
    </div>
  </div>
</div>

<!-- ── Settings Modal ───────────────────────────────────────────────────── -->
<div id="settings-modal" class="fixed inset-0 bg-black/80 z-[100] flex items-center justify-center" style="display:none">
  <div class="card p-6 w-[32rem] max-h-[80vh] overflow-y-auto">
    <div class="flex justify-between items-center mb-4">
      <h2 class="text-lg font-bold text-blue-400">Settings</h2>
      <button class="text-slate-400 hover:text-slate-200 text-xl" onclick="closeSettings()">&times;</button>
    </div>
    <div id="settings-error" class="text-red-400 text-sm mb-3" style="display:none"></div>

    <!-- Risk Rules -->
    <h3 class="text-xs text-slate-400 uppercase tracking-wide mt-4 mb-2">Risk Rules</h3>
    <div class="grid grid-cols-2 gap-3 text-xs">
      <div><label class="text-slate-400">Max Daily Drawdown %</label><input id="set-drawdown" type="number" step="0.01" class="w-full mt-1 p-1.5" /></div>
      <div><label class="text-slate-400">Max Consecutive Losses</label><input id="set-consec" type="number" class="w-full mt-1 p-1.5" /></div>
      <div><label class="text-slate-400">Max Lot Size</label><input id="set-lot" type="number" step="0.01" class="w-full mt-1 p-1.5" /></div>
      <div><label class="text-slate-400">Max Open Positions</label><input id="set-positions" type="number" class="w-full mt-1 p-1.5" /></div>
      <div><label class="text-slate-400">Max Daily Trades</label><input id="set-daily-trades" type="number" class="w-full mt-1 p-1.5" /></div>
      <div><label class="text-slate-400">Pause on Rule Breach</label><select id="set-pause" class="w-full mt-1 p-1.5"><option value="true">Yes</option><option value="false">No</option></select></div>
    </div>

    <!-- Notifications -->
    <h3 class="text-xs text-slate-400 uppercase tracking-wide mt-4 mb-2">Notifications</h3>
    <div class="grid grid-cols-2 gap-3 text-xs">
      <div><label class="text-slate-400">Telegram Chat ID</label><input id="set-tg-chat" type="text" class="w-full mt-1 p-1.5" /></div>
      <div><label class="text-slate-400">Telegram Enabled</label><select id="set-tg-enabled" class="w-full mt-1 p-1.5"><option value="true">Yes</option><option value="false">No</option></select></div>
    </div>
    <div class="grid grid-cols-2 gap-2 text-xs mt-2">
      <label><input id="set-n-open" type="checkbox" class="mr-1" />Trade Open</label>
      <label><input id="set-n-close" type="checkbox" class="mr-1" />Trade Close</label>
      <label><input id="set-n-summary" type="checkbox" class="mr-1" />Daily Summary</label>
      <label><input id="set-n-risk" type="checkbox" class="mr-1" />Risk Breach</label>
    </div>

    <!-- MT5 Connection -->
    <h3 class="text-xs text-slate-400 uppercase tracking-wide mt-4 mb-2">MT5 Connection</h3>
    <div class="grid grid-cols-2 gap-3 text-xs">
      <div><label class="text-slate-400">MT5 Login</label><input id="set-mt5-login" type="number" class="w-full mt-1 p-1.5" /></div>
      <div><label class="text-slate-400">MT5 Server</label><input id="set-mt5-server" type="text" class="w-full mt-1 p-1.5" /></div>
      <div><label class="text-slate-400">Mode</label><select id="set-mt5-mode" class="w-full mt-1 p-1.5"><option value="ea">EA</option><option value="bridge">Bridge</option></select></div>
    </div>

    <button class="btn btn-primary w-full mt-5" onclick="saveSettings()">Save Settings</button>
    <div id="settings-saved" class="text-emerald-400 text-xs text-center mt-2" style="display:none">Settings saved</div>
  </div>
</div>

<main class="max-w-screen-2xl mx-auto px-4 py-6 space-y-6">

  <!-- ── Stat cards ──────────────────────────────────────────────────────── -->
  <section class="grid grid-cols-2 md:grid-cols-4 xl:grid-cols-8 gap-3" id="stat-cards">
    <div class="card p-4"><div class="text-xs text-slate-400">Total Trades</div><div class="text-2xl font-bold mt-1" id="s-total">—</div></div>
    <div class="card p-4"><div class="text-xs text-slate-400">Win Rate</div><div class="text-2xl font-bold mt-1 text-emerald-400" id="s-wr">—</div></div>
    <div class="card p-4"><div class="text-xs text-slate-400">Profit Factor</div><div class="text-2xl font-bold mt-1 text-blue-400" id="s-pf">—</div></div>
    <div class="card p-4"><div class="text-xs text-slate-400">Net P&amp;L</div><div class="text-2xl font-bold mt-1" id="s-pnl">—</div></div>
    <div class="card p-4"><div class="text-xs text-slate-400">Avg R:R</div><div class="text-2xl font-bold mt-1 text-violet-400" id="s-rr">—</div></div>
    <div class="card p-4"><div class="text-xs text-slate-400">Open Now</div><div class="text-2xl font-bold mt-1 text-amber-400" id="s-open">—</div></div>
    <div class="card p-4"><div class="text-xs text-slate-400">EA Trades</div><div class="text-2xl font-bold mt-1 text-blue-300" id="s-ea">—</div></div>
    <div class="card p-4"><div class="text-xs text-slate-400">Manual Trades</div><div class="text-2xl font-bold mt-1 text-violet-300" id="s-manual">—</div></div>
  </section>

  <!-- ── Analysis charts ─────────────────────────────────────────────────── -->
  <section class="grid grid-cols-1 md:grid-cols-3 gap-4">
    <div class="card p-4">
      <h3 class="text-xs text-slate-400 uppercase tracking-wide mb-2">Session Performance</h3>
      <div id="chart-session" class="plotly-chart"></div>
    </div>
    <div class="card p-4">
      <h3 class="text-xs text-slate-400 uppercase tracking-wide mb-2">Hour of Day</h3>
      <div id="chart-hour" class="plotly-chart"></div>
    </div>
    <div class="card p-4">
      <h3 class="text-xs text-slate-400 uppercase tracking-wide mb-2">Setup Tag Performance</h3>
      <div id="chart-setup" class="plotly-chart"></div>
    </div>
  </section>

  <!-- ── Equity curve ─────────────────────────────────────────────────────── -->
  <div class="card p-4">
    <h3 class="text-xs text-slate-400 uppercase tracking-wide mb-2">Equity Curve</h3>
    <div id="chart-equity" class="equity-chart"></div>
  </div>

  <!-- ── Open positions ──────────────────────────────────────────────────── -->
  <div class="card p-4">
    <h3 class="text-sm font-semibold text-slate-200 mb-3">Open Positions <span class="text-xs text-slate-500 font-normal">(auto-refresh 30s)</span></h3>
    <div class="overflow-x-auto">
      <table id="open-table">
        <thead><tr>
          <th>Symbol</th><th>Dir</th><th>Entry</th><th>SL</th><th>TP</th>
          <th>Lots</th><th>Source</th><th>Opened</th><th>Session</th>
        </tr></thead>
        <tbody id="open-body"><tr><td colspan="9" class="text-slate-500 text-center py-4">Loading…</td></tr></tbody>
      </table>
    </div>
  </div>

  <!-- ── Trade journal table ─────────────────────────────────────────────── -->
  <div class="card p-4">
    <div class="flex flex-wrap items-center justify-between gap-3 mb-3">
      <h3 class="text-sm font-semibold text-slate-200">Trade Journal</h3>
      <div class="flex flex-wrap gap-2 text-xs">
        <select id="f-symbol" onchange="loadTrades(1)"><option value="">All Symbols</option></select>
        <select id="f-dir" onchange="loadTrades(1)">
          <option value="">All Directions</option>
          <option value="BUY">BUY</option><option value="SELL">SELL</option>
        </select>
        <select id="f-session" onchange="loadTrades(1)">
          <option value="">All Sessions</option>
          <option value="London">London</option>
          <option value="New_York">New York</option>
          <option value="Asian">Asian</option>
          <option value="Off-Hours">Off-Hours</option>
        </select>
        <select id="f-source" onchange="loadTrades(1)">
          <option value="">All Sources</option>
          <option value="ea">EA</option><option value="manual">Manual</option>
        </select>
        <select id="f-status" onchange="loadTrades(1)">
          <option value="">All Statuses</option>
          <option value="CLOSED">Closed</option>
          <option value="OPEN">Open</option>
        </select>
      </div>
    </div>
    <div class="overflow-x-auto">
      <table>
        <thead><tr>
          <th>#</th><th>Symbol</th><th>Dir</th><th>Source</th>
          <th>Entry</th><th>Exit</th><th>P&amp;L</th><th>R:R</th>
          <th>Session</th><th>Outcome</th><th>Exit Reason</th>
          <th>Setup Tag</th><th>Notes</th>
        </tr></thead>
        <tbody id="trades-body"><tr><td colspan="13" class="text-slate-500 text-center py-4">Loading…</td></tr></tbody>
      </table>
    </div>
    <div class="flex items-center justify-between mt-3 text-xs text-slate-400">
      <span id="page-info">—</span>
      <div class="flex gap-2">
        <button class="btn btn-ghost text-xs" id="btn-prev" onclick="changePage(-1)">‹ Prev</button>
        <button class="btn btn-ghost text-xs" id="btn-next" onclick="changePage(1)">Next ›</button>
      </div>
    </div>
  </div>

</main>

<script>
const API = '';  // same origin
const TOKEN_KEY = 'aegis_token';
let currentPage = 1;
let totalPages  = 1;
let setupTags   = [];

// ── Auth helpers ─────────────────────────────────────────────────────────────
function getToken()  { return localStorage.getItem(TOKEN_KEY); }
function setToken(t) { localStorage.setItem(TOKEN_KEY, t); }
function clearToken(){ localStorage.removeItem(TOKEN_KEY); }

function showAuthModal() { $('auth-modal').style.display = 'flex'; }
function hideAuthModal() { $('auth-modal').style.display = 'none'; $('auth-error').style.display='none'; }
function showLogin()    { $('login-form').style.display=''; $('register-form').style.display='none'; }
function showRegister() { $('login-form').style.display='none'; $('register-form').style.display=''; }
function authError(msg) { const e=$('auth-error'); e.textContent=msg; e.style.display=''; }

async function doLogin() {
  const form = new URLSearchParams();
  form.append('username', $('login-email').value);
  form.append('password', $('login-pass').value);
  try {
    const r = await fetch(API+'/api/auth/login',{method:'POST',headers:{'Content-Type':'application/x-www-form-urlencoded'},body:form});
    if (!r.ok) { const d=await r.json(); authError(d.detail||'Login failed'); return; }
    const data = await r.json();
    setToken(data.access_token);
    hideAuthModal();
    await initDashboard();
  } catch(e) { authError('Network error'); }
}

async function doRegister() {
  try {
    const r = await fetch(API+'/api/auth/register',{
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({email:$('reg-email').value, username:$('reg-user').value, password:$('reg-pass').value})
    });
    if (!r.ok) { const d=await r.json(); authError(d.detail||'Registration failed'); return; }
    const data = await r.json();
    setToken(data.access_token);
    hideAuthModal();
    await initDashboard();
  } catch(e) { authError('Network error'); }
}

function logout() {
  clearToken();
  $('btn-logout').style.display='none';
  $('hdr-user').textContent='';
  showAuthModal();
}

// ── Utility ──────────────────────────────────────────────────────────────────
const $ = id => document.getElementById(id);

function fmt(v, decimals=2) {
  if (v === null || v === undefined) return '—';
  return parseFloat(v).toFixed(decimals);
}

function fmtDate(iso) {
  if (!iso) return '—';
  const d = new Date(iso);
  return d.toLocaleString('en-GB', {month:'short', day:'2-digit', hour:'2-digit', minute:'2-digit'});
}

function pnlClass(v) { return v > 0 ? 'pnl-pos' : v < 0 ? 'pnl-neg' : ''; }

function outcomeClass(o) {
  if (o === 'WIN') return 'win'; if (o === 'LOSS') return 'loss'; return 'be';
}

async function apiFetch(path) {
  const headers = {};
  const token = getToken();
  if (token) headers['Authorization'] = 'Bearer ' + token;
  const r = await fetch(API + path, {headers});
  if (r.status === 401) { clearToken(); showAuthModal(); throw new Error('Unauthorized'); }
  if (!r.ok) throw new Error(r.statusText);
  return r.json();
}

// ── Header / stat cards ───────────────────────────────────────────────────────
async function loadStats() {
  const s = await apiFetch('/api/journal/stats');
  $('s-total').textContent  = s.total_trades;
  $('s-wr').textContent     = (s.win_rate * 100).toFixed(1) + '%';
  $('s-pf').textContent     = s.profit_factor !== null ? fmt(s.profit_factor) : '—';
  const pnl = s.net_pnl;
  $('s-pnl').textContent    = (pnl >= 0 ? '+$' : '-$') + Math.abs(pnl).toFixed(2);
  $('s-pnl').className      = 'text-2xl font-bold mt-1 ' + (pnl >= 0 ? 'text-emerald-400' : 'text-red-400');
  $('s-rr').textContent     = fmt(s.avg_rr);
  $('s-open').textContent   = s.open_trades;
  $('s-ea').textContent     = s.ea_trades;
  $('s-manual').textContent = s.manual_trades;
}

async function loadLatestSnapshot() {
  try {
    const data = await apiFetch('/api/journal/equity?limit=2');
    if (data.length) {
      const latest = data[data.length - 1];
      $('hdr-balance').textContent = 'Balance: $' + fmt(latest.balance);
      $('hdr-equity').textContent  = 'Equity: $'  + fmt(latest.equity);
    }
  } catch(_) {}
}

// ── Open positions ────────────────────────────────────────────────────────────
async function loadOpen() {
  const trades = await apiFetch('/api/journal/trades/open');
  $('hdr-open').textContent = 'Open: ' + trades.length;
  const tbody = $('open-body');
  if (!trades.length) {
    tbody.innerHTML = '<tr><td colspan="9" class="text-slate-500 text-center py-4">No open positions</td></tr>';
    return;
  }
  tbody.innerHTML = trades.map(t => `
    <tr>
      <td class="font-mono font-semibold">${t.symbol}</td>
      <td class="${t.direction==='BUY'?'text-emerald-400':'text-red-400'} font-bold">${t.direction}</td>
      <td class="font-mono">${fmt(t.entry_price, 5)}</td>
      <td class="font-mono text-red-400">${fmt(t.stop_loss, 5)}</td>
      <td class="font-mono text-emerald-400">${t.take_profit_1 ? fmt(t.take_profit_1, 5) : '—'}</td>
      <td>${fmt(t.lot_size)}</td>
      <td><span class="tag-badge ${t.trade_source==='ea'?'source-ea':'source-manual'}">${t.trade_source}</span></td>
      <td>${fmtDate(t.entry_time)}</td>
      <td>${t.trading_session || '—'}</td>
    </tr>
  `).join('');
}

// ── Charts ────────────────────────────────────────────────────────────────────
const PLOTLY_LAYOUT = {
  paper_bgcolor:'transparent', plot_bgcolor:'transparent',
  font:{color:'#94a3b8', size:10},
  margin:{l:40, r:10, t:10, b:40},
  showlegend:false,
  xaxis:{gridcolor:'#1e293b', color:'#94a3b8'},
  yaxis:{gridcolor:'#1e293b', color:'#94a3b8'},
};
const PLOTLY_CFG = {displayModeBar:false, responsive:true};

async function loadSessionChart() {
  const data = await apiFetch('/api/journal/analysis/sessions');
  const names  = data.map(d => d.session);
  const wr     = data.map(d => +(d.win_rate * 100).toFixed(1));
  const pnl    = data.map(d => d.net_pnl);
  Plotly.newPlot('chart-session', [
    {type:'bar', name:'Win %', x:names, y:wr,
     marker:{color:wr.map(v => v >= 50 ? '#10b981' : '#ef4444')},
     text:wr.map(v => v+'%'), textposition:'outside'},
  ], {...PLOTLY_LAYOUT, yaxis:{...PLOTLY_LAYOUT.yaxis, title:'Win %'}}, PLOTLY_CFG);
}

async function loadHourChart() {
  const data = await apiFetch('/api/journal/analysis/hours');
  const hours = data.map(d => d.hour + ':00');
  const wr    = data.map(d => +(d.win_rate * 100).toFixed(1));
  Plotly.newPlot('chart-hour', [
    {type:'bar', x:hours, y:wr,
     marker:{color:wr.map(v => v >= 50 ? '#3b82f6' : '#6366f1')}},
  ], {...PLOTLY_LAYOUT, yaxis:{...PLOTLY_LAYOUT.yaxis, title:'Win %'}}, PLOTLY_CFG);
}

async function loadSetupChart() {
  const data = await apiFetch('/api/journal/analysis/setups');
  if (!data.length) { $('chart-setup').innerHTML = '<p class="text-slate-500 text-xs mt-4">No tagged trades yet</p>'; return; }
  const names = data.map(d => d.setup);
  const pnl   = data.map(d => d.net_pnl);
  Plotly.newPlot('chart-setup', [
    {type:'bar', x:names, y:pnl,
     marker:{color:pnl.map(v => v >= 0 ? '#10b981' : '#ef4444')},
     text:pnl.map(v => '$'+v.toFixed(0)), textposition:'outside'},
  ], {...PLOTLY_LAYOUT, yaxis:{...PLOTLY_LAYOUT.yaxis, title:'Net P&L'}}, PLOTLY_CFG);
}

async function loadEquityChart() {
  const data = await apiFetch('/api/journal/equity?limit=500');
  if (!data.length) { $('chart-equity').innerHTML = '<p class="text-slate-500 text-xs mt-4">No account snapshots yet</p>'; return; }
  const times   = data.map(d => d.time);
  const balance = data.map(d => d.balance);
  const equity  = data.map(d => d.equity);
  Plotly.newPlot('chart-equity', [
    {type:'scatter', mode:'lines', name:'Balance', x:times, y:balance,
     line:{color:'#3b82f6', width:2}},
    {type:'scatter', mode:'lines', name:'Equity',  x:times, y:equity,
     line:{color:'#10b981', width:1.5, dash:'dot'}},
  ], {
    ...PLOTLY_LAYOUT,
    margin:{l:60, r:10, t:10, b:40},
    showlegend:true,
    legend:{orientation:'h', x:0, y:1.1, font:{size:10}},
    yaxis:{...PLOTLY_LAYOUT.yaxis, title:'USD'},
  }, PLOTLY_CFG);
}

// ── Setup tags ────────────────────────────────────────────────────────────────
async function loadTags() {
  setupTags = await apiFetch('/api/journal/tags');
}

function tagsDropdown(currentTag, tradeId) {
  const opts = ['<option value="">— no tag —</option>']
    .concat(setupTags.map(t =>
      `<option value="${t.name}" ${t.name===currentTag?'selected':''}>${t.name}</option>`
    ));
  return `<select class="text-xs" onchange="patchTrade('${tradeId}', 'setup_tag', this.value)">${opts.join('')}</select>`;
}

// ── Trade list ────────────────────────────────────────────────────────────────
async function loadTrades(page) {
  if (page) currentPage = page;
  const sym    = $('f-symbol').value;
  const dir    = $('f-dir').value;
  const ses    = $('f-session').value;
  const src    = $('f-source').value;
  const status = $('f-status').value;

  const params = new URLSearchParams({page: currentPage, per_page: 50});
  if (sym)    params.append('symbol',    sym);
  if (dir)    params.append('direction', dir);
  if (ses)    params.append('session',   ses);
  if (src)    params.append('source',    src);
  if (status) params.append('status',    status);

  const res = await apiFetch('/api/journal/trades?' + params.toString());
  totalPages = res.pages || 1;

  $('page-info').textContent = `Page ${currentPage} / ${totalPages} — ${res.total} trades`;
  $('btn-prev').disabled = currentPage <= 1;
  $('btn-next').disabled = currentPage >= totalPages;

  if (!res.items.length) {
    $('trades-body').innerHTML = '<tr><td colspan="13" class="text-slate-500 text-center py-6">No trades found</td></tr>';
    return;
  }

  const offset = (currentPage - 1) * 50;
  $('trades-body').innerHTML = res.items.map((t, i) => `
    <tr>
      <td class="text-slate-500">${offset + i + 1}</td>
      <td class="font-mono font-semibold">${t.symbol}</td>
      <td class="${t.direction==='BUY'?'text-emerald-400':'text-red-400'} font-bold">${t.direction || '—'}</td>
      <td><span class="tag-badge ${t.trade_source==='ea'?'source-ea':'source-manual'}">${t.trade_source || '—'}</span></td>
      <td class="font-mono text-xs">${fmt(t.entry_price,5)}<br><span class="text-slate-500">${fmtDate(t.entry_time)}</span></td>
      <td class="font-mono text-xs">${t.exit_price ? fmt(t.exit_price,5) : '—'}<br><span class="text-slate-500">${fmtDate(t.exit_time)}</span></td>
      <td class="${pnlClass(t.profit_loss)}">${t.profit_loss !== null ? (t.profit_loss>=0?'+':'')+fmt(t.profit_loss) : '—'}</td>
      <td>${t.risk_reward_actual !== null ? fmt(t.risk_reward_actual) : '—'}</td>
      <td>${t.trading_session || '—'}</td>
      <td class="${outcomeClass(t.outcome)}">${t.outcome || t.status || '—'}</td>
      <td class="text-xs text-slate-400">${t.exit_reason || '—'}</td>
      <td>${tagsDropdown(t.setup_tag, t.id)}</td>
      <td><input type="text" class="text-xs w-32" placeholder="notes…"
        value="${(t.journal_notes || '').replace(/"/g,'&quot;')}"
        onblur="patchTrade('${t.id}', 'journal_notes', this.value)" /></td>
    </tr>
  `).join('');
}

function changePage(delta) {
  const np = currentPage + delta;
  if (np >= 1 && np <= totalPages) loadTrades(np);
}

// ── Patch (annotate) a trade ──────────────────────────────────────────────────
async function patchTrade(id, field, value) {
  try {
    const headers = {'Content-Type':'application/json'};
    const token = getToken();
    if (token) headers['Authorization'] = 'Bearer ' + token;
    const r = await fetch(`${API}/api/journal/trades/${id}`, {
      method:'PATCH', headers,
      body: JSON.stringify({[field]: value}),
    });
    if (r.status === 401) { clearToken(); showAuthModal(); }
  } catch(e) {
    console.error('Patch failed', e);
  }
}

// ── Populate symbol filter from loaded trades ─────────────────────────────────
async function populateSymbolFilter() {
  try {
    const data = await apiFetch('/api/journal/analysis/symbols');
    const sel = $('f-symbol');
    data.forEach(s => {
      const o = document.createElement('option');
      o.value = o.textContent = s.symbol;
      sel.appendChild(o);
    });
  } catch(_) {}
}

// ── Main refresh ──────────────────────────────────────────────────────────────
async function refresh() {
  await Promise.all([
    loadStats(),
    loadLatestSnapshot(),
    loadOpen(),
    loadSessionChart(),
    loadHourChart(),
    loadSetupChart(),
    loadEquityChart(),
  ]);
  await loadTrades(currentPage);
}

// ── Init ──────────────────────────────────────────────────────────────────────
async function initDashboard() {
  // Show user info in header
  try {
    const me = await apiFetch('/api/auth/me');
    window.userProfile = me;
    $('hdr-user').textContent = me.username || me.email;
    $('btn-logout').style.display = '';
    $('btn-settings').style.display = '';

    // Tier badge
    if (me.subscription) {
      const tierEl = $('hdr-tier');
      const t = me.subscription.tier.toUpperCase();
      tierEl.textContent = t;
      tierEl.style.display = '';
      tierEl.style.background = t === 'AUTOPILOT' ? '#1e3a5f' : t === 'PRO' ? '#2d1b4e' : '#1f2937';
      tierEl.style.color = t === 'AUTOPILOT' ? '#60a5fa' : t === 'PRO' ? '#a78bfa' : '#94a3b8';
    }
  } catch(e) { return; } // 401 handled by apiFetch → auth modal shown

  await loadTags();
  await populateSymbolFilter();
  await refresh();
  setInterval(loadOpen, 30_000);
}

// ── Settings modal ─────────────────────────────────────────────────────────
async function openSettings() {
  try {
    const s = await apiFetch('/api/settings');
    $('set-drawdown').value = s.max_daily_drawdown_pct;
    $('set-consec').value = s.max_consecutive_losses;
    $('set-lot').value = s.max_lot_size;
    $('set-positions').value = s.max_open_positions;
    $('set-daily-trades').value = s.max_daily_trades;
    $('set-pause').value = s.pause_on_rule_breach ? 'true' : 'false';
    $('set-tg-chat').value = s.telegram_chat_id || '';
    $('set-tg-enabled').value = s.telegram_enabled ? 'true' : 'false';
    $('set-n-open').checked = s.notify_on_trade_open;
    $('set-n-close').checked = s.notify_on_trade_close;
    $('set-n-summary').checked = s.notify_on_daily_summary;
    $('set-n-risk').checked = s.notify_on_risk_breach;
    $('set-mt5-login').value = s.mt5_login || '';
    $('set-mt5-server').value = s.mt5_server || '';
    $('set-mt5-mode').value = s.mt5_mode;
    $('settings-saved').style.display = 'none';
    $('settings-error').style.display = 'none';
    $('settings-modal').style.display = 'flex';
  } catch(e) { console.error('Failed to load settings', e); }
}

function closeSettings() { $('settings-modal').style.display = 'none'; }

async function saveSettings() {
  const body = {
    max_daily_drawdown_pct: parseFloat($('set-drawdown').value),
    max_consecutive_losses: parseInt($('set-consec').value),
    max_lot_size: parseFloat($('set-lot').value),
    max_open_positions: parseInt($('set-positions').value),
    max_daily_trades: parseInt($('set-daily-trades').value),
    pause_on_rule_breach: $('set-pause').value === 'true',
    telegram_chat_id: $('set-tg-chat').value || null,
    telegram_enabled: $('set-tg-enabled').value === 'true',
    notify_on_trade_open: $('set-n-open').checked,
    notify_on_trade_close: $('set-n-close').checked,
    notify_on_daily_summary: $('set-n-summary').checked,
    notify_on_risk_breach: $('set-n-risk').checked,
    mt5_login: parseInt($('set-mt5-login').value) || null,
    mt5_server: $('set-mt5-server').value || null,
    mt5_mode: $('set-mt5-mode').value,
  };
  try {
    const headers = {'Content-Type':'application/json'};
    const token = getToken();
    if (token) headers['Authorization'] = 'Bearer ' + token;
    const r = await fetch(API + '/api/settings', {method:'PATCH', headers, body:JSON.stringify(body)});
    if (r.status === 401) { clearToken(); showAuthModal(); return; }
    if (!r.ok) { const d = await r.json(); const e = $('settings-error'); e.textContent = d.detail||'Save failed'; e.style.display=''; return; }
    $('settings-saved').style.display = '';
    setTimeout(() => $('settings-saved').style.display = 'none', 2000);
  } catch(e) { console.error('Save failed', e); }
}

(async () => {
  const token = getToken();
  if (!token) { showAuthModal(); return; }
  // Validate existing token
  await initDashboard();
})();
</script>
</body>
</html>
"""


@journal_router.get("/", response_class=HTMLResponse, include_in_schema=False)
async def dashboard():
    """Trade Journal dashboard — single-page application."""
    return HTMLResponse(content=_DASHBOARD_HTML)
