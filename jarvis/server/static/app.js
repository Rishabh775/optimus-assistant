(() => {
  const $ = s => document.querySelector(s);
  const log = $('#log');
  let ws = null;
  let sid = null;          // current session id
  let busy = false;

  const scroll = () => { log.scrollTop = log.scrollHeight; };

  const bubble = (role) => {
    const d = document.createElement('div');
    d.className = 'msg ' + role;
    log.appendChild(d);
    scroll();
    return d;
  };

  const plain = (t, role = 'user') => { bubble(role).textContent = t; };

  const note = (t) => { bubble('note').textContent = t; };

  const renderMd = (el, src) => {
    window._mdsrc = src || '';
    if (window.marked && window.DOMPurify)
      el.innerHTML = DOMPurify.sanitize(marked.parse(window._mdsrc));
    else
      el.textContent = window._mdsrc;
  };

  let typing = null;
  const showTyping = () => { typing = bubble('typing'); typing.textContent = 'working on it...'; };
  const stopTyping = () => { if (typing && typing.parentNode) typing.remove(); typing = null; };

  const setDot = (state) => { $('#dot').className = 'dot' + (state ? ' ' + state : ''); };

  const clock = (ts) => {
    if (!ts) return '';
    const d = new Date(ts * 1000), now = new Date();
    if (d.toDateString() === now.toDateString())
      return d.toTimeString().slice(0, 5);
    return d.toISOString().slice(0, 10);
  };

  const headers = (token) => ({
    'Content-Type': 'application/json',
    Authorization: `Bearer ${token}`
  });

  async function postLogin(token) {
    const ctrl = new AbortController();
    const t = setTimeout(() => ctrl.abort(), 10000);
    try {
      return await fetch('/api/login', {
        method: 'POST', headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({token}), signal: ctrl.signal
      });
    } finally {
      clearTimeout(t);
    }
  }

  // ---- session rail ----

  async function loadSessions(token) {
    const list = await fetch('/api/sessions', {headers: headers(token)}).then(r => r.json());
    const box = $('#slist');
    box.innerHTML = '';
    for (const s of list) {
      const row = document.createElement('div');
      row.className = 'srow' + (s.id === sid ? ' active' : '');
      row.dataset.id = s.id;
      row.innerHTML = `
        <span class="t"></span>
        <span class="act">
          <button title="rename">✎</button>
          <button title="delete">🗑</button>
        </span>`;
      const title = s.title || 'untitled';
      row.querySelector('.t').innerHTML =
        `${title.replace(/[<>&]/g, '')}<small>${clock(s.updated_at)} · ${s.last.replace(/[<>&]/g, '').slice(0, 40)}</small>`;
      row.onclick = (e) => {
        if (e.target.closest('button')) return;
        openSession(s.id, token);
      };
      row.querySelector('[title=rename]').onclick = async () => {
        const t = prompt('Rename chat:', title);
        if (t && t.trim()) {
          await fetch(`/api/sessions/${s.id}`, {
            method: 'PATCH', headers: headers(token),
            body: JSON.stringify({title: t.trim()})
          });
          loadSessions(token);
        }
      };
      row.querySelector('[title=delete]').onclick = async () => {
        if (confirm('Delete this chat?')) {
          await fetch(`/api/sessions/${s.id}`, {method: 'DELETE', headers: headers(token)});
          if (s.id === sid) { sid = null; $('#stitle').textContent = ''; log.innerHTML = ''; }
          loadSessions(token);
        }
      };
      box.appendChild(row);
    }
  }

  async function openSession(id, token) {
    sid = id;
    busy = false;
    log.innerHTML = '';
    rowmark(id);
    const msgs = await fetch(`/api/sessions/${id}/messages`, {headers: headers(token)}).then(r => r.json());
    for (const m of msgs.slice(-40)) {
      if (m.role === 'user') plain(m.content, 'user');
      else renderMd(bubble('assistant'), m.content);
    }
    const s = await fetch('/api/sessions', {headers: headers(token)}).then(r => r.json());
    $('#stitle').textContent = (s.find(x => x.id === id) || {}).title || 'untitled';
    closeRail();
    scroll();
  }

  const rowmark = (id) => {
    document.querySelectorAll('.srow').forEach(r => r.classList.toggle('active', r.dataset.id === id));
  };

  const closeRail = () => { $('#rail').classList.remove('open'); $('#shade').style.display = 'none'; };

  $('#burger').onclick = () => {
    $('#rail').classList.add('open');
    $('#shade').style.display = 'block';
  };
  $('#shade').onclick = closeRail;

  $('#newChat').onclick = async () => {
    const token = localStorage.getItem('sw-token');
    const s = await fetch('/api/sessions', {
      method: 'POST', headers: headers(token),
      body: JSON.stringify({})
    }).then(r => r.json());
    log.innerHTML = '';
    $('#stitle').textContent = 'untitled';
    openSessionNow(s, token);
    function openSessionNow(s, token) {
      sid = s.id;
      rowmark(s.id);
      loadSessions(token);
      closeRail();
    }
  };

  // ---- websocket ----

  let streamBuf = null;   // the in-flight <div> while deltas stream

  function connect(token) {
    setDot('on');
    ws = new WebSocket(`${location.origin.replace(/^http/, 'ws')}/ws?token=${encodeURIComponent(token)}`);
    ws.onmessage = (e) => {
      const m = JSON.parse(e.data);
      if (m.type === 'hello') {
        $('#machine').textContent = m.machine || '';
        note('connected to ' + (m.machine || 'PC') + ' — type below to start');
      }
      else if (m.type === 'delta') {
        if (m.sid !== sid) return;              // other session; reply will re-render
        stopTyping();
        if (!streamBuf) streamBuf = bubble('assistant');
        streamBuf._src = (streamBuf._src || '') + m.text;
        const marked_ = window.marked;
        streamBuf.innerHTML =
          `${window.DOMPurify.sanitize(marked_ ? marked_.parse(streamBuf._src) : streamBuf._src)}<span class="cursor"></span>`;
        scroll();
      }
      else if (m.type === 'reply') {
        stopTyping();
        busy = false;
        if (m.sid === sid) {
          if (streamBuf && streamBuf._src && m.text.startsWith(streamBuf._src))
            renderMd(streamBuf, m.text);
          else
            renderMd(bubble('assistant'), m.text);
          streamBuf = null;
          loadSessions(token);
        }
        setDot('on');
        if (m.sid !== sid) loadSessions(token);
      }
      else if (m.type === 'confirm') {
        stopTyping();
        const card = document.createElement('div');
        card.className = 'confirm';
        card.innerHTML = `
          <div class="reason"></div>
          <div class="row">
            <button class="yes">Confirm</button>
            <button class="no">Deny</button>
          </div>`;
        card.querySelector('.reason').textContent = `${m.reason}\n(tool: ${m.tool})`;
        card.querySelector('.yes').onclick = () => answer(m.id, true, card);
        card.querySelector('.no').onclick = () => answer(m.id, false, card);
        log.appendChild(card);
        scroll();
      }
      else if (m.type === 'note') {
        note(m.text);
      }
      else if (m.type === 'error') {
        resetLogin('Token rejected. Enter a valid token.');
      }
    };
    ws.onclose = () => { setDot(''); stopTyping(); };
    ws.onerror = () => {};
  }

  function answer(id, yes, card) {
    ws.send(JSON.stringify({type: 'confirm', id, answer: yes}));
    card.querySelector('.row').remove();
    card.querySelector('.reason').textContent += `\n→ ${yes ? 'confirmed' : 'denied'}`;
  }

  async function send(text) {
    const token = localStorage.getItem('sw-token');
    plain(text, 'user');
    showTyping();
    if (!sid) {
      const s = await fetch('/api/sessions', {
        method: 'POST', headers: headers(token), body: JSON.stringify({})
      }).then(r => r.json());
      sid = s.id;
      rowmark(sid); /* added below via loadSessions */
      await loadSessions(token);
      $(`#stitle`).textContent = text.slice(0, 38);
    }
    setDot('busy');
    busy = true;
    ws.send(JSON.stringify({type: 'chat', session_id: sid, text}));
  }

  // ---- boot / login ----

  function openChat(token) {
    $('#login').hidden = true;
    $('#app').style.display = 'flex';
    connect(token);
    loadSessions(token).then(() => {
      const first = $('#slist .srow');
      if (first) openSession(first.dataset.id, token);
      else note('No chats yet — send a message to start.');
    });
    $('#bar').addEventListener('submit', (e) => {
      e.preventDefault();
      const v = $('#msg').value.trim();
      if (!v) return;
      if (!ws || ws.readyState !== 1) { note('not connected'); return; }
      if (busy) { note('still working...'); return; }
      $('#msg').value = '';
      send(v);
    });
  }

  function resetLogin(msg) {
    $('#app').style.display = 'none';
    $('#login').hidden = false;
    $('#loginErr').textContent = msg || '';
  }

  $('#go').onclick = async () => {
    try {
      $('#loginErr').textContent = 'Connecting...';
      const token = $('#tok').value.trim();
      const r = await postLogin(token);
      if (!r.ok) throw new Error('token rejected (' + r.status + ')');
      localStorage.setItem('sw-token', token);
      openChat(token);
      $('#loginErr').textContent = '';
    } catch (e) {
      $('#loginErr').textContent = (e.name === 'AbortError')
        ? 'Timed out — phone could not reach the PC over Tailscale.'
        : 'Failed: ' + e.message;
    }
  };
  $('#tok').addEventListener('keydown', (e) => { if (e.key === 'Enter') $('#go').click(); });

  window.onerror = (msg) => {
    if ($('#login').hidden === false)
      $('#loginErr').textContent = 'UI error: ' + msg;
    return false;
  };
  window.addEventListener('unhandledrejection', (e) => {
    const m = (e.reason && e.reason.message) || e.reason || '?';
    note('UI error: ' + m);
  });

  (async () => {
    try {
      const saved = localStorage.getItem('sw-token');
      if (saved) {
        const r = await postLogin(saved);
        if (r.ok) openChat(saved);
        else localStorage.removeItem('sw-token');
      }
    } catch (e) {
      $('#loginErr').textContent = 'Auto-connect failed, enter token again.';
    }
  })();

  if ('serviceWorker' in navigator) {
    navigator.serviceWorker.register('/static/sw.js').catch(() => {});
  }
})();
