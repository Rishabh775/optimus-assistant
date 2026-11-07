/**
 * Soundwave WhatsApp bridge.
 *
 * Links as a WhatsApp Web device (QR scan), forwards allow-listed incoming
 * messages to Soundwave's agent and returns the reply. Also exposes
 * POST /send so Soundwave can push notifications into WhatsApp.
 *
 * ⚠️ Unofficial client: violates WhatsApp ToS; use a secondary number if
 * the primary one matters to you.
 *
 * Env (loaded from ~/jarvis/.env by systemd):
 *   JARVIS_APP_TOKEN                 auth for bridge -> Soundwave
 *   JARVIS_WHATSAPP_BRIDGE_TOKEN     auth for Soundwave -> bridge /send
 *   JARVIS_WHATSAPP_ALLOW            comma-separated phone numbers allowed
 *                                    to chat (empty = self-chat only)
 *   SOUNDWAVE_URL                    default http://127.0.0.1:8080
 *   BRIDGE_PORT                      default 8090
 */
const express = require('express');
const qrcode = require('qrcode');
const fs = require('fs');
const path = require('path');
const { Client, LocalAuth } = require('whatsapp-web.js');

const PORT = parseInt(process.env.BRIDGE_PORT || '8090', 10);
const BRIDGE_TOKEN = process.env.JARVIS_WHATSAPP_BRIDGE_TOKEN || '';
const SOUNDWAVE_URL = (process.env.SOUNDWAVE_URL || 'http://127.0.0.1:8080')
  .replace(/\/+$/, '');
const SOUNDWAVE_TOKEN = process.env.JARVIS_APP_TOKEN || '';
const ALLOW = (process.env.JARVIS_WHATSAPP_ALLOW || '')
  .split(',').map((s) => s.replace(/\D/g, '')).filter(Boolean);
const AUTH_DIR = process.env.AUTH_DIR || path.join(__dirname, '.auth');
const QR_FILE = path.join(__dirname, 'qr.png');

let ready = false;
let me = null;
let lastQR = null;
const sentIds = new Set();

const client = new Client({
  authStrategy: new LocalAuth({ dataPath: AUTH_DIR }),
  puppeteer: {
    headless: true,
    executablePath: process.env.PUPPETEER_EXECUTABLE_PATH || undefined,
    args: ['--no-sandbox', '--disable-setuid-sandbox', '--disable-dev-shm-usage'],
  },
});

client.on('qr', async (qr) => {
  lastQR = qr;
  try {
    await qrcode.toFile(QR_FILE, qr, { width: 512, margin: 2 });
  } catch (e) { /* ignore */ }
  console.log(`QR ready: ${QR_FILE} — WhatsApp > Linked devices > Link a device`);
});

client.on('ready', () => {
  ready = true;
  me = client.info.wid._serialized;
  console.log(`WhatsApp connected as ${me}`);
});

client.on('authenticated', () => console.log('WhatsApp authenticated'));
client.on('auth_failure', (m) => console.error('WhatsApp auth failure:', m));
client.on('disconnected', (r) => {
  ready = false;
  console.error('WhatsApp disconnected:', r);
});

const digits = (s) => String(s || '').replace(/\D/g, '');
const jid = (n) => `${digits(n)}@c.us`;

async function sendText(to, text) {
  if (!ready) throw new Error('whatsapp not ready');
  const target = to ? jid(to) : client.info.wid._serialized; // self-chat
  const chunks = String(text || '').match(/[\s\S]{1,3500}/g) || [''];
  for (const chunk of chunks) {
    const sent = await client.sendMessage(target, chunk);
    const id = sent && sent.id && sent.id._serialized;
    if (id) {
      sentIds.add(id);
      if (sentIds.size > 1000) sentIds.clear();
    }
  }
}

async function askSoundwave(sender, name, text) {
  const res = await fetch(`${SOUNDWAVE_URL}/api/whatsapp/inbound`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      Authorization: `Bearer ${SOUNDWAVE_TOKEN}`,
    },
    body: JSON.stringify({ from: sender, name, text }),
    signal: AbortSignal.timeout(180000),
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.detail || `soundwave ${res.status}`);
  return data.reply || '(no reply)';
}

client.on('message_create', async (msg) => {
  try {
    if (msg.isStatus || msg.from.endsWith('@g.us')) return;
    const ownId = client.info.wid._serialized;
    const sender = digits(msg.from.split('@')[0]);
    const selfNumber = digits(client.info.wid.user || ownId.split('@')[0]);

    if (msg.fromMe) {
      // Ignore messages this bridge sent; treat YOUR self-chat messages
      // (typed on the phone) as commands to Soundwave.
      if (sentIds.has(msg.id._serialized)) return;
      if (msg.to !== ownId) return;
    } else if (ALLOW.length) {
      if (!ALLOW.includes(sender)) return;   // stranger — ignore
    } else if (sender !== selfNumber) {
      return;                                // default: self-chat only
    }

    const text = (msg.body || '').trim();
    if (!text) {
      await sendText(sender, 'Text messages only for now.');
      return;
    }
    const name = (msg._data && msg._data.notifyName) || '';
    const reply = await askSoundwave(sender, name, text);
    await sendText(sender, reply);
  } catch (e) {
    console.error('inbound error:', e.message || e);
    try { await sendText(digits(msg.from.split('@')[0]),
                         `Soundwave error: ${e.message || e}`); } catch (_) {}
  }
});

// ---- HTTP API for Soundwave ------------------------------------------------
const app = express();
app.use(express.json({ limit: '256kb' }));

function auth(req, res) {
  if (!BRIDGE_TOKEN) return true;
  if (req.get('authorization') === `Bearer ${BRIDGE_TOKEN}`) return true;
  res.status(403).json({ error: 'bad token' });
  return false;
}

app.get('/health', (req, res) => res.json({ ok: true, ready, me }));
app.get('/qr', async (req, res) => {
  if (ready) return res.json({ ready: true, me });
  if (!lastQR) return res.status(404).json({ error: 'no QR yet' });
  res.type('png').send(await qrcode.toBuffer(lastQR, { width: 512, margin: 2 }));
});
app.post('/send', async (req, res) => {
  if (!auth(req, res)) return;
  try {
    await sendText(req.body.to, req.body.text);
    res.json({ ok: true });
  } catch (e) {
    res.status(502).json({ error: String(e.message || e) });
  }
});
app.get('/qr.png', (req, res) => {
  if (!fs.existsSync(QR_FILE)) return res.status(404).end();
  res.type('png').send(fs.readFileSync(QR_FILE));
});

app.listen(PORT, '127.0.0.1', () => {
  console.log(`WhatsApp bridge on http://127.0.0.1:${PORT}`);
});

client.initialize();
