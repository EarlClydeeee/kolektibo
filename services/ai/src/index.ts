import 'dotenv/config'
import express from 'express'
import cors from 'cors'
import OpenAI from 'openai'
import { z } from 'zod'
import { execFile } from 'node:child_process'
import { promisify } from 'node:util'
import { authRouter } from './auth'
import { walletRouter } from './wallet'
import { deployPool, mintUsdc, sdkBackendConfigured } from './chain'
import { startIndexer } from './indexer'

const pExecFile = promisify(execFile)

// ── Backend mode: SDK (new, production-ready) vs CLI (legacy demo fallback) ──
// Set USE_SDK_BACKEND=0 to fall back to the CLI-shelling path (e.g. if secrets
// aren't set yet). Default is SDK when configured, CLI otherwise.
const USE_SDK = process.env.USE_SDK_BACKEND !== '0' && sdkBackendConfigured()

const app = express()
app.use(cors())
app.use(express.json({ limit: '1mb' }))
app.use('/auth', authRouter)
app.use('/wallet', walletRouter)

const apiKey = process.env.OPENAI_API_KEY
const model = process.env.OPENAI_MODEL || 'gpt-4o-mini'
const client = apiKey ? new OpenAI({ apiKey }) : null

// ── Chain ops: thin wrapper over the Stellar CLI on this machine ──────────────
// Keeps issuer/deployer identities in the local CLI keystore — no secrets in code,
// env, transcript, or the browser bundle. Fine for a local demo backend.
const CHAIN = {
  bin: process.env.STELLAR_BIN || 'C:/Program Files (x86)/Stellar CLI/stellar.exe',
  network: process.env.STELLAR_NETWORK || 'testnet',
  usdcSac: process.env.USDC_SAC_ID || '',
  usdcIssuer: process.env.USDC_ISSUER || '',
  issuer: process.env.ISSUER_IDENTITY || 'kolektibo-usdc-issuer',
  deployer: process.env.DEPLOYER_IDENTITY || 'kolektibo-deployer',
  wasmPath: process.env.TREASURY_WASM_PATH || '',
  rpcUrl: process.env.SOROBAN_RPC_URL || 'https://soroban-testnet.stellar.org',
  passphrase: 'Test SDF Network ; September 2015',
  categories: ['Equipment', 'Venue', 'Refreshments'],
  limits: ['50000000000', '30000000000', '15000000000'], // 5000/3000/1500 USDC * 1e7
}

async function stellar(args: string[]): Promise<string> {
  // Strip RPC/passphrase env vars so `--network testnet` fully defines the network
  // (a leaked SOROBAN_RPC_URL makes the CLI demand a matching passphrase).
  const env = { ...process.env }
  delete env.SOROBAN_RPC_URL
  delete env.STELLAR_RPC_URL
  delete env.STELLAR_NETWORK_PASSPHRASE
  delete env.SOROBAN_NETWORK_PASSPHRASE
  const { stdout } = await pExecFile(CHAIN.bin, args, {
    maxBuffer: 16 * 1024 * 1024,
    env,
  })
  return stdout.trim()
}

function chainErr(e: unknown): string {
  const anyE = e as { stderr?: string; message?: string }
  return (anyE?.stderr || anyE?.message || 'chain op failed').toString()
}

// ── Policy schema: the on-chain-ready shape the contract will be initialized with ──
const PolicySchema = z.object({
  currency: z.string().default('USDC'),
  dues: z
    .object({
      amount: z.coerce.number(),
      period: z.enum(['monthly', 'weekly', 'once']),
    })
    .nullable(),
  categories: z.array(
    z.object({
      name: z.string(),
      monthlyLimit: z.coerce.number().nullable(),
    }),
  ),
  approval: z.object({
    threshold: z.coerce.number().int().min(1),
    of: z.coerce.number().int().min(1),
  }),
  summary: z.string(),
})

const RULES_SYSTEM = `You convert a community group's plain-language money rules into a strict JSON policy for an on-chain group treasury.

Return ONLY a JSON object with EXACTLY this shape:
{
  "currency": string,                      // asset the fund settles in; default "USDC"
  "dues": { "amount": number, "period": "monthly"|"weekly"|"once" } | null,
  "categories": [ { "name": string, "monthlyLimit": number | null } ],
  "approval": { "threshold": number, "of": number },
  "summary": string                        // one warm, plain-language sentence restating the rules
}

Rules:
- Amounts are plain numbers: strip currency symbols (₱, $) and commas. "₱5,000" -> 5000.
- "2 of 3 officers" -> approval.threshold=2, approval.of=3. If unspecified, default threshold=2, of=3.
- monthlyLimit is null when a category has no stated cap.
- If no dues are mentioned, dues=null.
- Keep category names short and title-cased (e.g. "Equipment", "Venue", "Refreshments").`

const ASK_SYSTEM = `You are Kolektibo, the AI treasurer for a community pooled fund in the Philippines.
Answer the member's question in a warm, trustworthy, PLAIN-language tone — like a helpful barangay treasurer.
Use ONLY the on-chain state provided (balance, contributions, spend history, policy). Never invent numbers.
Amounts are in Philippine pesos, written like ₱1,200. Keep answers to 2-4 short sentences.
If the question is "can we afford X", compare X to the current balance and any category limit, then say yes/no clearly.
If the state doesn't contain the answer, say so honestly and suggest what's needed.`

app.get('/health', (_req, res) => {
  res.json({ ok: true, model, hasKey: Boolean(apiKey) })
})

app.post('/rules', async (req, res) => {
  if (!client) return res.status(500).send('OPENAI_API_KEY not set on the AI service')
  const text = String(req.body?.text ?? '').trim()
  if (!text) return res.status(400).send('Missing "text"')
  try {
    const completion = await client.chat.completions.create({
      model,
      temperature: 0,
      response_format: { type: 'json_object' },
      messages: [
        { role: 'system', content: RULES_SYSTEM },
        { role: 'user', content: text },
      ],
    })
    const raw = completion.choices[0]?.message?.content ?? '{}'
    const policy = PolicySchema.parse(JSON.parse(raw))
    res.json({ policy })
  } catch (e) {
    console.error('[/rules]', e)
    res.status(422).send(e instanceof Error ? e.message : 'Failed to parse rules')
  }
})

app.post('/ask', async (req, res) => {
  if (!client) return res.status(500).send('OPENAI_API_KEY not set on the AI service')
  const question = String(req.body?.question ?? '').trim()
  const state = req.body?.state ?? {}
  if (!question) return res.status(400).send('Missing "question"')
  try {
    const completion = await client.chat.completions.create({
      model,
      temperature: 0.2,
      messages: [
        { role: 'system', content: ASK_SYSTEM },
        {
          role: 'user',
          content: `On-chain state (JSON):\n${JSON.stringify(state, null, 2)}\n\nQuestion: ${question}`,
        },
      ],
    })
    res.json({ answer: completion.choices[0]?.message?.content ?? '' })
  } catch (e) {
    console.error('[/ask]', e)
    res.status(500).send(e instanceof Error ? e.message : 'AI error')
  }
})

// Public chain config for the client (no secrets).
app.get('/config', (_req, res) => {
  res.json({
    network: CHAIN.network,
    usdcSac: CHAIN.usdcSac,
    usdcIssuer: CHAIN.usdcIssuer,
    rpcUrl: CHAIN.rpcUrl,
    passphrase: CHAIN.passphrase,
    friendbotUrl: 'https://friendbot.stellar.org',
    categories: CHAIN.categories,
    limits: CHAIN.limits,
    threshold: 2,
    configured: Boolean(CHAIN.usdcSac && CHAIN.wasmPath),
  })
})

// Mint test USDC to an address (the address must already trust USDC).
app.post('/faucet', async (req, res) => {
  const address = String(req.body?.address ?? '').trim()
  const amount = String(req.body?.amount ?? '100000000000') // 10,000 USDC (7 decimals)
  if (!address.startsWith('G')) return res.status(400).send('Invalid Stellar address')
  try {
    if (USE_SDK) {
      // ── SDK path (Backend v1): uses stellar-sdk directly — deployable anywhere ──
      const txHash = await mintUsdc(address, BigInt(amount))
      res.json({ ok: true, txHash })
    } else {
      // ── CLI fallback (legacy demo): shells out to the local stellar.exe ──
      await stellar([
        'contract', 'invoke', '--id', CHAIN.usdcSac,
        '--source', CHAIN.issuer, '--network', CHAIN.network,
        '--', 'mint', '--to', address, '--amount', amount,
      ])
      res.json({ ok: true })
    }
  } catch (e) {
    console.error('[/faucet]', chainErr(e))
    res.status(500).send(chainErr(e))
  }
})

// Deploy + initialize a fresh treasury for the browser's officer public keys.
app.post('/pool/create', async (req, res) => {
  const officers: string[] = Array.isArray(req.body?.officers) ? req.body.officers : []
  const threshold = Number(req.body?.threshold ?? 2)
  if (officers.length < 2 || officers.some((o) => !String(o).startsWith('G'))) {
    return res.status(400).send('Provide at least 2 officer public keys')
  }
  try {
    if (USE_SDK) {
      // ── SDK path (Backend v1): uses stellar-sdk directly — deployable anywhere ──
      const contractId = await deployPool({
        officers,
        threshold,
        categories: CHAIN.categories,
        limits: CHAIN.limits.map((l) => BigInt(l)),
      })
      res.json({ ok: true, contractId })
    } else {
      // ── CLI fallback (legacy demo): shells out to the local stellar.exe ──
      const contractId = await stellar([
        'contract', 'deploy', '--wasm', CHAIN.wasmPath,
        '--source', CHAIN.deployer, '--network', CHAIN.network,
      ])
      await stellar([
        'contract', 'invoke', '--id', contractId,
        '--source', CHAIN.deployer, '--network', CHAIN.network,
        '--', 'initialize',
        '--token', CHAIN.usdcSac,
        '--officers', JSON.stringify(officers),
        '--threshold', String(threshold),
        '--categories', JSON.stringify(CHAIN.categories),
        '--limits', JSON.stringify(CHAIN.limits),
      ])
      res.json({ ok: true, contractId })
    }
  } catch (e) {
    console.error('[/pool/create]', chainErr(e))
    res.status(500).send(chainErr(e))
  }
})

const port = Number(process.env.PORT || 8787)
app.listen(port, () => {
  console.log(`Kolektibo AI service → http://localhost:${port}  (model: ${model}, key: ${apiKey ? 'set' : 'MISSING'})`)
  console.log(`  Chain backend: ${USE_SDK ? 'stellar-sdk (Backend v1)' : 'CLI shelling (legacy)'}`)

  // ── Indexer v0: auto-start alongside the API if INDEXER_AUTOSTART=1 ──────────
  // Alternatively run as a separate process: pnpm indexer
  if (process.env.INDEXER_AUTOSTART === '1') {
    startIndexer()
  }
})
