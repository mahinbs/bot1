// =============================================================================
//  Momentum-First (Multi-Horizon Regime) — for TradingSmart.AI / TS·AI Terminal
//
//  This is the same architectural idea as the Python bot in this repo, ported
//  into the simulator's single-onBar Web Worker contract.
//
//  Before any entry trigger fires, the bot builds an *understanding* of where
//  price is across THREE horizons (short / mid / long) on the active timeframe.
//  Each horizon score blends four ATR-normalised, tanh-squashed signals:
//
//    1. ROC                        (raw price displacement)
//    2. EMA alignment + slope      (trend structure)
//    3. MACD histogram             (acceleration)
//    4. Donchian position          (where in the recent range we sit)
//
//  Horizon scores are weighted (mid + long dominate over short) into a final
//  number. The regime is then classified using both score magnitude AND
//  whether all three horizons agree on direction:
//
//    |score| >= 0.55  AND  aligned    →  STRONG_UP / STRONG_DOWN
//    |score| >= 0.20                  →  UP / DOWN
//    otherwise                        →  NEUTRAL  (the bot will not trade)
//
//  Only when the regime is non-neutral does the entry trigger run: a 20-bar
//  Donchian breakout in the regime direction. If the bot does not understand
//  the market (chop, mixed horizons), it refuses to trade — that is the whole
//  point of "momentum first".
//
//  ─── How to use it in the simulator ────────────────────────────────────────
//  1. Open https://tradesimulator-one.vercel.app
//  2. + NEW BOT → GITHUB tab
//  3. Paste:  https://github.com/mahinbs/bot1
//  4. Pick:   tradesim/momentum_first.js
//  5. VALIDATE → DEPLOY
//  6. Configure investment / lot / target / stop in the bot card.
//
//  The simulator's worker harness exposes these helpers as globals (no imports
//  needed):  sma, ema, rsi, highest, lowest, crossover, crossunder, macd,
//  bollinger, stoch, atr, vwap.
// =============================================================================

function onBar(bar, ctx) {
  var C = ctx.closes, H = ctx.highs, L = ctx.lows;
  if (C.length < 130) return;
  var last = C[C.length - 1];

  function tanh(x) {
    if (!isFinite(x)) return 0;
    var p = Math.exp(x), n = Math.exp(-x);
    return (p - n) / (p + n);
  }
  function clamp(x, a, b) { return Math.max(a, Math.min(b, x)); }

  function scoreHorizon(n, fastP, slowP) {
    var a = atr(H, L, C, 14);
    if (a == null) return null;
    var atrPct = a / last;
    if (!(atrPct > 0)) return null;

    // 1) ROC normalised by typical move size.
    var past = C[C.length - 1 - n];
    var rocVal = past > 0 ? (last / past - 1) : 0;
    var rocScore = tanh(rocVal / (atrPct * 5));

    // 2) EMA alignment + slope.
    var fast = ema(C, fastP);
    var slow = ema(C, slowP);
    if (fast == null || slow == null) return null;
    var align = fast > slow ? 1 : -1;
    var fastPrev = ema(C.slice(0, -5), fastP);
    var slope = (fastPrev > 0) ? (fast / fastPrev - 1) : 0;
    var emaScore = align * Math.abs(tanh(slope / atrPct));

    // 3) MACD histogram, normalised by ATR.
    var m = macd(C, 12, 26, 9);
    var macdScore = (m && a > 0) ? tanh(m.histogram / (a * 0.5)) : 0;

    // 4) Donchian position over n bars: where in the range are we?
    var u = highest(H.slice(0, -1), n);
    var lo = lowest(L.slice(0, -1), n);
    var donchScore = 0;
    if (u != null && lo != null && u > lo) {
      donchScore = clamp((last - (u + lo) / 2) / ((u - lo) / 2), -1, 1);
    }

    return clamp(0.30 * rocScore + 0.30 * emaScore + 0.20 * macdScore + 0.20 * donchScore, -1, 1);
  }

  var sShort = scoreHorizon(20,  9, 21);   // recent
  var sMid   = scoreHorizon(60, 21, 55);   // swing
  var sLong  = scoreHorizon(120, 34, 89);  // position
  if (sShort == null || sMid == null || sLong == null) return;

  var score = 0.25 * sShort + 0.40 * sMid + 0.35 * sLong;
  var sigs = [sShort, sMid, sLong].filter(function (s) { return Math.abs(s) > 0.05; });
  var aligned = sigs.length === 3 && sigs.every(function (s) { return Math.sign(s) === Math.sign(score); });

  // ─── Regime classification (the "understanding" step) ───────────────────
  var STRONG = 0.55, WEAK = 0.20;
  var regime = 'NEUTRAL';
  if (Math.abs(score) >= STRONG && aligned) regime = score > 0 ? 'STRONG_UP' : 'STRONG_DOWN';
  else if (Math.abs(score) >= WEAK)         regime = score > 0 ? 'UP' : 'DOWN';

  ctx.state.regime = regime;
  ctx.state.score  = score;

  if (regime === 'NEUTRAL') return;  // refuse to trade what we do not understand

  // ─── Entry trigger (gated by regime) ────────────────────────────────────
  var upper = highest(H.slice(0, -1), 20);
  var lower = lowest(L.slice(0, -1), 20);
  if (upper == null || lower == null) return;

  var wantSide = null;
  if ((regime === 'UP'   || regime === 'STRONG_UP')   && last > upper) wantSide = 'BUY';
  if ((regime === 'DOWN' || regime === 'STRONG_DOWN') && last < lower) wantSide = 'SELL';
  if (!wantSide) return;

  // De-dupe: do not re-fire the same side back-to-back without a regime flip.
  if (ctx.state.lastSide === wantSide) return;
  ctx.state.lastSide = wantSide;

  ctx.log(wantSide + ' [' + regime + '] score=' + score.toFixed(2) + ' aligned=' + aligned);
  if (wantSide === 'BUY') ctx.buy();
  else                    ctx.sell();
}
